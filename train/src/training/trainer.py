import os
import torch
import torch.nn as nn
import inspect
from deepspeed.utils import safe_get_full_grad

from transformers import Trainer, TrainerCallback
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
try:
    from transformers.trainer_pt_utils import get_parameter_names
except Exception:
    def get_parameter_names(model, forbidden_layer_types):
        result = []
        for name, child in model.named_children():
            child_params = get_parameter_names(child, forbidden_layer_types)
            result += [f"{name}.{n}" for n in child_params]

        result += list(model._parameters.keys())
        result = [
            n for n in result
            if not any(isinstance(model.get_submodule(n.rsplit(".", 1)[0]), t) for t in forbidden_layer_types)
            or "." not in n
        ]
        return result

try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
except Exception:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
import safetensors
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenTrainer(Trainer):

    def __init__(self, processor, *args, **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)
        print(f"[DEBUG Trainer.__init__] model device after super().__init__: {next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'N/A'}")
        self.processor = processor
        
    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
        print("I got it! Maybe for future usage")
        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def _is_qwen3_vl(self):
        target_model = self.model
        if hasattr(self.model, "get_base_model"):
            try:
                target_model = self.model.get_base_model()
            except Exception:
                target_model = self.model

        model_type = ""
        if hasattr(target_model, "config") and target_model.config is not None:
            model_type = str(getattr(target_model.config, "model_type", "")).lower()

        class_name = target_model.__class__.__name__.lower()
        return ("qwen3" in model_type and "vl" in model_type) or ("qwen3" in class_name and "vl" in class_name)

    def _normalize_qwen3_inputs(self, inputs):
        # Qwen3-VL native forward expects a 2D attention mask for rope indexing.
        # Our dataset may provide custom 4D masks for qwen2.5-specific training paths.
        if not self._is_qwen3_vl():
            return inputs

        if "sequence_index" in inputs:
            inputs.pop("sequence_index", None)

        attention_mask = inputs.get("attention_mask", None)
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
            input_ids = inputs.get("input_ids", None)
            if isinstance(input_ids, torch.Tensor):
                pad_token_id = None
                if hasattr(self, "processor") and self.processor is not None:
                    tok = getattr(self.processor, "tokenizer", None)
                    if tok is not None:
                        pad_token_id = getattr(tok, "pad_token_id", None)

                if pad_token_id is None:
                    new_mask = torch.ones_like(input_ids, dtype=torch.long)
                else:
                    new_mask = (input_ids != pad_token_id).to(dtype=torch.long)

                inputs["attention_mask"] = new_mask

        return inputs

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)

        try:
            target_model = self.model
            if hasattr(self.model, "get_base_model"):
                try:
                    target_model = self.model.get_base_model()
                except Exception:
                    target_model = self.model

            signature = inspect.signature(target_model.forward)
            params = signature.parameters

            # PEFT wrappers often expose `forward(*args, **kwargs)`.
            # In that case we must not filter, otherwise critical keys like
            # `input_ids` may be dropped and downstream embedding() receives None.
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return self._normalize_qwen3_inputs(inputs)

            supported_keys = set(params.keys())
            filtered_inputs = {k: v for k, v in inputs.items() if k in supported_keys}

            # Safety fallback: never return a dict without core model inputs.
            core_keys = {"input_ids", "inputs_embeds", "pixel_values", "labels"}
            if len(filtered_inputs) == 0 or (core_keys & set(filtered_inputs.keys()) == set()):
                return self._normalize_qwen3_inputs(inputs)

            return self._normalize_qwen3_inputs(filtered_inputs)
        except Exception:
            # Fall back to default behavior if signature inspection fails.
            return self._normalize_qwen3_inputs(inputs)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],# and n not in projection_parameters
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],# and n not in projection_parameters
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        
        self.optimizer.param_groups
        opt = self.optimizer
        print(f"[Optimizer] {opt.__class__.__name__}")
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"  - group {i}: lr={param_group['lr']}, "
                      f"betas={param_group.get('betas', 'N/A')}, "
                      f"eps={param_group.get('eps', 'N/A')}, "
                      f"weight_decay={param_group.get('weight_decay', 'N/A')}")
        
        return self.optimizer
    

    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    
    
class UnfreezeLoRACallback(TrainerCallback):
    def __init__(self, unfreeze_step):
        self.unfreeze_step = unfreeze_step
        self.lora_lr = None
        self.trainer = None
    def set_trainer(self, trainer):
        self.trainer = trainer
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            for n, p in kwargs["model"].named_parameters():
                if "lora_" in n:
                    p.requires_grad = False
            kwargs["model"].print_trainable_parameters()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.unfreeze_step:
            for n, p in kwargs["model"].named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
            kwargs["model"].print_trainable_parameters()
            