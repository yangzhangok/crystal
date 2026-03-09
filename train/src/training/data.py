import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image,ImageFilter
from transformers import AutoImageProcessor
import re
import numpy as np
import cv2
from torchvision import transforms
import random
import math

from .params import DataArguments
from .constants import *


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future


    content = {
        "type": "image", 
        "image": image_path,
        "min_pixel": min_pixel,
        "max_pixel": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

def get_comt_data_in_response_livr(response):
    
    pad_token = "<|sam_pad|><|dino_pad|><|depth_pad|><|sd_pad|><|intern_pad|><|pidinet_pad|><|siglip_pad|><|metaclip_pad|>"
    CoT_start = f"<think>{pad_token}</think>"
    response = CoT_start + "<answer> " + response + " </answer>"
    return response
    
def get_comt_data_in_response_vloc(response, DEFAULT_IM_END_TOKEN):
    """
    Format response for VLOC (Visual Location-aware Chain-of-Thought) training mode.
    
    This function creates a more detailed chain-of-thought format with both
    <answer> tags for training. It adds two copies of the answer at different positions
    for specific training objectives.
    
    Args:
        response: The original text response
        DEFAULT_IM_END_TOKEN: End token for image
    
    Returns:
        Formatted response string with CoT and dual answer tokens
    """
    pad_token = "<|sam_pad|><|dino_pad|><|depth_pad|><|sd_pad|><|intern_pad|><|pidinet_pad|><|siglip_pad|><|metaclip_pad|>"

    CoT_start = "<think> Because "
    CoT_start += f"the feature of the image is {pad_token}."
    response = CoT_start + " </think>" + "<answer> " + response + " </answer>"+"\n" + DEFAULT_IM_END_TOKEN +"\n"+"<answer> " + response + " </answer>"

    return response



def create_custom_causal_mask_livr(input_ids, dtype=torch.float16, device='cuda'):
    seq_len = input_ids.shape[0]

    try:
        #import ipdb;ipdb.set_trace()
        idx_sep0 = (torch.where(input_ids == 151652)[0][0]).item()
        idx_sep1 = (torch.where(input_ids == 151653)[0][0]+1).item()
        idx_sep2 = (torch.where(input_ids == 151675)[0][0]).item() #<think>
        idx_sep3= (torch.where(input_ids == 151676)[0][0] + 2).item() #</think>

        sep4_candidates = torch.where(input_ids == 151645)[0]
        if len(sep4_candidates) > 1:
            idx_sep4 = (sep4_candidates[-1] + 2).item()
        else:
            idx_sep4 = seq_len 
        
    except IndexError:
        print("Warning: Special tokens not fully found in LIVR mode, falling back to standard Causal Mask")
        return torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device).triu(1)

    # Block A: Before Img
    range_a = (0, idx_sep0)
    # Block B: Img1
    range_b = (idx_sep0, idx_sep1)
    # Block C: Img2
    range_c = (idx_sep1, idx_sep2)
    range_d = (idx_sep2, idx_sep3)
    range_e = (idx_sep3, idx_sep4)
    
    # Initialize mask with zeros (visible)
    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    # Get minimum value for masking
    min_dtype = torch.finfo(dtype).min

    # Apply standard causal mask (upper triangle = future positions)
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask.masked_fill_(causal_mask, min_dtype)

    # Apply custom blocking rules for LIVR mode
    # Block C cannot attend to Block B
    mask[range_c[0]:range_c[1], range_b[0]:range_b[1]] = min_dtype
    # Block E cannot attend to Block B
    mask[range_e[0]:range_e[1], range_b[0]:range_b[1]] = min_dtype

    # Expand to 4D: (batch, head, seq, seq)
    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


def create_custom_causal_mask(input_ids, dtype=torch.float16, device='cuda'):
    seq_len = input_ids.shape[0]
    
    try:
        #import ipdb;ipdb.set_trace()
        idx_sep0 = (torch.where(input_ids == 151652)[0][0]).item()
        idx_sep1 = (torch.where(input_ids == 151652)[0][1]).item()
        idx_sep2 = idx_sep1+(idx_sep1-idx_sep0)
        idx_sep3 = (torch.where(input_ids == 151676)[0][0] + 1).item() #</think>
        idx_sep4 = (torch.where(input_ids == 151677)[0][1]).item() #<answer>
        
        sep5_candidates = torch.where(input_ids == 151645)[0]
        if len(sep5_candidates) > 1:
            idx_sep5 = (sep5_candidates[-1] + 2).item()
        else:
            idx_sep5 = seq_len 
        
    except IndexError:
        print("Warning: Special tokens not fully found, falling back to standard Causal Mask")
        return torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device).triu(1)

    # 2. Define each block's range [start, end)
    # Block A: Before Img
    range_a = (0, idx_sep0)
    # Block B: Img1
    range_b = (idx_sep0, idx_sep1)
    # Block C: Img2
    range_c = (idx_sep1, idx_sep2)
    # Block D: Think
    range_d = (idx_sep2, idx_sep3)
    # Block E: Answer1
    range_e = (idx_sep3, idx_sep4)
    # Block F: Answer2
    range_f = (idx_sep4, idx_sep5)

    # 3. Initialize Mask
    # Initially all zeros (Visible)
    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    
    # Get minimum value (for masking)
    min_dtype = torch.finfo(dtype).min

    # 4. Apply standard Causal Mask (masking future)
    # Upper triangle (diagonal=1) set to negative infinity
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask.masked_fill_(causal_mask, min_dtype)

    # 5. Apply custom Block masking rules (masking specific history)
    # Logic: mask[Query_Range, Key_Range] = min_dtype
    
    # --- Rule 1: Block B (Img1) can only be attended by F ---
    # Means C, D, E cannot see B
    # C cannot see B
    mask[range_c[0]:range_c[1], range_b[0]:range_b[1]] = min_dtype
    # D cannot see B
    mask[range_d[0]:range_d[1], range_b[0]:range_b[1]] = min_dtype
    # E cannot see B
    mask[range_e[0]:range_e[1], range_b[0]:range_b[1]] = min_dtype
    
    
    # F cannot see C
    mask[range_f[0]:range_f[1], range_c[0]:range_c[1]] = min_dtype

    # F cannot see E
    mask[range_f[0]:range_f[1], range_e[0]:range_e[1]] = min_dtype

    # 6. Dimension adjustment (Batch, Head, Seq, Seq)
    # Typically Attention Mask needs 4D: (1, 1, seq_len, seq_len) for broadcasting
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    sequence_index = [range_a, range_b, range_c, range_d, range_e, range_f]
    return mask,sequence_index

class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning (SFT).
    
    This dataset handles loading and preprocessing of image/video-text pairs for training
    vision-language models. It supports various data augmentation strategies including
    random erasing, Gaussian blur, color jitter, and attention-based feature dropping.
    
    The dataset handles special token formatting for different training modes:
    - VLOC mode: Visual location-aware chain-of-thought formatting
    - LIVR mode: Visual-language instruction-response formatting
    
    Attributes:
        model_id: Model identifier for processor selection
        processor: Tokenizer and feature extractor
        list_data_dict: List of data samples
        data_args: Configuration for data processing
        padding: Whether to pad sequences
    """

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        shuffle=True,
        random_seed=42,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.fps
        
        self.cur_step = 0
        self.stage_0_step = data_args.stage_0_step
        self.stage_1_step = data_args.stage_1_step
        self.stage_2_step = data_args.stage_2_step
        
        # for shuffle
        self.rng = np.random.default_rng(seed=random_seed)
        
        if shuffle:
            self.rng.shuffle(self.list_data_dict)
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        # import ipdb; ipdb.set_trace()
        
        sources = self.list_data_dict[i]
        
        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]

            if isinstance(image_files, str):
                image_files = Image.open(image_files).convert("RGB")
                if self.data_args.vloc == True:
                    if self.data_args.random_cut == True:
                        import torchvision.transforms as T

                        transform = T.Compose([
                            T.ToTensor(),
                            T.RandomErasing(p=1, scale=(0.1, 0.3), ratio=(0.3, 3.3), value=0), 
                        ])

                        tensor_img = transform(image_files)

                        image_files_damage = T.ToPILImage()(tensor_img)

                    elif self.data_args.random_drop == True:
                        
                        radius_gaussian = 2
                        print("random drop launch!")
                        radius_gaussian = self.rng.uniform(2.0, 5.0)
                        image_files_damage = image_files.filter(ImageFilter.GaussianBlur(radius=radius_gaussian))

                    elif self.data_args.attention_drop == True:
                        feature_index = True
                        image_files_damage = image_files.copy()
                    else:
                        color_distort = transforms.ColorJitter(
                        brightness=0.4, 
                        contrast=0.4, 
                        saturation=0.4, 
                        hue=0.1
                        )
                        print("color random")
                        image_files_damage = color_distort(image_files)

                    image_files = [image_files_damage,image_files]
                else:
                    image_files = [image_files]
            else:
                image_files = [Image.open(image_file).convert("RGB") for image_file in image_files]

            images = []
            
            for image_file in image_files:
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
           
        if images is None:
            
            print("No image or video found in the data.")
            images = []
            # Create a black image as a placeholder
            black_image = Image.new("RGB", (self.image_resized_w, self.image_resized_h), (0, 0, 0))
            images.append(get_image_info(black_image, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif len(images) == 0:
            print("No image or video found in the data.")
            # Create a black image as a placeholder
            black_image = Image.new("RGB", (self.image_resized_w, self.image_resized_h), (0, 0, 0))
            images.append(get_image_info(black_image, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
        
        if videos is not None:
            
            # import ipdb; ipdb.set_trace()
            pass
            
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        
        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
            
        for _, j in enumerate(range(0, len(sources), 2)):
            
            if j >= 2:
                break
            
            user_input = sources[j]
            gpt_response = sources[j + 1]
                        
            if self.data_args.vloc == True:
                #import ipdb;ipdb.set_trace()
                user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}"
                if DEFAULT_IMAGE_TOKEN in user_input:
                    gpt_response = get_comt_data_in_response_vloc(gpt_response,DEFAULT_IM_END_TOKEN)
                gpt_response = f"{gpt_response}\n{DEFAULT_IM_END_TOKEN}\n"
            else:
                user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}"
                if DEFAULT_IMAGE_TOKEN in user_input:
                    gpt_response = get_comt_data_in_response_livr(gpt_response)
                gpt_response = f"{gpt_response}\n{DEFAULT_IM_END_TOKEN}\n"
            
            # import ipdb; ipdb.set_trace() exit
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                # raise ValueError('Every man is a poet when he is in love')
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                
                # del dino_val
                torch.cuda.empty_cache()
                
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )
            #import ipdb;ipdb.set_trace()

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)

        if self.data_args.vloc == True:

            index = (torch.where(input_ids == 151675)[0][0]).item()
            attention_mask,sequence_index = create_custom_causal_mask(input_ids)
            print("sequence_index",sequence_index)
            
            if self.data_args.ce_loss == "double":
                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(input_ids[:index])), #sequence_index[3][0] 
                        input_ids[index:sequence_index[4][1]], 
                        torch.tensor([IGNORE_INDEX]),
                        input_ids[sequence_index[5][0]+1:sequence_index[5][1]],
                        # torch.tensor([IGNORE_INDEX] * len(input_ids[sequence_index[4][1]:])),
                    ],
                    dim=0,
                )
            elif self.data_args.ce_loss == "positive_only":
                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(input_ids[:index])), #sequence_index[3][0] 
                        input_ids[index:sequence_index[4][1]], 
                        torch.tensor([IGNORE_INDEX] * len(input_ids[sequence_index[4][1]:])),
                    ],
                    dim=0,
                )
            elif self.data_args.ce_loss == "negative_only":
                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(input_ids[:index])), #sequence_index[3][0] 
                        input_ids[index:sequence_index[3][1]+1],torch.tensor([IGNORE_INDEX] *len(input_ids[sequence_index[3][1]+1:sequence_index[4][1]])),
                        torch.tensor([IGNORE_INDEX]),
                        input_ids[sequence_index[5][0]+1:sequence_index[5][1]],
                        # torch.tensor([IGNORE_INDEX] * len(input_ids[sequence_index[4][1]:])),
                    ],
                    dim=0,
                )
        else:
            labels = torch.cat(all_labels, dim=0).to(torch.long)

            # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
            # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

            #attention_mask = create_custom_causal_mask_livr(input_ids)
            attention_mask = (input_ids > -1000000).to(torch.long)
        
        

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict["image_files"] = image_files

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        if self.data_args.vloc == True and self.data_args.need_KL == True:
            data_dict["sequence_index"] = sequence_index

        if self.data_args.attention_drop ==True:
            data_dict["feature_index"] = feature_index
        self.cur_step += 1
        
        return data_dict


class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.
    
    This collator handles batching of variable-length sequences and properly
    handles both 1D and 4D attention masks. For custom causal masks (4D),
    it pads them appropriately to handle different sequence lengths within a batch.
    
    Attributes:
        pad_token_id: Token ID used for padding
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_sequence_index = []
        batch_image_files = []
        batch_attention_masks = []
        batch_feature_index = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            if "image_files" in keys:
                batch_image_files.append(example["image_files"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_attention_masks.append(example["attention_mask"])
            
            if "sequence_index" in keys:
                batch_sequence_index.append(example["sequence_index"])

            if "feature_index" in keys:
                batch_feature_index.append(example["feature_index"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        # Use attention masks from dataset; support custom 4D masks.
        if len(batch_attention_masks) > 0 and batch_attention_masks[0].dim() == 4:
            # Pad to (batch, heads, max_len, max_len) with -inf-like values.
            max_len = max(len(seq) for seq in batch_input_ids)
            padded_masks = []
            for mask, seq in zip(batch_attention_masks, batch_input_ids):
                mask = mask.detach().cpu()
                seq_len = len(seq)
                min_dtype = torch.finfo(mask.dtype).min
                padded = torch.full(
                    (mask.shape[0], mask.shape[1], max_len, max_len),
                    fill_value=min_dtype,
                    dtype=mask.dtype,
                    device=mask.device,
                )
                padded[:, :, :seq_len, :seq_len] = mask
                padded_masks.append(padded)
            attention_mask = torch.cat(padded_masks, dim=0)
        else:
            # Default padding for 1D masks.
            attention_mask = input_ids != self.pad_token_id

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
            
        if len(batch_image_files) > 0:
            data_dict["image_files"] = batch_image_files

        if len(batch_sequence_index) > 0:
            data_dict["sequence_index"] = batch_sequence_index

        if len(batch_feature_index) > 0:
            data_dict["feature_index"] = batch_feature_index

        return data_dict

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)