import argparse
from utils import get_model_name_from_path, load_pretrained_model
import ast

def merge_lora(args, anchor_model_id):
    print("-=-"*10)
    print(args.model_base)
    print(args.model_path)
    print("-=-"*10)
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(model_path=args.model_path, model_base=args.model_base,
                                             model_name=model_name, device_map='cpu', anchor_model_id=anchor_model_id)

    model.save_pretrained(args.save_model_path, safe_serialization=args.safe_serialization)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--safe-serialization", action='store_true')
    parser.add_argument("--anchor-model-id", type=str, required=True)
    
    args = parser.parse_args()

    anchor_model_id = ast.literal_eval(args.anchor_model_id)

    merge_lora(args, anchor_model_id)