# CoVT Training Instructions

## Preparations

1. Firstly install the dependencies. To ensure compatibility between the training environment of Qwen2.5-VL, we run:
    ```
    cd train
    bash install_training.sh
    ```

2. Prepare the training data.

We follow the [github](https://github.com/Wakals/CoVT) to get training data on [huggingface](https://huggingface.co/datasets/Wakals/CoVT-Dataset). Our training script is based on the implementation from [github](https://github.com/Wakals/CoVT). Therefore, to keep the consistency, the dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.

The example is like:
<details>
<summary>Example</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```
Make sure that `<image>` appears at the beginning of the first row in `conversations`.
</details>

## Training

The training process consists of four stages. By using `4 x A40`, they use around `6 hours` in total.
By running the following command, we can train the model.
```
bash scripts/finetune.sh
```
The final checkpoint will be saved under `train/output/lora_merged`.

You can customize the configure in `scripts/finetune.sh`.