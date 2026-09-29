# Fine-tuning LLM for Belle Bot

This directory contains scripts to fine-tune the `Qwen/Qwen3-0.6B` model using LoRA (Low-Rank Adaptation) and SFT (Supervised Fine-Tuning).

## Prerequisites

You need a GPU with at least 8GB of VRAM to run the fine-tuning script with 4-bit quantization.

Install the required dependencies:

```bash
python3 -m pip install -r belle_bot/bellm/requirements_finetune.txt
```

## Dataset Format

The fine-tuning script expects a JSONL file where each line is a chat session:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
```

See `belle_bot/bellm/dataset_sample.jsonl` for a sample.

## Fine-tuning

Run the fine-tuning script:

```bash
PYTHONPATH=. python3 belle_bot/bellm/finetune.py
```

You can customize parameters in the script or by modifying the `finetune()` call in `if __name__ == "__main__":`.

## Inference

To test your fine-tuned model:

```bash
PYTHONPATH=. python3 belle_bot/bellm/inference.py
```

This will load the base model and apply the LoRA adapter saved in `./qwen-finetuned`.
