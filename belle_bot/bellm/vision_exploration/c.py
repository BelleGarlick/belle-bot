import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from belle_bot.bellm.vision_exploration.a import process_nemotron_sample, dataset
from belle_bot.bellm.vision_exploration.b import QwenVisionModel

# Initialise components
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.6B-Instruct")
my_vision_encoder = MyCustomVisionEncoder()  # Replace with your instantiated vision encoder instance

model = QwenVisionModel(vision_encoder=my_vision_encoder, vision_dim=1024)
model.to("cuda").to(torch.bfloat16)

# Optimizer strictly observing parameters with requires_grad=True (the projector)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

print("Starting training on Nemotron-Image-Training-v3 data stream...")
model.train()

for i, sample in enumerate(dataset):
    try:
        input_ids, image_tensors = process_nemotron_sample(sample, tokenizer)
    except Exception as e:
        # Skip occasional broken image urls or parsing anomalies
        continue

    input_ids = input_ids.to("cuda")
    image_tensors = image_tensors.to("cuda").to(torch.bfloat16)

    # Causal LM targets match input IDs exactly
    labels = input_ids.clone()

    # Forward Pass
    outputs = model(input_ids=input_ids, image_tensors=image_tensors, labels=labels)
    loss = outputs.loss

    # Backward pass & step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        print(f"Step {i} | Alignment Loss: {loss.item():.4f}")

    if i >= 10000:  # Stop constraint or custom break
        break
