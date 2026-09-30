import torch
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

# 1. Select a sub-dataset from Nemotron-Image-Training-v3 (e.g., ocr, charts, captioning)
dataset_name = "nvidia/Nemotron-Image-Training-v3"
sub_dataset = "sharegpt4v"  # Change to any of the 76 available sub-datasets

# Stream the dataset to avoid downloading massive terabyte files locally
dataset = load_dataset(dataset_name, sub_dataset, split="train", streaming=True)

# 2. Simple vision transform matching your vision encoder requirements
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def process_nemotron_sample(sample, tokenizer):
    """
    Parses a single row from Nemotron-Image-Training-v3.
    Standard Nemotron schema includes a 'messages' list and an 'images' list.
    """
    messages = sample["messages"]
    images_data = sample.get("images", [])

    # Process text content into a single string for causal LM
    full_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        full_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    # Tokenise text
    text_inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = text_inputs.input_ids.squeeze(0)

    # Process associated images
    processed_images = []
    for img in images_data:
        # If dataset yields PIL objects natively or paths
        if isinstance(img, Image.Image):
            processed_images.append(vision_transform(img))

    if processed_images:
        image_tensors = torch.stack(processed_images)
    else:
        image_tensors = torch.empty(0)

    return input_ids, image_tensors
