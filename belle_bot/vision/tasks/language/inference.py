import os
import re

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from datasets import get_dataset_config_names
from transformers import AutoTokenizer

from belle_bot.vision.training.ml_model import VAE2_448
from belle_bot.vision.tasks.encoder.data_loader import merge
from tasks.language.loss import VisionLossModule
from tasks.language.model import VisualCortexLanguageProjection
from belle_bot.vision.utils.image import load_image

DEVICE = "mps"
BATCH_SIZE = 1
MAX_TEXT_LEN = 512
MAX_IMAGE_TENSORS_MB = 200

IMAGE_PLACEHOLDER_TOKEN = 151643

# 1. Dataset & Local Storage Configuration
dataset_name = "nvidia/Nemotron-Image-Training-v3"
sub_datasets = get_dataset_config_names(dataset_name)[0:1]

MODEL_LOCAL_DIR = os.environ.get("MODEL_LOCAL_DIR", os.path.expanduser("~/models/Qwen3.5-0.8B"))

# 2. Vision Transforms & Model Setup
vision_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vision_encoder = VAE2_448()
vision_encoder.load_state_dict(
    torch.load("/Users/belle/Developer/belle-bot/belle_bot/vision/encoder/tasks/vision_encoder.pt",
               map_location=DEVICE)
)
vision_encoder.to(DEVICE).to(torch.bfloat16)
for param in vision_encoder.parameters():
    param.requires_grad = False


def preprocess_image(image):
    main_image = np.array(image)
    main_image = cv2.resize(main_image, (448, 448))
    depth_data = np.zeros((448, 448))

    return merge((main_image, depth_data))



def process_nemotron_sample(sample, tokenizer):
    """Parses a single row into 1D input_ids, labels, and 4D image tensors."""
    input_ids = []
    labels = []
    images = []

    def tokenise(text):
        return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

    for msg in sample["messages"]:
        role = msg.get("role", "user")
        content = msg["content"]

        # Ensure content is a list of dicts
        if isinstance(content, str):
            content = [{'text': content, 'type': 'text'}]

        for data in content:
            # Handle text strings
            if isinstance(data, str):
                data = {'text': data, 'type': 'text'}

            if data.get("type") == "image":
                # Process image
                img_path = data.get('image')
                pil_img = load_image(img_path)
                if pil_img is None:
                    continue

                img_data = preprocess_image(pil_img)
                # Convert (448, 448, 4) -> (4, 448, 448)
                img_tensor = torch.from_numpy(img_data).permute(2, 0, 1)
                images.append(img_tensor)

                t_tokens = torch.tensor([IMAGE_PLACEHOLDER_TOKEN])
                input_ids.append(t_tokens)
                labels.append(torch.full_like(t_tokens, -100))

            elif data.get("type") == "text":
                text = data.get('text', '')
                # Remove think tags
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if not text:
                    continue

                t_tokens = tokenise(text)
                input_ids.append(t_tokens)

                # Labels: if assistant, keep tokens; else -100
                if role == "assistant":
                    labels.append(t_tokens)
                else:
                    labels.append(torch.full_like(t_tokens, -100))

    return {
        "input_ids": torch.cat(input_ids),
        "labels": torch.cat(labels),
        "images": torch.stack(images) if images else torch.tensor([])
    }


if __name__ == "__main__":
    tokenizer_config_path = os.path.join(MODEL_LOCAL_DIR, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        tokenizer.save_pretrained(MODEL_LOCAL_DIR)

    loss_fn = VisionLossModule().to(DEVICE)
    projector = VisualCortexLanguageProjection(1, 1024, loss_fn.qwen.config.hidden_size)
    projector.to(DEVICE).to(torch.bfloat16)

    print(f"Starting batched training (Batch Size = {BATCH_SIZE})...")

    batch = process_nemotron_sample({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Hi, how are you?"}
            ]}
        ]
    }, tokenizer)

    input_ids_list = batch["input_ids"]
    labels_list = batch["labels"]
    image_tensors_list = batch["images"]

    input_ids_list = [ids.to(DEVICE) for ids in input_ids_list]
    labels_list = [lbls.to(DEVICE) for lbls in labels_list]
    image_tensors_list = [
        imgs.to(DEVICE).to(torch.bfloat16) if imgs.numel() > 0 else imgs
        for imgs in image_tensors_list
    ]

    # Vision Encoder Forward Pass
    all_vis_features = None
    if any(imgs.numel() > 0 for imgs in image_tensors_list):
         # Concatenate all images
         all_imgs = torch.cat([imgs for imgs in image_tensors_list if imgs.numel() > 0], dim=0)
         with torch.no_grad():
             all_vis_features = vision_encoder.encode(all_imgs)

    # Project features
    projected_features = None
    if all_vis_features is not None:
        # all_vis_features is (recon, mu, logvar)
        # We want mu (latent representation)
        latent_features = all_vis_features[1]
        projected_features = projector(latent_features)

    image_counts = [imgs.size(0) if imgs.numel() > 0 else 0 for imgs in image_tensors_list]

    loss = loss_fn(
        input_ids_list=input_ids_list,
        labels_list=labels_list,
        projected_features=projected_features,
        image_counts=image_counts
    )
