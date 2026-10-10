import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from tasks.encoder.data_loader import merge
from tasks.language.loss import VisionLossModule
from tasks.utils.models import VisualCortexStepArgs

BATCH_SIZE = 1
MAX_TEXT_LEN = 512
MAX_IMAGE_TENSORS_MB = 200

IMAGE_PLACEHOLDER_TOKEN = 151643


MODEL_LOCAL_DIR = os.environ.get("MODEL_LOCAL_DIR", os.path.expanduser("~/models/Qwen3.5-0.8B"))


def load_image(image_id):
    for split in ["train2017", "val2017"]:
        img_path = f"/Users/belle/Downloads/{split}/{image_id}"
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    return img.convert("RGB")
            except Exception:
                pass

    return None


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


def collate_fn(batch, tokenizer):
    batch_dict = {"input_ids": [], "labels": [], "images": []}

    for sample in batch:
        try:
            data = process_nemotron_sample(sample, tokenizer)
            imgs = data["images"]
            if imgs.numel() * imgs.element_size() > MAX_IMAGE_TENSORS_MB * 1024 * 1024:
                continue

            batch_dict["input_ids"].append(data["input_ids"])
            batch_dict["labels"].append(data["labels"])
            batch_dict["images"].append(imgs)
        except Exception as e:
            continue

    return batch_dict


tokenizer_config_path = os.path.join(MODEL_LOCAL_DIR, "tokenizer_config.json")
if os.path.exists(tokenizer_config_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    tokenizer.save_pretrained(MODEL_LOCAL_DIR)


loss_fn = VisionLossModule()


def perform_step(args: VisualCortexStepArgs):
    loss_fn.to(args.device)

    input_ids_list = args.batch["input_ids"]
    labels_list = args.batch["labels"]
    image_tensors_list = args.batch["images"]

    input_ids_list = [ids.to(args.device) for ids in input_ids_list]
    labels_list = [lbls.to(args.device) for lbls in labels_list]
    image_tensors_list = [
        imgs.to(args.device) if imgs.numel() > 0 else imgs
        for imgs in image_tensors_list
    ]

    # Vision Encoder Forward Pass
    all_vis_features = None
    if any(imgs.numel() > 0 for imgs in image_tensors_list):
         # Concatenate all images
         all_imgs = torch.cat([imgs for imgs in image_tensors_list if imgs.numel() > 0], dim=0)
         with torch.no_grad():
             all_vis_features = args.vision_encoder.encode(all_imgs)

    # Project features
    projected_features = None
    if all_vis_features is not None:
        # all_vis_features is (recon, mu, logvar)
        # We want mu (latent representation)
        latent_features = all_vis_features[1]
        projected_features = args.model(latent_features)

    image_counts = [imgs.size(0) if imgs.numel() > 0 else 0 for imgs in image_tensors_list]

    loss = loss_fn(
        input_ids_list=input_ids_list,
        labels_list=labels_list,
        projected_features=projected_features,
        image_counts=image_counts
    )

    # todo add the kl divergence to the vae

    return {
        "loss": loss,
    }
