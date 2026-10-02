import os
import gc
from collections import deque

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps"
BATCH_SIZE = 1
MAX_TEXT_LEN = 512

# todo
#  randomly sample the subsets
#  have encoder split to different patches
#  clean up code and make easier to use. possibly a way to plug into the image encoder so this trains during the image encoder training too
#  have a way to interact with the saved model

# 1. Dataset & Local Storage Configuration
dataset_name = "nvidia/Nemotron-Image-Training-v3"
sub_dataset = "aokvqa_1"

MODEL_LOCAL_DIR = os.environ.get("MODEL_LOCAL_DIR", os.path.expanduser("~/models/Qwen3.5-0.8B"))
COCO_LOCAL_DIR = os.environ.get("COCO_LOCAL_DIR", "/Users/belle/data/coco/train2017")

dataset = load_dataset(dataset_name, sub_dataset, split="train", streaming=True)

# 2. Vision Transforms
vision_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def process_nemotron_sample(sample, tokenizer):
    """Parses a single row into 1D input_ids, labels, and 4D image tensors."""
    messages = sample["messages"]
    images_data = sample.get("images", [])

    input_ids = []
    labels = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            text_content = "".join([
                i["content"] if isinstance(i, dict) and i.get("type") == "text" else str(i)
                for i in content if not (isinstance(i, dict) and i.get("type") == "image")
            ])
        else:
            text_content = content

        # Remove thinking trace from text_content if present
        if "<think>" in text_content and "</think>" in text_content:
            import re
            text_content = re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()
        elif "<think>" in text_content:
            # Handle cases where thinking might not be closed
            text_content = text_content.split("<think>")[0].strip()

        # Handle images in content list
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_id = item.get("image")
                    if isinstance(img_id, str) and img_id.endswith(".jpg"):
                        local_path = os.path.join(COCO_LOCAL_DIR, img_id)
                        if not os.path.exists(local_path):
                            os.makedirs(COCO_LOCAL_DIR, exist_ok=True)
                            for split in ["train2017", "val2017"]:
                                url = f"http://images.cocodataset.org/{split}/{img_id}"
                                try:
                                    response = requests.get(url, timeout=10)
                                    if response.status_code == 200:
                                        with open(local_path, "wb") as f:
                                            f.write(response.content)
                                        break
                                except Exception:
                                    pass

                        if os.path.exists(local_path):
                            try:
                                images_data.append(Image.open(local_path).convert("RGB"))
                            except Exception:
                                pass

        # Tokenize this message block
        msg_text = f"<|im_start|>{role}\n{text_content}<|im_end|>\n"
        msg_ids = tokenizer(msg_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

        input_ids.append(msg_ids)
        if role == "assistant":
            # Assistant tokens are targets
            labels.append(msg_ids.clone())
        else:
            # System and User tokens are ignored in loss
            labels.append(torch.full(msg_ids.shape, -100, dtype=torch.long))

    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.cat(labels, dim=0)

    # Truncate if necessary
    if input_ids.size(0) > MAX_TEXT_LEN:
        input_ids = input_ids[:MAX_TEXT_LEN]
        labels = labels[:MAX_TEXT_LEN]

    # Process images
    processed_images = []
    seen_images = set()

    for img in images_data:
        if isinstance(img, Image.Image) and id(img) not in seen_images:
            transformed = vision_transform(img)
            depth_channel = torch.zeros((1, transformed.size(1), transformed.size(2)), dtype=transformed.dtype)
            rgbd = torch.cat([transformed, depth_channel], dim=0)
            processed_images.append(rgbd)
            seen_images.add(id(img))

    image_tensors = torch.stack(processed_images) if processed_images else torch.empty(0)
    return input_ids, labels, image_tensors


def collate_fn(batch, tokenizer):
    """Batches individual dataset items and discards corrupted rows."""
    input_ids_list = []
    labels_list = []
    image_tensors_list = []

    for sample in batch:
        try:
            ids, lbls, imgs = process_nemotron_sample(sample, tokenizer)
            input_ids_list.append(ids)
            labels_list.append(lbls)
            image_tensors_list.append(imgs)
        except Exception:
            continue

    return input_ids_list, labels_list, image_tensors_list


class QwenVisionModel(nn.Module):
    def __init__(self, vision_encoder, vision_dim=1024, qwen_model_id="Qwen/Qwen3.5-0.8B"):
        super().__init__()
        self.vision_encoder = vision_encoder

        model_config_path = os.path.join(MODEL_LOCAL_DIR, "config.json")
        if os.path.exists(model_config_path):
            self.qwen = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL_DIR, torch_dtype=torch.bfloat16)
        else:
            self.qwen = AutoModelForCausalLM.from_pretrained(qwen_model_id, torch_dtype=torch.bfloat16)
            self.qwen.save_pretrained(MODEL_LOCAL_DIR)

        # Enable gradient checkpointing to save VRAM
        self.qwen.gradient_checkpointing_enable()

        text_dim = self.qwen.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

        # Freeze backbones (Phase 1 Alignment)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.qwen.parameters():
            param.requires_grad = False
        for param in self.projector.parameters():
            param.requires_grad = True

    def forward(self, input_ids_list, labels_list, image_tensors_list):
        batch_size = len(input_ids_list)
        device = input_ids_list[0].device
        dtype = self.projector[0].weight.dtype

        # 1. Track image counts per sample and aggregate all images across the batch
        image_counts = [imgs.size(0) if imgs.numel() > 0 else 0 for imgs in image_tensors_list]
        total_images = sum(image_counts)

        # 2. Batched visual encoding pass across all batch images
        if total_images > 0:
            all_images = torch.cat([imgs for imgs in image_tensors_list if imgs.numel() > 0], dim=0)
            with torch.no_grad():
                # [total_images, num_patches, vision_dim]
                all_vis_features = self.vision_encoder.encode(all_images)[0]

            # Project all vision patches: [total_images, num_patches, text_dim]
            all_proj_features = self.projector(all_vis_features)
        else:
            all_proj_features = None

        # 3. Concatenate visual embeddings and text embeddings per sample
        vis_idx = 0
        sample_embeds = []
        sample_labels = []

        for i in range(batch_size):
            text_ids = input_ids_list[i]
            text_labels = labels_list[i]
            text_embeds = self.qwen.get_input_embeddings()(text_ids)  # [seq_len, text_dim]

            num_imgs = image_counts[i]
            if num_imgs > 0:
                sample_vis = all_proj_features[vis_idx: vis_idx + num_imgs]  # [num_imgs, patches, text_dim]
                vis_idx += num_imgs

                # Flatten all image patches into a single token sequence for this sample
                sample_vis_flat = sample_vis.view(-1, sample_vis.size(-1))  # [num_imgs * patches, text_dim]

                # Prepend visual tokens to text tokens
                comb_embeds = torch.cat([sample_vis_flat, text_embeds], dim=0)

                # Labels: -100 for visual patch tokens, use pre-calculated text_labels
                ignore_labels = torch.full((sample_vis_flat.size(0),), -100, dtype=torch.long, device=device)
                comb_labels = torch.cat([ignore_labels, text_labels], dim=0)
            else:
                comb_embeds = text_embeds
                comb_labels = text_labels

            sample_embeds.append(comb_embeds)
            sample_labels.append(comb_labels)

        # 4. Pad embeddings, labels, and construct attention_mask for the mini-batch
        max_len = max(e.size(0) for e in sample_embeds)
        text_dim = sample_embeds[0].size(-1)

        batch_embeds = torch.zeros((batch_size, max_len, text_dim), dtype=dtype, device=device)
        batch_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i in range(batch_size):
            seq_len = sample_embeds[i].size(0)
            batch_embeds[i, :seq_len] = sample_embeds[i]
            batch_labels[i, :seq_len] = sample_labels[i]
            attention_mask[i, :seq_len] = 1

        # 5. Forward through language model
        outputs = self.qwen(
            inputs_embeds=batch_embeds,
            attention_mask=attention_mask,
            labels=batch_labels
        )
        return outputs


if __name__ == "__main__":
    from belle_bot.vision.encoder.training.ml_model import VAE2_448

    tokenizer_config_path = os.path.join(MODEL_LOCAL_DIR, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        tokenizer.save_pretrained(MODEL_LOCAL_DIR)

    my_vision_encoder = VAE2_448()
    my_vision_encoder.load_state_dict(
        torch.load("/Users/belle/Developer/belle-bot/belle_bot/vision/encoder/vision_encoder.pt", map_location=DEVICE)
    )

    model = QwenVisionModel(vision_encoder=my_vision_encoder, vision_dim=1024)
    model.to(DEVICE).to(torch.bfloat16)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Wrap dataset streaming into PyTorch DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    model.train()
    print(f"Starting batched training (Batch Size = {BATCH_SIZE})...")

    running_loss = deque(maxlen=100)
    for step, (input_ids_list, labels_list, image_tensors_list) in enumerate(dataloader):
        if not input_ids_list:
            continue

        # Move input tensors to target device
        input_ids_list = [ids.to(DEVICE) for ids in input_ids_list]
        labels_list = [lbls.to(DEVICE) for lbls in labels_list]
        image_tensors_list = [
            imgs.to(DEVICE).to(torch.bfloat16) if imgs.numel() > 0 else imgs
            for imgs in image_tensors_list
        ]

        # Forward Pass
        outputs = model(input_ids_list=input_ids_list, labels_list=labels_list, image_tensors_list=image_tensors_list)
        loss = outputs.loss

        # Backward pass & optimizer step
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.append(loss.item())
            if (step + 1) % 100 == 0:
                torch.mps.empty_cache()
                gc.collect()
                print(f"Step {step + 1} | Alignment Loss: {np.mean(running_loss):.4f}")

        if step >= 10_000:
            break