import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM


MODEL_LOCAL_DIR = os.environ.get("MODEL_LOCAL_DIR", os.path.expanduser("~/models/Qwen3.5-0.8B"))


class VisionLossModule(nn.Module):
    def __init__(self, qwen_model_id="Qwen/Qwen3.5-0.8B"):
        super().__init__()

        model_config_path = os.path.join(MODEL_LOCAL_DIR, "config.json")
        if os.path.exists(model_config_path):
            self.qwen = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL_DIR)
        else:
            self.qwen = AutoModelForCausalLM.from_pretrained(qwen_model_id)
            self.qwen.save_pretrained(MODEL_LOCAL_DIR)

        # Freeze backbones
        for param in self.qwen.parameters():
            param.requires_grad = False

        # Enable gradient checkpointing on frozen model
        self.qwen.gradient_checkpointing_enable()
        self.qwen.enable_input_require_grads()

        # Get image token ID
        self.image_token_id = 151643  # Qwen's <|extra_0|>

    def forward(self, input_ids_list, labels_list, projected_features=None, image_counts=None):
        batch_size = len(input_ids_list)
        device = input_ids_list[0].device

        if projected_features is not None:
            # We need to know how many patches per image.
            # Let's assume each image has 256 patches (as hinted by VAE2_448)
            num_patches = 256
            # projected_features = [ (num_patches, hidden_dim), (num_patches, hidden_dim), ... ]
            image_features_list = torch.split(projected_features, num_patches, dim=0)
        else:
            image_features_list = []

        sample_embeds, sample_labels = [], []
        image_idx = 0

        for i in range(batch_size):
            text_ids = input_ids_list[i]
            text_labels = labels_list[i]
            text_embeds = self.qwen.get_input_embeddings()(text_ids)

            # Reconstruct sequence with interleaved images
            final_embeds = []
            final_labels = []

            curr_pos = 0
            # Identify all indices of image tokens
            # Find tokens == self.image_token_id
            image_indices = (text_ids == self.image_token_id).nonzero(as_tuple=True)[0]

            last_idx = 0
            for idx in image_indices:
                # Add text chunk before image
                final_embeds.append(text_embeds[last_idx:idx])
                final_labels.append(text_labels[last_idx:idx])

                # Add image features
                if image_idx < len(image_features_list):
                    final_embeds.append(image_features_list[image_idx])
                    final_labels.append(
                        torch.full((image_features_list[image_idx].size(0),), -100, dtype=torch.long, device=device))
                    image_idx += 1

                last_idx = idx + 1

            # Add remaining text
            final_embeds.append(text_embeds[last_idx:])
            final_labels.append(text_labels[last_idx:])

            sample_embeds.append(torch.cat(final_embeds, dim=0))
            sample_labels.append(torch.cat(final_labels, dim=0))

        # Non-mutating padding to avoid autograd slice-assignment overhead
        batch_embeds = torch.nn.utils.rnn.pad_sequence(sample_embeds, batch_first=True, padding_value=0.0).to(self.qwen.get_input_embeddings().weight.dtype)
        batch_labels = torch.nn.utils.rnn.pad_sequence(sample_labels, batch_first=True, padding_value=-100)

        max_len = batch_embeds.size(1)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for i, emb in enumerate(sample_embeds):
            attention_mask[i, :emb.size(0)] = 1

        outputs = self.qwen(
            inputs_embeds=batch_embeds,
            attention_mask=attention_mask,
            labels=batch_labels,
            use_cache=False
        )
        return outputs.loss
