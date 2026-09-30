import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenVisionModel(nn.Module):
    def __init__(self, vision_encoder, vision_dim=1024, qwen_model_id="Qwen/Qwen3.5-0.6B-Instruct"):
        super().__init__()
        self.vision_encoder = vision_encoder

        # Load the text backbone
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_model_id, torch_dtype=torch.bfloat16)
        text_dim = self.qwen.config.hidden_size  # Usually 896 or 1024 for small Qwen variants

        # Projection layer: Two-layer MLP with non-linearity
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

        # Freezing logic for Feature Alignment (Phase 1)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.qwen.parameters():
            param.requires_grad = False
        for param in self.projector.parameters():
            param.requires_grad = True  # Only train the bridge layer

    def forward(self, input_ids, image_tensors, labels=None):
        # 1. Obtain text token embeddings via Qwen's standard embedding matrix
        text_embeds = self.qwen.get_input_embeddings()(input_ids.unsqueeze(0))  # [1, seq_len, text_dim]

        if image_tensors.numel() > 0:
            # 2. Extract visual representations via your custom encoder
            with torch.no_grad():
                # Expected output shape: [num_images, num_patches, vision_dim]
                visual_features = self.vision_encoder(image_tensors)

                # 3. Shape features into text dimension
            projected_visual_embeds = self.projector(visual_features)  # [num_images, num_patches, text_dim]

            # 4. Collapse patches across images if multiple exist
            projected_visual_embeds = projected_visual_embeds.view(1, -1, projected_visual_embeds.size(-1))

            # 5. Concatenate visual tokens dynamically before text tokens
            # (Simplification: prepending images; adjust layout based on text placeholder index if necessary)
            inputs_embeds = torch.cat([projected_visual_embeds, text_embeds], dim=1)

            # Adjust labels to calculate loss only on text targets
            if labels is not None:
                # Prepend -100 to ignore visual tokens in loss calculation
                ignore_labels = torch.full((1, projected_visual_embeds.size(1)), -100, dtype=torch.long,
                                           device=labels.device)
                labels = torch.cat([ignore_labels, labels.unsqueeze(0)], dim=1)
        else:
            inputs_embeds = text_embeds
            if labels is not None:
                labels = labels.unsqueeze(0)

        # 6. Pass everything directly into Qwen's forward function
        outputs = self.qwen(inputs_embeds=inputs_embeds, labels=labels)
        return outputs
