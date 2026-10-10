import torch
import torch.nn as nn


class VisualCortexLanguageProjection(nn.Module):
    def __init__(self, num_modalities: int, latent_dim: int, target_dim: int):
        super().__init__()
        # Initialize N independent linear layers AND subsequent activation functions
        self.projection_layers = nn.ModuleList([
            nn.Linear(in_features=latent_dim, out_features=target_dim)
            for _ in range(num_modalities)
        ])
        # You can optionally add a single global non-linearity *after* concatenation, but typically per-module is better.

    def forward(self, embeddings: list[torch.Tensor]):
        projected_tokens = []

        # 1. Projection & Activation Phase
        for proj_layer in self.projection_layers:
            # Step A: Linear Transformation
            projection = proj_layer(embeddings)

            # Step B: APPLY THE NON-LINEAR ACTIVATION
            activated_token = torch.nn.functional.gelu(projection)  # Using GELU is best practice!

            projected_tokens.append(activated_token)

        # 2. Fusion Phase (Return the first projection, assuming single modality)
        fused_output = projected_tokens[0]
        
        return fused_output
