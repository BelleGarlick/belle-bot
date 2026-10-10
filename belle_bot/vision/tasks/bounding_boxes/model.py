import torch
import torch.nn as nn


class VisualCortexBoundingBoxModel(nn.Module):
    def __init__(self, latent_dim: int = 1024, num_boxes: int = 300, num_classes: int = 80, query_dim: int = 256):
        super().__init__()
        self.num_boxes = num_boxes

        # Project 1024 latent state -> 300 distinct box query representations
        self.query_proj = nn.Linear(latent_dim, num_boxes * query_dim)
        self.norm = nn.LayerNorm(query_dim)

        # Shared heads applied identically across all 300 queries
        self.box_head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 4)
        )
        self.obj_head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # [B, latent_dim] -> [B, 300, query_dim]
        queries = self.norm(self.query_proj(x).view(B, self.num_boxes, -1))

        # Shared projections across dimension -1
        boxes = torch.sigmoid(self.box_head(queries))  # Coordinates strictly in [0, 1]
        obj_logits = self.obj_head(queries)  # Un-normalized logits
        cls_logits = self.cls_head(queries)  # Un-normalized logits

        # [B, 300, 4 + 1 + num_classes]
        return torch.cat([boxes, obj_logits, cls_logits], dim=-1)