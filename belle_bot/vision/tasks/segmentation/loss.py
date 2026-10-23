import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedVaeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, y):
        recon_x = recon_x.contiguous()
        y = y.contiguous()
        loss = F.cross_entropy(recon_x, y)

        target_labels = torch.argmax(F.softmax(recon_x, dim=1), dim=1)
        accuracy = (target_labels == y).float().mean().item()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "error": 1 - accuracy
        }
