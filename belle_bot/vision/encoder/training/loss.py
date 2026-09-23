import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class LightweightPerceptualLoss(nn.Module):
    """
    Hardware-friendly perceptual loss using MobileNetV3-Small features across multiple depths.
    Uses ~2.5M parameters vs VGG16's ~14M+ parameters.
    """
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        features = mobilenet_v3_small(weights=weights).features.eval()

        # Multi-scale feature slices
        self.slice1 = features[:2]   # Low-level textures (16 channels)
        self.slice2 = features[2:4]  # Mid-level details (24 channels)
        self.slice3 = features[4:9]  # Higher-level semantics (48 channels)

        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        # Enforce eval mode for feature extractor regardless of parent module state
        super().train(mode)
        self.slice1.eval()
        self.slice2.eval()
        self.slice3.eval()
        return self

    def normalize(self, x):
        # Only normalize RGB channels if 4 channels are provided
        if x.shape[1] == 4:
            rgb = x[:, :3, :, :]
            return (rgb - self.mean) / self.std
        return (x - self.mean) / self.std

    def forward(self, x, y):
        # We only apply perceptual loss to the RGB channels if 4 channels are present
        x_in = x[:, :3, :, :] if x.shape[1] == 4 else x
        y_in = y[:, :3, :, :] if y.shape[1] == 4 else y
        
        x_norm = self.normalize(x_in)
        y_norm = self.normalize(y_in)

        # Extract features across multiple scales
        h1_x = self.slice1(x_norm)
        h1_y = self.slice1(y_norm)

        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)

        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)

        # Weighted loss (heavier on mid-to-high level features)
        loss1 = F.huber_loss(h1_x, h1_y, reduction="mean")
        loss2 = F.huber_loss(h2_x, h2_y, reduction="mean")
        loss3 = F.huber_loss(h3_x, h3_y, reduction="mean")

        return 0.2 * loss1 + 0.5 * loss2 + 1.0 * loss3


class OptimizedVaeLoss(nn.Module):
    def __init__(self, perceptual_weight=0.5):
        super().__init__()
        self.perceptual_loss = LightweightPerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(self, recon_x, x, mu, logvar, kl_beta=0.0005): # Reduced kl_beta
        # 1. Pixel Reconstruction Loss
        recon_loss = F.huber_loss(recon_x, x, reduction="mean")

        # 2. Perceptual Loss
        perc_loss = self.perceptual_loss(recon_x, x)

        # 3. KL Loss averaged across latent dimension to keep magnitude comparable to mean recon loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + (self.perceptual_weight * perc_loss) + (kl_beta * kl_loss)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "perc_loss": perc_loss.detach(),
            "kl_loss": kl_loss.detach()
        }