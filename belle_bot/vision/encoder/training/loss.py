import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torchvision.models as models


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
        x_in = x[:, :3, :, :]
        y_in = y[:, :3, :, :]
        
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


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        self.slice1 = vgg[:4]  # Relu1_2
        self.slice2 = vgg[4:9]  # Relu2_2
        self.slice3 = vgg[9:16]  # Relu3_3

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x_norm = (x[:, :3] - self.mean) / self.std
        y_norm = (y[:, :3] - self.mean) / self.std

        h1_x, h1_y = self.slice1(x_norm), self.slice1(y_norm)
        h2_x, h2_y = self.slice2(h1_x), self.slice2(h1_y)
        h3_x, h3_y = self.slice3(h2_x), self.slice3(h2_y)

        return (
                F.l1_loss(h1_x, h1_y) +
                F.l1_loss(h2_x, h2_y) +
                F.l1_loss(h3_x, h3_y)
        )


class SpatialGradientLoss(nn.Module):
    """Calculates image gradient differences to sharply penalize blur on edges."""

    def forward(self, pred, target):
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return loss_x + loss_y


class OptimizedVaeLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0, depth_weight=0.5, grad_weight=0.5):
        super().__init__()
        # Use full-resolution VGG perceptual loss instead of downsampled
        self.perceptual_loss = VGGPerceptualLoss()
        self.gradient_loss = SpatialGradientLoss()
        self.perceptual_weight = perceptual_weight
        self.depth_weight = depth_weight
        self.grad_weight = grad_weight

    def forward(self, recon_x, x, mu, logvar, kl_beta=0.0005):
        recon_rgb, recon_depth = recon_x[:, :3, :, :], recon_x[:, 3:, :, :]
        rgb, depth = x[:, :3, :, :], x[:, 3:, :, :]

        # Pixel Reconstruction Loss
        rgb_loss = F.l1_loss(recon_rgb, rgb)

        depth_mask = (depth > 0.005).float()
        depth_loss = F.l1_loss(recon_depth * depth_mask, depth * depth_mask, reduction='sum')
        depth_loss = depth_loss / (depth_mask.sum() + 1e-8)

        recon_loss = rgb_loss + (self.depth_weight * depth_loss)

        # Image Edge Sharpening Loss
        edge_loss = self.gradient_loss(recon_rgb, rgb) + self.gradient_loss(recon_depth, depth)

        # Perceptual Loss on unscaled resolution
        perc_loss = self.perceptual_loss(recon_rgb, rgb)

        # KL Loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = (
                recon_loss
                + (self.perceptual_weight * perc_loss)
                + (self.grad_weight * edge_loss)
                + (kl_beta * kl_loss)
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "rgb_loss": rgb_loss.detach(),
            "depth_loss": depth_loss.detach(),
            "perc_loss": perc_loss.detach(),
            "kl_loss": kl_loss.detach()
        }