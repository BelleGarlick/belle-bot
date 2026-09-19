import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:8].eval() # Up to a certain layer
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        return F.huber_loss(self.vgg(x), self.vgg(y), reduction='mean')


class VaeLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.perception_loss = PerceptualLoss()

    def forward(self, recon_x, x, mu, logvar, kl_beta_annealing=0.1):
        recon_loss = F.huber_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss \
            + (kl_beta_annealing * kl_loss) \
            + 2 * self.perception_loss(x, recon_x)


