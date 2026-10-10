import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class VisualCortexEncoder(nn.Module):
    def __init__(self, img_channels=4, latent_dim=1024):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(img_channels, 32, stride=2),  # 224x224
            DepthwiseSeparableConv(32, 64, stride=2),  # 112x112
            DepthwiseSeparableConv(64, 128, stride=2),  # 56x56
            DepthwiseSeparableConv(128, 256, stride=2),  # 28x28
            DepthwiseSeparableConv(256, 512, stride=2),  # 14x14
            DepthwiseSeparableConv(512, 512, stride=2),  # 7x7
        )

        self.flatten_dim = 512 * 7 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
