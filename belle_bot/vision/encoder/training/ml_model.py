import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        # x.shape[0] is the batch size
        return x.view(x.shape[0], *self.shape)


class UpConvBlock(nn.Module):
    """Bilinear upsampling followed by standard convolution to eliminate checkerboard artifacts."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Standard Convolution Block with Residual Connection to smooth feature maps."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


class SmoothUpBlock(nn.Module):
    """Bilinear Upsampling followed by double standard convolutions to eliminate screendoor grids."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.upsample(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class VAE2_448(nn.Module):
    def __init__(self, img_channels=4, latent_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: Downsamples 6 times (448 -> 224 -> 112 -> 56 -> 28 -> 14 -> 7)
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(img_channels, 32, stride=2),   # 448x448 -> 224x224
            DepthwiseSeparableConv(32, 64, stride=2),            # 224x224 -> 112x112
            DepthwiseSeparableConv(64, 128, stride=2),           # 112x112 -> 56x56
            DepthwiseSeparableConv(128, 256, stride=2),          # 56x56   -> 28x28
            DepthwiseSeparableConv(256, 512, stride=2),          # 28x28   -> 14x14
            DepthwiseSeparableConv(512, 512, stride=2),          # 14x14   -> 7x7
        )

        self.flatten_dim = 512 * 7 * 7  # 25,088

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder: Upsamples 6 times with increased channel capacity (7 -> 14 -> 28 -> 56 -> 112 -> 224 -> 448)
        self.decoder = nn.Sequential(
            SmoothUpBlock(512, 512),                                # 7x7     -> 14x14
            SmoothUpBlock(512, 256),                                # 14x14   -> 28x28
            SmoothUpBlock(256, 128),                                # 28x28   -> 56x56
            SmoothUpBlock(128, 64),                                 # 56x56   -> 112x112
            SmoothUpBlock(64, 32),                                  # 112x112 -> 224x224
            SmoothUpBlock(32, 16),                                  # 224x224 -> 448x448
            nn.Conv2d(16, img_channels, kernel_size=3, padding=1),  # Final projection
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
