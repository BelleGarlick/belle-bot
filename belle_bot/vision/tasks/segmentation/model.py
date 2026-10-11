import torch.nn as nn


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


class SubPixelUpBlock(nn.Module):
    """Sub-pixel convolution upsampling to eliminate checkerboard/screendoor artifacts."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # PixelShuffle reduces channels by factor of 4 while doubling height and width
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class VisualCortexSegmentationDecoder(nn.Module):
    def __init__(self, img_channels=80, latent_dim=1024):
        super().__init__()

        self.latent_dim = latent_dim

        self.flatten_dim = 512 * 7 * 7

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            SubPixelUpBlock(512, 512),  # 14x14
            SubPixelUpBlock(512, 256),  # 28x28
            SubPixelUpBlock(256, 128),  # 56x56
            SubPixelUpBlock(128, 64),  # 112x112
            SubPixelUpBlock(64, 32),  # 224x224
            SubPixelUpBlock(32, 32),  # 448x448
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),  # Final output projection
            nn.Sigmoid()
        )

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(x)
