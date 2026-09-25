import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.lr_out = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lr_out(x + self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    """Lightweight convolution block to reduce compute and parameter size."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        # x.shape[0] is the batch size
        return x.view(x.shape[0], *self.shape)


class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=484):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: Downsamples 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(img_channels, 32, stride=2),   # 112x112
            DepthwiseSeparableConv(32, 64, stride=2),            # 56x56
            DepthwiseSeparableConv(64, 128, stride=2),           # 28x28
            DepthwiseSeparableConv(128, 256, stride=2),          # 14x14
            DepthwiseSeparableConv(256, 256, stride=2),          # 7x7
        )

        # Latent Projections (4x4 feature map flattened = 256 * 4 * 4 = 4096)
        self.fc_mu = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 7, latent_dim)

        # Decoder Setup
        self.decoder_input = nn.Linear(latent_dim, 256 * 7 * 7)

        # Decoder: Upsamples 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),  # 224x224
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
        h = h.view(-1, 256, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    """Inverted Residual Block with Depthwise Separable Convolutions and Skip Connections."""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []
        # Expansion phase (1x1 Conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # Depthwise phase (3x3 Conv)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # Pointwise phase (1x1 Conv)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class UpSampleBlock(nn.Module):
    """Lightweight upsampling using Nearest Neighbor + MBConv to prevent checkerboard artifacts."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = MBConvBlock(in_channels, out_channels, stride=1)

    def forward(self, x):
        return self.conv(self.upsample(x))


class OptimizedVAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        self.encoder = nn.Sequential(
            MBConvBlock(32, 64, stride=2),    # 112x112 -> 56x56
            MBConvBlock(64, 128, stride=2),   # 56x56   -> 28x28
            MBConvBlock(128, 256, stride=2),  # 28x28   -> 14x14
            MBConvBlock(256, 256, stride=2),  # 14x14   -> 7x7
        )

        self.flatten_dim = 256 * 7 * 7 # 12,544 dims

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            UpSampleBlock(256, 256),          # 7x7     -> 14x14
            UpSampleBlock(256, 128),          # 14x14   -> 28x28
            UpSampleBlock(128, 64),           # 28x28   -> 56x56
            UpSampleBlock(64, 32),            # 56x56   -> 112x112
            UpSampleBlock(32, 32),            # 112x112 -> 224x224
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.stem(x)
        h = self.encoder(h)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 7, 7)
        return self.decoder(h)