import torch
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


class VAEOld(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAEOld, self).__init__()

        # Encoder: Downsampling 224x224 -> 7x7
        self.encoder = nn.Sequential(
            *self._make_downsample_block(3, 64),  # 112x112
            *self._make_downsample_block(64, 128),  # 56x56
            *self._make_downsample_block(128, 256),  # 28x28
            *self._make_downsample_block(256, 512),  # 14x14
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.LeakyReLU(0.2),
        )

        # Latent space: Mean and Log-Variance
        # 512 * 7 * 7 = 25088
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 512 * 7 * 7),
            nn.LeakyReLU(0.2),
            Reshape((512, 7, 7)),
            *self._make_upsample_block(512, 256),
            *self._make_upsample_block(256, 128),
            *self._make_upsample_block(128, 64),
            *self._make_upsample_block(64, 32),
            *self._make_upsample_block(32, 16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_downsample_block(self, in_channels, out_channels):
        return (
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(0.2),
            # ResidualBlock(out_channels),
        )

    def _make_upsample_block(self, in_channels, out_channels):
        return (
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(0.2),
            # ResidualBlock(out_channels),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x_flat = self.encoder(x)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        logvar = torch.clamp(logvar, -10, 10)

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar


# class LightweightVAE(nn.Module):
class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
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
