"""V model: convolutional Variational Autoencoder (paper Fig. 21 / Appendix A.1).

Encoder: four stride-2 4x4 convolutions (32, 64, 128, 256 channels, ReLU)
         -> flatten (2*2*256 = 1024) -> two linear heads for mu and log-variance.
Decoder: linear 32 -> 1024, reshape to 1x1x1024, then four stride-2
         deconvolutions (128@5x5, 64@5x5, 32@6x6, 3@6x6) with ReLU and a final
         sigmoid, giving back a 64x64x3 image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Z_SIZE


class VAE(nn.Module):
    def __init__(self, z_size: int = Z_SIZE):
        super().__init__()
        self.z_size = z_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),     # 64 -> 31
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),    # 31 -> 14
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),   # 14 -> 6
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),  # 6  -> 2
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(1024, z_size)
        self.fc_logvar = nn.Linear(1024, z_size)

        self.fc_dec = nn.Linear(z_size, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),  # 1  -> 5
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),    # 5  -> 13
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),     # 13 -> 30
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid(),   # 30 -> 64
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(-1, 1024, 1, 1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, kl_tolerance: float = 0.5):
    """Loss of the reference implementation.

    * reconstruction: sum of squared errors over all pixels (L2), batch mean
    * KL divergence to N(0, I), clamped from below at ``kl_tolerance * z_size``
      ("free bits") so the encoder is not forced to waste capacity on matching
      the prior once the KL is already small.
    """
    r_loss = F.mse_loss(recon, x, reduction="none").sum(dim=(1, 2, 3)).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    kl = torch.clamp(kl, min=kl_tolerance * mu.shape[1]).mean()
    return r_loss + kl, r_loss, kl
