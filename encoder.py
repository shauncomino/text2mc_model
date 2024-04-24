import torch
from torch import nn
from torch.nn import functional as F
from decoder import (
    VAE_AttentionBlock as VAE_AttentionBlock3D,
    VAE_ResidualBlock as VAE_ResidualBlock3D,
)


class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # First layer: Initial convolution without changing size, just increasing channels
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            VAE_ResidualBlock3D(128, 128),
            VAE_ResidualBlock3D(128, 128),
            # Second layer: First downsampling, reduce dimensions by half
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),  # Size is halved
            nn.ReLU(),
            VAE_ResidualBlock3D(128, 256),
            # Third layer: Second downsampling, reduce dimensions by half again
            nn.Conv3d(
                256, 256, kernel_size=3, stride=2, padding=1
            ),  # Size is halved again
            nn.ReLU(),
            VAE_ResidualBlock3D(256, 512),
            # Fourth layer: Third downsampling, reduce dimensions by half again
            nn.Conv3d(
                512, 512, kernel_size=3, stride=2, padding=1
            ),  # Size is halved again
            nn.ReLU(),
            VAE_ResidualBlock3D(512, 512),
            VAE_ResidualBlock3D(512, 512),
            VAE_AttentionBlock3D(512),
            VAE_ResidualBlock3D(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            # Final layers to adjust the feature map size without changing spatial dimensions
            nn.Conv3d(512, 8, kernel_size=3, padding=1),
            nn.Conv3d(8, 8, kernel_size=1),
        )

    def forward(self, x, noise):
        x = self.layers(x)

        # Splitting the channels into mean and log variance for VAE
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # Reparameterization trick for VAE
        x = mean + stdev * noise
        x *= 0.18215  # Scaling factor

        return x
