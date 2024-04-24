import torch
import torch.optim as optim
from decoder import VAE_Decoder  # Ensure the decoder is designed for 3D data
from encoder import VAE_Encoder  # Ensure the encoder is designed for 3D data
import torch

block2vec_embedding_dim = 64
# Instantiate the model components for 3D
encoder = VAE_Encoder()
decoder = VAE_Decoder()

# Configuration for 3D data
batch_size = 4  # Number of random noise volumes to generate
depth = 64  # Depth of the noise volumes
height = 64  # Height of the noise volumes
width = 64  # Width of the noise volumes
channel_size = block2vec_embedding_dim

# Generate random noise volumes
noise_volumes = torch.rand(
    batch_size, channel_size, depth, height, width
)  # 3 channels, assumes an RGB-like 3D data
random_noise = torch.rand(
    batch_size, 4, 8, 8, 8
)  # Noise for VAE sampling, should match the encoder output

# Pass through the encoder, adding noise
encoded_representation = encoder(noise_volumes, random_noise)

# Pass through the decoder
reconstructed_volumes = decoder(encoded_representation)

print(f"Shape of reconstructed volumes: {reconstructed_volumes.shape}")
