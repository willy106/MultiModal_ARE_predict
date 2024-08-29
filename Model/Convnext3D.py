import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXt3DBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ConvNeXt3DBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Conv3d(dim, 4 * dim, 1)
        self.dw_conv = nn.Conv3d(4 * dim, 4 * dim, kernel_size, padding=kernel_size//2, groups=4 * dim)
        self.pw_conv2 = nn.Conv3d(4 * dim, dim, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.norm(x.permute(0, 2, 3, 4, 1))  # Change shape for LayerNorm
        x = x.permute(0, 4, 1, 2, 3)  # Revert shape back
        x = self.pw_conv1(x)
        x = self.gelu(x)
        x = self.dw_conv(x)
        x = self.gelu(x)
        x = self.pw_conv2(x)
        return x + identity

class ConvNeXt3DEncoder(nn.Module):
    def __init__(self, input_channels, num_blocks, dim):
        super(ConvNeXt3DEncoder, self).__init__()
        self.initial_conv = nn.Conv3d(input_channels, dim, kernel_size=7, stride=2, padding=3)
        self.blocks = nn.Sequential(*[ConvNeXt3DBlock(dim) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(dim)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.final_norm(x.permute(0, 2, 3, 4, 1))  # Change shape for LayerNorm
        x = x.permute(0, 4, 1, 2, 3)  # Revert shape back
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = self.flatten(x)
        return x

# # Example Usage
# input_channels = 1  # For example, for grayscale 3D images
# num_blocks = 6      # Number of ConvNeXt blocks
# dim = 64            # Dimensionality of the feature space

# model = ConvNeXt3DEncoder(input_channels, num_blocks, dim)

# # Example Input (e.g., a 3D medical image)
# example_input = torch.rand(1, 1, 128, 128, 128)  # Batch size 1, channel 1, 128x128x128 3D image

# # Forward Pass
# output = model(example_input)
# print(output.shape)  # Output shape
