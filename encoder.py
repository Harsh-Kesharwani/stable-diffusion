import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2 , Width/2)
            nn.Conv2d(128, 128, kernel_size=3,stride=2, padding=0),

            # (Batch_Size, 128, Height/2 , Width/2) -> (Batch_Size, 256, Height/2 , Width/2)
            VAE_ResidualBlock(128, 256),
            # (Batch_Size, 256, Height/2 , Width/2) -> (Batch_Size, 256, Height/2 , Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height/2 , Width/2) -> (Batch_Size, 256, Height/4 , Width/4)
            nn.Conv2d(256, 256, kernel_size=3,stride=2, padding=0),

            # (Batch_Size, 256, Height/4 , Width/4) -> (Batch_Size, 512, Height/4 , Width/4)
            VAE_ResidualBlock(256, 512),
            # (Batch_Size, 512, Height/4 , Width/4) -> (Batch_Size, 512, Height/4 , Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/4 , Width/4) -> (Batch_Size, 512, Height/8 , Width/8)
            nn.Conv2d(512, 512, kernel_size=3,stride=2, padding=0),

            # (Batch_Size, 512, Height/8 , Width/8) -> (Batch_Size, 512, Height/8 , Width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8 , Width/8) -> (Batch_Size, 512, Height/8 , Width/8)
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_Size, 8, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x=F.pad(x, (0,1,0,1))

            x=module(x)

        # (Batch_Size, 8, Height, Height/8, Width/8) -> two tensors of shape (Batch_Size, 4, Height/8, Width/8)
        mean, log_var=torch.chunk(x, 2, dim=1)
        
        log_var=torch.clamp(log_var, -30, 20)
        var=log_var.exp()
        stdev=var.sqrt()

        Z=mean + stdev * noise

        Z*=0.18215

        # print('-'*100)
        # print('Z shape: ', Z.shape)
        # print('-'*100)

        return Z
    
if __name__ == "__main__":
    model = VAE_Encoder()
    model.eval()

    # Create a dummy input tensor: (batch_size=1, channels=3, height=64, width=64)
    x = torch.randn(1, 3, 64, 64)
    noise = torch.randn(1, 4, 8, 8)  # Match the latent shape (Z)

    with torch.no_grad():
        output = model(x, noise)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
