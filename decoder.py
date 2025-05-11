import torch 
from torch import nn
from attention import SelfAttention
from torch.nn import functional as F


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.grpnorm_1=nn.GroupNorm(32, in_channels)
        self.conv_1=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.grpnorm_2=nn.GroupNorm(32, out_channels)
        self.conv_2=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer=nn.Identity()
        else:
            self.residual_layer=nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        residue=x

        x=self.grpnorm_1(x)
        x=F.silu(x)

        x=self.conv_1(x)

        x=self.grpnorm_2(x)
        x=F.silu(x)

        x=self.conv_2(x)

        return x+self.residual_layer(residue)

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.grpnorm=nn.GroupNorm(32, channels)
        self.attention=SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Features, Height, Width)
        residue=x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x=self.grpnorm(x)
        n, c, h, w=x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x=x.view((n,c,h*w))

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x=x.transpose(-1, -2)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features) 
        x=self.attention(x)

        # (Batch_Size, Height * Width, Features)  -> (Batch_Size, Features, Height * Width)
        x=x.transpose(-1, -2)

        # (Batch_Size, Features, Height , Width)
        x=x.view((n, c, h, w))

        x+=residue

        return x

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x/=0.18215

        for module in self:
            x=module(x)

        return x

if __name__ == "__main__":
    model = VAE_Decoder()
    model.eval()

    # Create a dummy input tensor: (batch_size=1, channels=4, height=16, width=16)
    x = torch.randn(1, 4, 8, 8)

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
