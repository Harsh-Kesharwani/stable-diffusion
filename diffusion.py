import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear_1=nn.Linear(n_embed, 4*n_embed)
        self.linear_2=nn.Linear(4*n_embed, 4*n_embed)

    def forward(self, x):
        x=self.linear_1(x)
        x=F.silu(x)
        x=self.linear_2(x)
        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.grpnorm_feature=nn.GroupNorm(32, in_channels)
        self.conv_feature=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time=nn.Linear(n_time, out_channels)

        self.grpnorm_merged=nn.GroupNorm(32, out_channels)
        self.conv_merged=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels==out_channels:
            self.residual_layer=nn.Identity()
        else:
            self.residual_layer=nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, feature, time):
        residue=feature

        feature=self.grpnorm_feature(feature)
        feature=F.silu(feature)

        feature=self.conv_feature(feature)

        time=F.silu(time)
        time=self.linear_time(time)

        merged=feature+time.unsqueeze(-1).unsqueeze(-1)

        merged=self.grpnorm_merged(merged)
        merged=F.silu(merged)
        merged=self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embed, d_context=768):
        super().__init__()

        channels=n_head*n_embed

        self.grpnorm=nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input=nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1=nn.LayerNorm(channels)
        self.attention_1=SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2=nn.LayerNorm(channels)
        self.attention_2=CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.layernorm_3=nn.LayerNorm(channels)

        self.linear_geglu_1=nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2=nn.Linear(4*channels, channels)

        self.conv_output=nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long=x

        x=self.grpnorm(x)
        x=self.conv_input(x)

        n, c, h, w=x.shape

        x=x.view((n,c,h*w))

        x=x.transpose(-1, -2)
        residue_short=x

        x=self.layernorm_1(x)
        x=self.attention_1(x)

        x+=residue_short

        residue_short=x

        x=self.layernorm_2(x)
        x=self.attention_2(x, context)

        x+=residue_short

        residue_short=x

        x=self.layernorm_3(x)
        x, gate=self.linear_geglu_1(x).chunk(2, dim=-1)

        x=x*F.gelu(gate)

        x=self.linear_geglu_2(x)

        x+=residue_short
        x=x.transpose(-1, -2)

        x=x.view((n, c, h, w))

        return self.conv_output(x)+residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv=nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
# passing arguments to the parent class nn.Sequential, not to your SwitchSequential class directly — because you did not override the __init__ method in SwitchSequential
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x=layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x=layer(x, time)
            else:
                x=layer(x)
        return x
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders=nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160), 
            UNET_ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.grpnorm = nn.GroupNorm(32, in_channels)

        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x=self.grpnorm(x)
        x=F.silu(x)

        x=self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding=TimeEmbedding(320)
        self.unet=UNET()
        self.final=UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        time=self.time_embedding(time)

        output=self.unet(latent, context, time)

        output=self.final(output)

        return output
    
if __name__ == "__main__":
    # Dummy inputs
    batch_size = 10
    height = 64
    width = 64
    in_channels = 4
    context_dim = 768
    seq_len = 77
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and move to device
    model = Diffusion().to(device)

    # Random input tensor with 4 channels
    x = torch.randn(batch_size, in_channels, height, width).to(device)

    print('Input shape to UNET: ', x.shape)

    # Time embedding (e.g., timestep from a diffusion schedule)
    t = torch.randn(batch_size, 320).to(device)

    print('Time Embedding shape to UNET: ',t.shape)

    # Context for cross attention (e.g., text embedding from CLIP or transformer)
    context = torch.randn(batch_size, seq_len, context_dim).to(device)

    print('context shape to UNET: ', context.shape)

    # Forward pass
    with torch.no_grad():
        output = model(x, context, t)
        print(output)

    print("Output shape of UNET:", output.shape)
