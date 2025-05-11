import torch 
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embed, n_token):
        super().__init__()
        self.token_embedding=nn.Embedding(n_vocab, n_embed)
        self.position_embedding=nn.Parameter(torch.zeros((n_token, n_embed)))

    def forward(self, tokens: torch.Tensor):
        x=self.token_embedding(tokens)
        x+=self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()
        self.layernorm_1=nn.LayerNorm(n_embed)
        self.attention=SelfAttention(n_head, n_embed)
        self.layernorm_2=nn.LayerNorm(n_embed)

        self.linear_1=nn.Linear(n_embed, 4*n_embed)
        self.linear_2=nn.Linear(4*n_embed, n_embed)

    def forward(self, x):
        residue=x

        x=self.layernorm_1(x)

        x=self.attention(x, causal_mask=True)

        x+=residue

        residue=x
        
        x=self.layernorm_2(x)

        x=self.linear_1(x)

        x=x*torch.sigmoid(1.702*x)

        x=self.linear_2(x)
        x+=residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=CLIPEmbedding(49408, 768, 77)
        self.layers=nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm=nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens=tokens.type(torch.long)

        state=self.embedding(tokens)

        for layer in self.layers:
            state=layer(state)

        output=self.layernorm(state)

        return output
    
if __name__ == "__main__":
    dummy_tokens = torch.randint(0, 49408, (1, 77))  # (Batch_Size, Seq_Len)

    # Instantiate the model
    model = CLIP()

    # Forward pass
    with torch.no_grad():  # no need to track gradients for testing
        output = model(dummy_tokens)

    # Print the output shape
    # Output shape: torch.Size([1, 77, 768])
    print("Output shape:", output.shape)

