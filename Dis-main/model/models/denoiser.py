
import torch
import torch.nn as nn
import math


def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DenoiseMLP(nn.Module):
    def __init__(self, dims, emb_size, dropout=0.1):
        super(DenoiseMLP, self).__init__()
        self.emb_size = emb_size
        self.emb_layer = nn.Linear(emb_size, emb_size)

        input_dim = dims[0] + emb_size
        output_dim = dims[-1]


        mlp_modules = []
        hidden_dims = [input_dim] + dims[1:-1]

        for i in range(len(hidden_dims) - 1):
            mlp_modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))

        self.mlp_body = nn.Sequential(*mlp_modules)


        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z, timesteps):

        time_emb = timestep_embedding(timesteps, self.emb_size)
        time_emb = self.emb_layer(time_emb)


        h = torch.cat([z, time_emb], dim=1)


        h = self.mlp_body(h)


        h = self.output_layer(h)


        return h + z