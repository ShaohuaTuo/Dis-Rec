import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, decoder_dims, n_item):
        super().__init__()
        all_dims = [latent_dim] + eval(decoder_dims) + [n_item]
        modules = []
        for i in range(len(all_dims) - 1):
            modules.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*modules)

    def forward(self, z):
        return self.mlp(z)