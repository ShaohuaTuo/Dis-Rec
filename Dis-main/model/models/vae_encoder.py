import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class VAE_Encoder(nn.Module):


    def __init__(self, dims, dropout=0.1, act_func='relu'):

        super(VAE_Encoder, self).__init__()
        self.dims = dims
        self.dropout = nn.Dropout(dropout)


        encoder_modules = []
        for i in range(len(dims) - 2):
            encoder_modules.append(nn.Linear(dims[i], dims[i + 1]))
            if act_func == 'relu':
                encoder_modules.append(nn.ReLU())
            elif act_func == 'tanh':
                encoder_modules.append(nn.Tanh())

        self.base_encoder = nn.Sequential(*encoder_modules)


        self.fc_mu = nn.Linear(dims[-2], dims[-1])
        self.fc_logvar = nn.Linear(dims[-2], dims[-1])

        self.apply(self.xavier_normal_initialization)

    def forward(self, x):

        h = self.dropout(x)
        h = self.base_encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def xavier_normal_initialization(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)


def reparameterize(mu, logvar):

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)