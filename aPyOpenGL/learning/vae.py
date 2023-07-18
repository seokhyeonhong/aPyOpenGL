import torch
import torch.nn as nn

""" Encoder for VAE """
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 256, 256]):
        super(VariationalEncoder, self).__init__()

        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(nn.Linear(input_dim, h_dim), nn.PReLU()))
            input_dim = h_dim
        self.layers = nn.Sequential(*layers)

        self.mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        x = self.layers(x)
        mean, log_var = self.mean(x), self.log_var(x)
        return mean, log_var

""" Decoder for VAE that concatenates latent vector to input for every layer """
class LatentConcatDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, hidden_dims=[256, 256, 256]):
        super(LatentConcatDecoder, self).__init__()

        self.layers = nn.ModuleList()
        for h_dim in hidden_dims:
            self.layers.append(nn.Sequential(nn.Linear(input_dim + latent_dim, h_dim), nn.PReLU()))
            input_dim = h_dim
        self.layers.append(nn.Linear(input_dim + latent_dim, output_dim))
    
    def forward(self, x, z):
        for layer in self.layers:
            x = layer(torch.cat([x, z], dim=-1))
        return x

