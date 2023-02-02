import torch
import torch.nn as nn

from pymovis.learning.embedding import PositionalEmbedding

class Encoder(nn.Module):
    """ MLP encoder """
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 256, 256]):
        super(Encoder, self).__init__()

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

class Decoder(nn.Module):
    """ MLP decoder """
    def __init__(self, latent_dim, output_dim, hidden_dims=[256, 256, 256]):
        super(Decoder, self).__init__()

        input_dim = latent_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(nn.Linear(input_dim, h_dim), nn.PReLU()))
            input_dim = h_dim
        layers.append(nn.Sequential(nn.Linear(hidden_dims[-1], output_dim), nn.PReLU()))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class TrajectoryVAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 256], latent_dim=32, embedding_dim=32, max_len=60):
        super(TrajectoryVAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # layers
        self.encoder = Encoder(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.decoder = Decoder(latent_dim + input_dim + embedding_dim, output_dim, hidden_dims=hidden_dims)

        # embedding
        self.embedding = PositionalEmbedding(embedding_dim, max_len)
    
    def reparameterize(self, mean, log_var):
        stdev = torch.exp(0.5 * log_var)
        eps = torch.randn_like(stdev)
        return mean + eps * stdev

    def forward(self, env_curr, env_next, env_target, time_to_arrive):
        # encoder
        env_in = torch.cat([env_curr, env_next], dim=-1)
        mean, log_var = self.encoder(env_in)
        z = self.reparameterize(mean, log_var)

        # decoder
        time_emb = self.embedding(time_to_arrive).expand(env_curr.shape[0], -1) # (B, embedding_dim)
        env_out = self.decoder(torch.cat([z, env_curr, env_target, time_emb], dim=-1))

        return env_out, mean, log_var
    
    def sample(self, env_curr, env_target, time_to_arrive):
        z = torch.randn(env_curr.shape[0], self.latent_dim).to(env_curr.device) # (B, latent_dim)
        time_emb = self.embedding(time_to_arrive).expand(env_curr.shape[0], -1) # (B, embedding_dim)
        env_out = self.decoder(torch.cat([z, env_curr, env_target, time_emb], dim=-1))
        return env_out