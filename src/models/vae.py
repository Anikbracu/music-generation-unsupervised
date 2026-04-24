"""
Task 2: Variational Autoencoder (VAE) for multi-genre music generation.

    q_φ(z|X) = N(μ(X), σ(X))
    z = μ + σ ⊙ ε,  ε ~ N(0, I)
    L_VAE = L_recon + β · D_KL(q_φ(z|X) ‖ p(z))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicVAE(nn.Module):
    """Variational Autoencoder conditioned on genre."""

    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128,
                 num_layers=2, n_genres=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_genres   = n_genres

        self.genre_emb = nn.Embedding(n_genres, 16)

        # Encoder: X -> (μ, log σ²)
        self.enc_lstm = nn.LSTM(input_dim + 16, hidden_dim, num_layers,
                                batch_first=True, bidirectional=True, dropout=0.2)
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder: z -> X̂
        self.dec_init = nn.Linear(latent_dim + 16, hidden_dim)
        self.dec_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.2)
        self.out_fc   = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, genre):
        g = self.genre_emb(genre).unsqueeze(1).expand(-1, x.size(1), -1)
        x_in = torch.cat([x, g], dim=-1)
        _, (h, _) = self.enc_lstm(x_in)
        h = torch.cat([h[-2], h[-1]], dim=1)   # bidirectional last states
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ ⊙ ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, genre, seq_len):
        g  = self.genre_emb(genre)
        zg = torch.cat([z, g], dim=-1)
        h_init = self.dec_init(zg)
        h0 = h_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        inp = h_init.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.dec_lstm(inp, (h0, c0))
        return torch.sigmoid(self.out_fc(out))

    def forward(self, x, genre):
        mu, logvar = self.encode(x, genre)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, genre, x.size(1))
        return x_hat, mu, logvar, z


def vae_loss_fn(x_hat, x, mu, logvar, beta=0.5):
    """L_VAE = L_recon + β · D_KL(q_φ(z|X) ‖ p(z))

    Closed-form D_KL for N(μ, σ²) vs N(0, I):
        D_KL = -0.5 · Σ (1 + log σ² − μ² − σ²)
    """
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon + beta * kl
    return total, recon.item(), kl.item()
