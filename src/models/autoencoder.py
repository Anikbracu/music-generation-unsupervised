"""
Task 1: LSTM Autoencoder for single-genre music generation.

    z   = f_φ(X)        [encoder]
    X̂  = g_θ(z)         [decoder]
    L_AE = Σ ‖x_t − x̂_t‖²
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for piano-roll sequences."""

    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128, num_layers=2):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: X -> z
        self.enc_lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.2)
        self.enc_fc   = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> X̂
        self.dec_fc   = nn.Linear(latent_dim, hidden_dim)
        self.dec_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.2)
        self.out_fc   = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)
        return self.enc_fc(h[-1])

    def decode(self, z, seq_len):
        h_init = self.dec_fc(z)
        h0 = h_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        inp = h_init.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.dec_lstm(inp, (h0, c0))
        return torch.sigmoid(self.out_fc(out))

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z, x.size(1))
        return x_hat, z
