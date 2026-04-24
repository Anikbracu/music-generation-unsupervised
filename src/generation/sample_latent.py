"""
Sample from the latent space of a trained AE or VAE.
"""

import torch

from ..config import cfg


def sample_from_latent(model, n_samples=5, genre=None, seq_len=None, seed=None):
    """Return generated piano-rolls by sampling random latents z ~ N(0, I)."""
    seq_len = seq_len or cfg.SEQ_LEN
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, cfg.LATENT_DIM).to(cfg.DEVICE)
        if genre is not None:
            g = torch.tensor([genre] * n_samples, dtype=torch.long).to(cfg.DEVICE)
            x_hat = model.decode(z, g, seq_len)
        else:
            x_hat = model.decode(z, seq_len)
    return x_hat.cpu().numpy()


def latent_interpolation(model, genre=0, n_steps=8, seq_len=None, seed=None):
    """Interpolate between two random latent vectors."""
    seq_len = seq_len or cfg.SEQ_LEN
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        z1 = torch.randn(1, cfg.LATENT_DIM).to(cfg.DEVICE)
        z2 = torch.randn(1, cfg.LATENT_DIM).to(cfg.DEVICE)
        g  = torch.tensor([genre], dtype=torch.long).to(cfg.DEVICE)

        rolls = []
        import numpy as np
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * z1 + alpha * z2
            x_hat = model.decode(z, g, seq_len) if genre is not None else model.decode(z, seq_len)
            rolls.append(x_hat.cpu().numpy()[0])
    return rolls
