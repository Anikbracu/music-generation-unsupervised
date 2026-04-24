"""
Optional: Diffusion model for music generation (bonus / extensibility).

This is a minimal DDPM-style framework for piano-roll data.
Included per the assignment's source structure; not required for grading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine variance schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SimpleDenoisingMLP(nn.Module):
    """Tiny denoising network over flattened piano-roll patches."""

    def __init__(self, input_dim=128 * 128, hidden_dim=512, time_embed=128):
        super().__init__()
        self.time_embed = nn.Embedding(1000, time_embed)
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = torch.cat([x.flatten(1), t_emb], dim=-1)
        return self.net(h).view(x.shape)


class DiffusionMusic(nn.Module):
    """Simple DDPM over flattened piano-rolls (stub for extension)."""

    def __init__(self, timesteps=200, input_shape=(128, 128)):
        super().__init__()
        self.timesteps = timesteps
        self.input_shape = input_shape
        self.register_buffer('betas', cosine_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.model = SimpleDenoisingMLP(
            input_dim=input_shape[0] * input_shape[1])

    def q_sample(self, x0, t, noise):
        """Forward diffusion: add noise at step t."""
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def p_loss(self, x0):
        """Training loss: predict noise at a random timestep."""
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)
