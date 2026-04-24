"""
Task 3: Transformer for long-horizon music generation.

    p(X) = Π_{t=1..T} p(x_t | x_<t)
    L_TR = −Σ log p_θ(x_t | x_<t)
    Perplexity = exp(L_TR / T)
    h_t = Emb(x_t) + Emb(genre) + PosEmb(t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicTransformer(nn.Module):
    """Causal (decoder-only) Transformer with genre conditioning."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 dim_ff=1024, max_len=512, n_genres=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb   = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len,    d_model)
        self.genre_emb = nn.Embedding(n_genres,   d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.ln   = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, genre=None):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        if genre is not None:
            h = h + self.genre_emb(genre).unsqueeze(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)
        h = self.ln(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prompt, genre, max_new=512, temperature=1.0, top_k=40):
        """Iterative autoregressive sampling: x_t ~ p_θ(x_t | x_<t)."""
        self.eval()
        x = prompt.clone()
        for _ in range(max_new):
            x_cond = x[:, -self.max_len:]
            logits = self.forward(x_cond, genre)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return x
