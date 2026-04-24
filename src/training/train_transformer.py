"""
Task 3: Train the Transformer on tokenized MIDI sequences.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..config import cfg
from ..preprocessing import find_midi_files, build_token_dataset
from ..models import MusicTransformer


class TokenDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train():
    cfg.make_dirs()
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)

    files = find_midi_files(cfg.RAW_MIDI_DIR)[:cfg.MAX_FILES]
    assert len(files) > 0, f"No MIDI files in {cfg.RAW_MIDI_DIR}"
    X, y = build_token_dataset(files, seq_len=256, max_per_file=6)

    ds = TokenDataset(X, y)
    n_tr = int(0.8 * len(ds))
    tr, vl = random_split(ds, [n_tr, len(ds) - n_tr],
                           generator=torch.Generator().manual_seed(cfg.SEED))
    tr_loader = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(vl, batch_size=cfg.BATCH_SIZE)

    model = MusicTransformer(
        vocab_size=cfg.VOCAB_SIZE, d_model=cfg.EMBED_DIM,
        n_heads=cfg.NUM_HEADS, n_layers=4, max_len=512,
        n_genres=cfg.N_GENRES).to(cfg.DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS_TR)
    print(f"Transformer params: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses, ppls = [], [], []
    for ep in range(cfg.EPOCHS_TR):
        model.train()
        tl, n = 0, 0
        for X_b, y_b in tqdm(tr_loader, desc=f'Ep {ep+1}/{cfg.EPOCHS_TR}', leave=False):
            X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
            inp, tgt = X_b[:, :-1], X_b[:, 1:]
            logits = model(inp, y_b)
            loss = F.cross_entropy(logits.reshape(-1, cfg.VOCAB_SIZE), tgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * X_b.size(0); n += X_b.size(0)
        sched.step()
        train_losses.append(tl / n)

        model.eval()
        vll, n = 0, 0
        with torch.no_grad():
            for X_b, y_b in vl_loader:
                X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
                logits = model(X_b[:, :-1], y_b)
                l = F.cross_entropy(logits.reshape(-1, cfg.VOCAB_SIZE), X_b[:,1:].reshape(-1))
                vll += l.item() * X_b.size(0); n += X_b.size(0)
        val_losses.append(vll / n)
        ppls.append(float(np.exp(vll / n)))
        print(f"Ep {ep+1:2d}: train={train_losses[-1]:.4f} | val={val_losses[-1]:.4f} | PPL={ppls[-1]:.3f}")

    torch.save(model.state_dict(), os.path.join(cfg.MODEL_DIR, 'task3_transformer.pt'))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(train_losses, 'b-o', label='Train')
    axes[0].plot(val_losses,   'r-s', label='Val')
    axes[0].set_title('Transformer Cross-Entropy Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ppls, 'g-^')
    axes[1].set_title('Validation Perplexity = exp(L_TR / T)')
    axes[1].grid(alpha=0.3)
    out_plot = os.path.join(cfg.PLOT_OUTPUT, 'task3_perplexity_report.png')
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    print(f"Saved Transformer plots → {out_plot}")


if __name__ == '__main__':
    train()
