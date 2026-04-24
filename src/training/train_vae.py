"""
Task 2: Train the VAE on multi-genre piano-roll data.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..config import cfg
from ..preprocessing import find_midi_files, build_pianoroll_dataset
from ..models import MusicVAE, vae_loss_fn


class PianoRollDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train():
    cfg.make_dirs()
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)

    files = find_midi_files(cfg.RAW_MIDI_DIR)[:cfg.MAX_FILES]
    assert len(files) > 0, f"No MIDI files in {cfg.RAW_MIDI_DIR}"
    X, y = build_pianoroll_dataset(files, seq_len=cfg.SEQ_LEN, max_per_file=8)

    ds = PianoRollDataset(X, y)
    n_tr = int(0.8 * len(ds))
    tr, vl = random_split(ds, [n_tr, len(ds) - n_tr],
                           generator=torch.Generator().manual_seed(cfg.SEED))
    tr_loader = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(vl, batch_size=cfg.BATCH_SIZE)

    model = MusicVAE(cfg.N_PITCHES, cfg.HIDDEN_DIM, cfg.LATENT_DIM,
                      cfg.NUM_LAYERS, cfg.N_GENRES).to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    print(f"VAE params: {sum(p.numel() for p in model.parameters()):,}")

    totals, recons, kls = [], [], []
    for ep in range(cfg.EPOCHS_VAE):
        model.train()
        tot = rs = ks = 0; n = 0
        beta_ep = min(cfg.BETA, cfg.BETA * (ep + 1) / 5)  # KL annealing
        for X_b, y_b in tqdm(tr_loader, desc=f'Ep {ep+1}/{cfg.EPOCHS_VAE}', leave=False):
            X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
            x_hat, mu, logvar, _ = model(X_b, y_b)
            loss, r, k = vae_loss_fn(x_hat, X_b, mu, logvar, beta=beta_ep)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * X_b.size(0)
            rs  += r * X_b.size(0)
            ks  += k * X_b.size(0)
            n   += X_b.size(0)
        totals.append(tot/n); recons.append(rs/n); kls.append(ks/n)
        print(f"Ep {ep+1:2d}: total={totals[-1]:7.3f} | recon={recons[-1]:7.3f} | KL={kls[-1]:6.3f} | β={beta_ep:.3f}")

    torch.save(model.state_dict(), os.path.join(cfg.MODEL_DIR, 'task2_vae.pt'))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(totals, 'b-o', label='Total')
    axes[0].plot(recons, 'g-s', label='Recon')
    axes[0].set_title('VAE Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(kls, 'r-^')
    axes[1].set_title('KL Divergence'); axes[1].grid(alpha=0.3)
    out_plot = os.path.join(cfg.PLOT_OUTPUT, 'task2_vae_loss.png')
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    print(f"Saved VAE loss plot → {out_plot}")


if __name__ == '__main__':
    train()
