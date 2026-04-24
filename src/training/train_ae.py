"""
Task 1: Train the LSTM Autoencoder on single-genre piano-roll data.
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
from ..preprocessing import find_midi_files, build_pianoroll_dataset
from ..models import LSTMAutoencoder


class PianoRollDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train():
    cfg.make_dirs()
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)

    # --- Load data ---
    files = find_midi_files(cfg.RAW_MIDI_DIR)[:cfg.MAX_FILES]
    assert len(files) > 0, f"No MIDI files in {cfg.RAW_MIDI_DIR}"
    X, y = build_pianoroll_dataset(files, seq_len=cfg.SEQ_LEN, max_per_file=8)

    # --- Filter to single genre (Task 1 spec) ---
    target_genre = 0
    mask = (y == target_genre)
    if mask.sum() < 20:
        vals, counts = np.unique(y, return_counts=True)
        target_genre = int(vals[np.argmax(counts)])
        mask = (y == target_genre)
    X, y = X[mask], y[mask]
    print(f"Training on genre '{cfg.GENRES[target_genre]}' ({len(X)} samples)")

    ds = PianoRollDataset(X, y)
    n_tr = int(0.8 * len(ds))
    tr, vl = random_split(ds, [n_tr, len(ds) - n_tr],
                           generator=torch.Generator().manual_seed(cfg.SEED))
    tr_loader = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(vl, batch_size=cfg.BATCH_SIZE)

    # --- Model ---
    model = LSTMAutoencoder(cfg.N_PITCHES, cfg.HIDDEN_DIM,
                             cfg.LATENT_DIM, cfg.NUM_LAYERS).to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Train ---
    train_losses, val_losses = [], []
    for ep in range(cfg.EPOCHS_AE):
        model.train()
        tl, n = 0, 0
        for X_b, _ in tqdm(tr_loader, desc=f'Ep {ep+1}/{cfg.EPOCHS_AE}', leave=False):
            X_b = X_b.to(cfg.DEVICE)
            x_hat, _ = model(X_b)
            loss = F.mse_loss(x_hat, X_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * X_b.size(0); n += X_b.size(0)
        train_losses.append(tl / n)

        model.eval()
        vl_loss, n = 0, 0
        with torch.no_grad():
            for X_b, _ in vl_loader:
                X_b = X_b.to(cfg.DEVICE)
                x_hat, _ = model(X_b)
                vl_loss += F.mse_loss(x_hat, X_b).item() * X_b.size(0); n += X_b.size(0)
        val_losses.append(vl_loss / n)
        print(f"Ep {ep+1:2d}: train={train_losses[-1]:.5f} | val={val_losses[-1]:.5f}")

    torch.save(model.state_dict(), os.path.join(cfg.MODEL_DIR, 'task1_lstm_ae.pt'))

    # --- Plot loss curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Train')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   'r-s', label='Val')
    plt.title('Task 1: LSTM Autoencoder — Reconstruction Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE  (L_AE = Σ‖x_t − x̂_t‖²)')
    plt.legend(); plt.grid(alpha=0.3)
    out_plot = os.path.join(cfg.PLOT_OUTPUT, 'task1_reconstruction_loss_curve.png')
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    print(f"Saved loss curve → {out_plot}")


if __name__ == '__main__':
    train()
