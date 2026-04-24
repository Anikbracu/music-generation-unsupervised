# ---------- Task1: Autoencoder Implementation ----------
print("="*60)
print("TASK 1 — DELIVERABLE 1: LSTM Autoencoder Implementation")
print("="*60)
 
class LSTMAutoencoder(nn.Module):
    """Vanilla LSTM Autoencoder per assignment math:
       z = f_φ(X), X̂ = g_θ(z), L_AE = Σ||x_t − x̂_t||²
    """
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128, num_layers=2):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
 
        # Encoder
        self.enc_lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.2)
        self.enc_fc   = nn.Linear(hidden_dim, latent_dim)
 
        # Decoder
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
        return self.decode(z, x.size(1)), z
 
 
# ---- Filter to SINGLE GENRE (Task 1 specification) ----
single_genre_idx = 0  # Classical
mask = (y_pr == single_genre_idx)
if mask.sum() < 20:
    vals, counts = np.unique(y_pr, return_counts=True)
    single_genre_idx = int(vals[np.argmax(counts)])
    mask = (y_pr == single_genre_idx)
 
X_pr_single = X_pr[mask]
y_pr_single = y_pr[mask]
print(f"Training on single genre: {cfg.GENRES[single_genre_idx]} ({len(X_pr_single)} samples)")
 
single_ds = PianoRollDataset(X_pr_single, y_pr_single)
n_tr = int(0.8 * len(single_ds))
single_train, single_val = random_split(single_ds, [n_tr, len(single_ds)-n_tr],
                                         generator=torch.Generator().manual_seed(cfg.SEED))
single_train_loader = DataLoader(single_train, batch_size=cfg.BATCH_SIZE, shuffle=True)
single_val_loader   = DataLoader(single_val,   batch_size=cfg.BATCH_SIZE)
 
ae_model = LSTMAutoencoder(input_dim=cfg.N_PITCHES,
                            hidden_dim=cfg.HIDDEN_DIM,
                            latent_dim=cfg.LATENT_DIM,
                            num_layers=cfg.NUM_LAYERS).to(cfg.DEVICE)
ae_opt = torch.optim.Adam(ae_model.parameters(), lr=cfg.LR)
 
print(f"\nArchitecture:\n{ae_model}")
print(f"\nTrainable parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
 
with open(f'{cfg.OUTPUT_DIR}/task1_architecture.txt', 'w') as f:
    f.write("TASK 1: LSTM AUTOENCODER\n" + "="*50 + "\n")
    f.write(str(ae_model) + "\n\n")
    f.write(f"Total params: {sum(p.numel() for p in ae_model.parameters()):,}\n")
    f.write(f"Training genre: {cfg.GENRES[single_genre_idx]} ({len(X_pr_single)} samples)\n")
    f.write(f"Loss: L_AE = Σ ||x_t − x̂_t||²\n")
 
# ---------- Train ----------
print("\n" + "="*60)
print("Training LSTM Autoencoder...")
print("="*60)
 
ae_train_losses, ae_val_losses = [], []
 
for epoch in range(cfg.EPOCHS_AE):
    ae_model.train()
    tl, n = 0, 0
    for X, _ in tqdm(single_train_loader, desc=f'AE Ep {epoch+1}/{cfg.EPOCHS_AE}', leave=False):
        X = X.to(cfg.DEVICE)
        x_hat, _ = ae_model(X)
        loss = F.mse_loss(x_hat, X)   # L_AE
        ae_opt.zero_grad(); loss.backward(); ae_opt.step()
        tl += loss.item()*X.size(0); n += X.size(0)
    ae_train_losses.append(tl/n)
 
    ae_model.eval()
    vl, n = 0, 0
    with torch.no_grad():
        for X, _ in single_val_loader:
            X = X.to(cfg.DEVICE)
            x_hat, _ = ae_model(X)
            vl += F.mse_loss(x_hat, X).item()*X.size(0); n += X.size(0)
    ae_val_losses.append(vl/n)
    print(f'  Epoch {epoch+1:2d}: train={ae_train_losses[-1]:.5f} | val={ae_val_losses[-1]:.5f}')
 
torch.save(ae_model.state_dict(), f'{cfg.OUTPUT_DIR}/models/task1_lstm_ae.pt')