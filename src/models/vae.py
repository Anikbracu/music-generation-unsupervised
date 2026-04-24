# ---------- Task2 1: VAE with KL-Divergence Loss ----------
print("="*60)
print("TASK 2 — DELIVERABLE 1: VAE with KL-Divergence Loss")
print("="*60)
 
class MusicVAE(nn.Module):
    """Genre-conditioned VAE per assignment math:
       q_φ(z|X) = N(μ(X), σ(X))
       z = μ + σ ⊙ ε
       L_VAE = L_recon + β · D_KL
    """
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128,
                 num_layers=2, n_genres=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_genres   = n_genres
 
        self.genre_emb = nn.Embedding(n_genres, 16)
 
        # Bidirectional encoder → μ, log σ²
        self.enc_lstm = nn.LSTM(input_dim + 16, hidden_dim, num_layers,
                                batch_first=True, bidirectional=True, dropout=0.2)
        self.fc_mu     = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim*2, latent_dim)
 
        # Decoder
        self.dec_init = nn.Linear(latent_dim + 16, hidden_dim)
        self.dec_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.2)
        self.out_fc   = nn.Linear(hidden_dim, input_dim)
 
    def encode(self, x, genre):
        g = self.genre_emb(genre).unsqueeze(1).expand(-1, x.size(1), -1)
        x_in = torch.cat([x, g], dim=-1)
        _, (h, _) = self.enc_lstm(x_in)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
 
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ ⊙ ε"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
 
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
        return self.decode(z, genre, x.size(1)), mu, logvar, z
 
 
def vae_loss_fn(x_hat, x, mu, logvar, beta=0.5):
    """L_VAE = L_recon + β · D_KL(q_φ(z|X) || p(z))
    
    Closed-form D_KL for N(μ,σ²) vs N(0,I):
        D_KL = -½ Σ (1 + log σ² − μ² − σ²)
    """
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta*kl, recon.item(), kl.item()
 
 
vae_model = MusicVAE(input_dim=cfg.N_PITCHES,
                      hidden_dim=cfg.HIDDEN_DIM,
                      latent_dim=cfg.LATENT_DIM,
                      num_layers=cfg.NUM_LAYERS,
                      n_genres=cfg.N_GENRES).to(cfg.DEVICE)
vae_opt = torch.optim.Adam(vae_model.parameters(), lr=cfg.LR)
 
print(f"\nArchitecture:\n{vae_model}")
print(f"\nTrainable parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
 
with open(f'{cfg.OUTPUT_DIR}/task2_architecture.txt', 'w') as f:
    f.write("TASK 2: VARIATIONAL AUTOENCODER (MULTI-GENRE)\n" + "="*50 + "\n")
    f.write(str(vae_model) + "\n\n")
    f.write(f"Total params: {sum(p.numel() for p in vae_model.parameters()):,}\n")
    f.write(f"β (KL weight): {cfg.BETA}\n")
    f.write(f"Genres: {cfg.GENRES}\n")
    f.write(f"Loss: L_VAE = L_recon + β·D_KL(q_φ(z|X) || N(0,I))\n")
 
# ---------- Train VAE on ALL genres ----------
print("\n" + "="*60)
print("Training VAE on ALL 5 genres (multi-genre)...")
print("="*60)
 
vae_total_losses, vae_recon_losses, vae_kl_losses = [], [], []
 
for epoch in range(cfg.EPOCHS_VAE):
    vae_model.train()
    tot, rsum, ksum, n = 0, 0, 0, 0
    # KL annealing: ramp β from 0 to target over first 5 epochs
    beta_ep = min(cfg.BETA, cfg.BETA * (epoch+1)/5)
 
    for X, y in tqdm(pr_train_loader, desc=f'VAE Ep {epoch+1}/{cfg.EPOCHS_VAE}', leave=False):
        X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
        x_hat, mu, logvar, _ = vae_model(X, y)
        loss, r, k = vae_loss_fn(x_hat, X, mu, logvar, beta=beta_ep)
        vae_opt.zero_grad(); loss.backward(); vae_opt.step()
        tot += loss.item()*X.size(0); rsum += r*X.size(0); ksum += k*X.size(0); n += X.size(0)
 
    vae_total_losses.append(tot/n)
    vae_recon_losses.append(rsum/n)
    vae_kl_losses.append(ksum/n)
    print(f'  Ep {epoch+1:2d}: total={vae_total_losses[-1]:7.3f} | recon={vae_recon_losses[-1]:7.3f} | KL={vae_kl_losses[-1]:6.3f} | β={beta_ep:.3f}')
 
torch.save(vae_model.state_dict(), f'{cfg.OUTPUT_DIR}/models/task2_vae.pt')
 
# VAE loss plot
fig, axes = plt.subplots(1, 2, figsize=(14,4))
axes[0].plot(vae_total_losses, 'b-o', label='Total L_VAE', markersize=4)
axes[0].plot(vae_recon_losses, 'g-s', label='Reconstruction', markersize=4)
axes[0].set_title('VAE Loss Components'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
 
axes[1].plot(vae_kl_losses, 'r-^', markersize=4)
axes[1].set_title('KL Divergence  D_KL(q_φ(z|X) || p(z))')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('KL'); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{cfg.OUTPUT_DIR}/plots/task2_vae_loss.png', dpi=130, bbox_inches='tight')
plt.show()