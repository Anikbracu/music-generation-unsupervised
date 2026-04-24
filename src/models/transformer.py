# ---------- DELIVERABLE 1: Transformer Implementation ----------
print("="*60)
print("TASK 3 — DELIVERABLE 1: Transformer Implementation")
print("="*60)

class MusicTransformer(nn.Module):
    '''Causal (decoder-only) Transformer.
       p(X) = Π p(x_t | x_<t)
       h_t = Emb(x_t) + Emb(genre) + PosEmb(t)
    '''
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 dim_ff=1024, max_len=512, n_genres=5, dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.max_len  = max_len

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
        '''Autoregressive sampling: x_t ~ p_θ(x_t | x_<t)'''
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


transformer = MusicTransformer(
    vocab_size=cfg.VOCAB_SIZE, d_model=cfg.EMBED_DIM,
    n_heads=cfg.NUM_HEADS, n_layers=4, max_len=512,
    n_genres=cfg.N_GENRES).to(cfg.DEVICE)
tr_opt   = torch.optim.AdamW(transformer.parameters(), lr=3e-4, weight_decay=0.01)
tr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(tr_opt, T_max=cfg.EPOCHS_TR)

print(f"Params: {sum(p.numel() for p in transformer.parameters()):,}")
print(f"d_model={cfg.EMBED_DIM}, heads={cfg.NUM_HEADS}, layers=4, vocab={cfg.VOCAB_SIZE}")

with open(f'{cfg.OUTPUT_DIR}/task3_architecture.txt', 'w') as f:
    f.write("TASK 3: MUSIC TRANSFORMER\n" + "="*50 + "\n")
    f.write(str(transformer) + "\n")
    f.write(f"Total params: {sum(p.numel() for p in transformer.parameters()):,}\n")

# ---------- Train Transformer ----------
print("\nTraining Transformer...")
tr_train_losses, tr_val_losses, tr_ppls = [], [], []

for epoch in range(cfg.EPOCHS_TR):
    transformer.train()
    tl, n = 0, 0
    for X, y in tqdm(tok_train_loader, desc=f'TR Ep {epoch+1}/{cfg.EPOCHS_TR}', leave=False):
        X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
        inp, tgt = X[:, :-1], X[:, 1:]
        logits = transformer(inp, y)
        # L_TR = -Σ log p_θ(x_t | x_<t)
        loss = F.cross_entropy(logits.reshape(-1, cfg.VOCAB_SIZE), tgt.reshape(-1))
        tr_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        tr_opt.step()
        tl += loss.item()*X.size(0); n += X.size(0)
    tr_sched.step()
    tr_train_losses.append(tl/n)

    transformer.eval()
    vl, n = 0, 0
    with torch.no_grad():
        for X, y in tok_val_loader:
            X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
            logits = transformer(X[:, :-1], y)
            l = F.cross_entropy(logits.reshape(-1, cfg.VOCAB_SIZE), X[:,1:].reshape(-1))
            vl += l.item()*X.size(0); n += X.size(0)
    val_loss = vl / n
    tr_val_losses.append(val_loss)
    tr_ppls.append(float(np.exp(val_loss)))   # Perplexity = exp(L_TR / T)
    print(f'  Ep {epoch+1:2d}: train={tr_train_losses[-1]:.4f} | val={val_loss:.4f} | PPL={tr_ppls[-1]:.3f}')

torch.save(transformer.state_dict(), f'{cfg.OUTPUT_DIR}/models/task3_transformer.pt')