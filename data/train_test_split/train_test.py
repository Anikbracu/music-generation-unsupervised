# ----- PyTorch Datasets + DataLoaders -----
class PianoRollDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class TokenDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# 80/20 split
pr_ds = PianoRollDataset(X_pr, y_pr)
n_tr = int(0.8 * len(pr_ds))
pr_train, pr_val = random_split(pr_ds, [n_tr, len(pr_ds)-n_tr],
                                 generator=torch.Generator().manual_seed(cfg.SEED))
pr_train_loader = DataLoader(pr_train, batch_size=cfg.BATCH_SIZE, shuffle=True)
pr_val_loader   = DataLoader(pr_val,   batch_size=cfg.BATCH_SIZE)

tok_ds = TokenDataset(X_tok, y_tok)
n_tr_t = int(0.8 * len(tok_ds))
tok_train, tok_val = random_split(tok_ds, [n_tr_t, len(tok_ds)-n_tr_t],
                                   generator=torch.Generator().manual_seed(cfg.SEED))
tok_train_loader = DataLoader(tok_train, batch_size=cfg.BATCH_SIZE, shuffle=True)
tok_val_loader   = DataLoader(tok_val,   batch_size=cfg.BATCH_SIZE)

print(f"Piano-roll: {len(pr_train)} train / {len(pr_val)} val")
print(f"Tokens:     {len(tok_train)} train / {len(tok_val)} val")