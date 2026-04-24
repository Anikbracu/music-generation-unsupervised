# ---------- DELIVERABLE 2: Reconstruction Loss Curve ----------
print("\n" + "="*60)
print("TASK 1 — DELIVERABLE 2: Reconstruction Loss Curve")
print("="*60)
 
plt.figure(figsize=(10,5))
plt.plot(range(1, len(ae_train_losses)+1), ae_train_losses, 'b-o', label='Train Loss', markersize=5)
plt.plot(range(1, len(ae_val_losses)+1),   ae_val_losses,   'r-s', label='Val Loss',   markersize=5)
plt.title('Task 1: LSTM Autoencoder — Reconstruction Loss Curve')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss  (L_AE = Σ||x_t − x̂_t||²)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{cfg.OUTPUT_DIR}/plots/task1_reconstruction_loss_curve.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"Final train loss: {ae_train_losses[-1]:.5f}")
print(f"Final val loss:   {ae_val_losses[-1]:.5f}")
 
TASK1_METRICS = {
    'final_train_loss': ae_train_losses[-1],
    'final_val_loss':   ae_val_losses[-1],
}