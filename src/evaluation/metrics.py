# ---------- DELIVERABLE 4: Metrics vs Task 1 Comparison ----------
print("\n" + "="*60)
print("TASK 2 — DELIVERABLE 4: Metrics vs Task 1 Comparison")
print("="*60)
 
# VAE reconstruction MSE on validation set
vae_model.eval()
vae_val_mse = 0; n = 0
with torch.no_grad():
    for X, y in pr_val_loader:
        X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)
        x_hat, _, _, _ = vae_model(X, y)
        vae_val_mse += F.mse_loss(x_hat, X).item() * X.size(0); n += X.size(0)
vae_val_mse /= n
 
def note_diversity(pr, thresh=0.5):
    active = (pr > thresh).astype(int)
    if active.sum() == 0: return 0.0
    unique_pitches = (active.sum(axis=1) > 0).sum()
    return float(unique_pitches) / 128.0
 
# Count notes per MIDI to show density difference
def count_midi_notes(paths):
    counts = []
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(p)
            counts.append(sum(len(inst.notes) for inst in pm.instruments))
        except Exception:
            counts.append(0)
    return counts
 
task1_note_counts = count_midi_notes(task1_midi_paths)
task2_note_counts = count_midi_notes(task2_midi_paths)
 
# Task 1 diversity from saved rolls
task1_div = [note_diversity(r.T) for r in generated_rolls]
task2_div = [note_diversity(r) for r, _ in generated_rolls_vae]
 
comparison_df = pd.DataFrame({
    'Metric': ['Validation Loss (MSE)',
               'Pitch Diversity (mean)',
               'Pitch Diversity (std)',
               'Avg Notes per MIDI',
               'Sequence Length (steps)',
               'Output Duration (s)',
               'Genre Conditioning',
               'Latent Space',
               'MIDI Instrument Variety'],
    'Task 1: LSTM AE': [f"{TASK1_METRICS['final_val_loss']:.5f}",
                        f"{np.mean(task1_div):.4f}",
                        f"{np.std(task1_div):.4f}",
                        f"{np.mean(task1_note_counts):.1f}",
                        f"{TASK1_SEQ_LEN}",
                        f"{TASK1_SEQ_LEN/cfg.FS:.1f}",
                        'Single genre',
                        'Deterministic',
                        '1 (piano)'],
    'Task 2: VAE':     [f"{vae_val_mse:.5f}",
                        f"{np.mean(task2_div):.4f}",
                        f"{np.std(task2_div):.4f}",
                        f"{np.mean(task2_note_counts):.1f}",
                        f"{TASK2_SEQ_LEN}",
                        f"{TASK2_SEQ_LEN/cfg.FS:.1f}",
                        f'{cfg.N_GENRES} genres',
                        'Probabilistic (μ,σ)',
                        f'{cfg.N_GENRES} (per genre)'],
})
print("\n" + comparison_df.to_string(index=False))
comparison_df.to_csv(f'{cfg.OUTPUT_DIR}/task2_vs_task1_comparison.csv', index=False)
 
# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
 
axes[0].bar(['Task 1 (AE)', 'Task 2 (VAE)'],
            [TASK1_METRICS['final_val_loss'], vae_val_mse],
            color=['steelblue', 'coral'])
axes[0].set_ylabel('Validation MSE'); axes[0].set_title('Reconstruction Loss')
axes[0].grid(alpha=0.3, axis='y')
 
axes[1].bar(['Task 1 (AE)', 'Task 2 (VAE)'],
            [np.mean(task1_div), np.mean(task2_div)],
            yerr=[np.std(task1_div), np.std(task2_div)],
            color=['steelblue', 'coral'], capsize=10)
axes[1].set_ylabel('Pitch Diversity'); axes[1].set_title('Generation Diversity')
axes[1].grid(alpha=0.3, axis='y')
 
axes[2].bar(['Task 1 (AE)', 'Task 2 (VAE)'],
            [np.mean(task1_note_counts), np.mean(task2_note_counts)],
            color=['steelblue', 'coral'])
axes[2].set_ylabel('Avg Notes per MIDI'); axes[2].set_title('Output Density')
axes[2].grid(alpha=0.3, axis='y')
 
plt.suptitle('Task 1 (AE) vs Task 2 (VAE) Comparison', fontsize=13)
plt.tight_layout()
plt.savefig(f'{cfg.OUTPUT_DIR}/plots/task2_vs_task1_comparison.png', dpi=130, bbox_inches='tight')
plt.show()
 
TASK2_METRICS = {
    'val_mse': vae_val_mse,
    'pitch_diversity_mean': np.mean(task2_div),
    'avg_notes': np.mean(task2_note_counts),
    'final_total_loss': vae_total_losses[-1],
    'final_kl_loss':    vae_kl_losses[-1],
}
 
print("\n" + "="*60)
print("TASK 2 COMPLETE — All 4 deliverables produced:")
print(f"  [1] VAE + KL-divergence  → models/task2_vae.pt + task2_architecture.txt")
print(f"  [2] 8 multi-genre MIDIs  → generated_midis/task2_<genre>_sample{{1..8}}.mid")
print(f"  [3] Latent interpolation → plots/task2_latent_interpolation.png (+ 8 MIDIs)")
print(f"  [4] vs-Task-1 comparison → task2_vs_task1_comparison.csv + plot")
print(f"  Signature: Multi-genre, long (12s), dense (thresh 0.38), 5 instruments")
print("="*60)