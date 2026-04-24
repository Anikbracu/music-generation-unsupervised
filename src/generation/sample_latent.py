# ---------- DELIVERABLE 3: Latent Interpolation Experiment ----------
print("\n" + "="*60)
print("TASK 2 — DELIVERABLE 3: Latent Interpolation Experiment")
print("="*60)
print("Demonstrates smoothness of VAE latent space by interpolating")
print("between two real encoded samples.")
print()
 
with torch.no_grad():
    # Use two real samples from different genres → interesting interpolation
    sample_g1 = 0  # Classical
    sample_g2 = 2  # Rock (if available)
    if sample_g2 not in genre_latents:
        sample_g2 = list(genre_latents.keys())[-1]
 
    real_1 = ref_samples_by_genre[sample_g1][0].unsqueeze(0).to(cfg.DEVICE)
    real_2 = ref_samples_by_genre[sample_g2][0].unsqueeze(0).to(cfg.DEVICE)
 
    # Encode both (use Classical genre for decoding interpolations)
    genre_tensor = torch.tensor([sample_g1]).to(cfg.DEVICE)
    mu1, _ = vae_model.encode(real_1, genre_tensor)
    mu2, _ = vae_model.encode(real_2, torch.tensor([sample_g2]).to(cfg.DEVICE))
 
    n_interp = 8
    alphas = np.linspace(0, 1, n_interp)
    interp_rolls = []
 
    for alpha in alphas:
        z_k = (1 - alpha) * mu1 + alpha * mu2
        gen = vae_model.decode(z_k, genre_tensor, TASK2_SEQ_LEN).cpu().numpy()[0]
        interp_rolls.append(gen)
 
        # Save as MIDI with default piano
        threshold = TASK2_THRESHOLD
        if (gen > threshold).sum() < 20:
            threshold = float(np.percentile(gen, 80))
        path = f'{cfg.OUTPUT_DIR}/generated_midis/task2_interp_alpha_{alpha:.2f}.mid'
        pianoroll_to_midi(gen.T, path, fs=cfg.FS, threshold=threshold)
 
fig, axes = plt.subplots(1, n_interp, figsize=(22, 3.5))
for k, (ax, alpha, roll) in enumerate(zip(axes, alphas, interp_rolls)):
    ax.imshow(roll.T, aspect='auto', origin='lower', cmap='hot')
    ax.set_title(f'α = {alpha:.2f}'); ax.set_xlabel('Time')
    if k == 0: ax.set_ylabel('Pitch')
plt.suptitle(f'Task 2: Latent Interpolation  z = (1-α)·z₁ + α·z₂   ({cfg.GENRES[sample_g1]} → {cfg.GENRES[sample_g2]})',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{cfg.OUTPUT_DIR}/plots/task2_latent_interpolation.png', dpi=130, bbox_inches='tight')
plt.show()
print(f"✓ Saved 8 interpolation MIDIs at α = {alphas.round(2).tolist()}")
 