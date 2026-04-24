# ---------- Task1: 5 Generated MIDI Samples ----------
print("\n" + "="*60)
print("TASK 1 — DELIVERABLE 3: 5 Generated MIDI Samples")
print("="*60)
print("Generation strategy: encode real training samples → get valid")
print("latent codes → decode with slight perturbation (AE-standard).")
print()
 
# === TASK 1 SIGNATURE SETTINGS — makes output audibly distinct from VAE ===
TASK1_SEQ_LEN    = 128    # short: ~8 seconds at fs=16
TASK1_THRESHOLD  = 0.55   # higher threshold → sparse, monophonic feel
TASK1_VELOCITY   = 70     # soft, uniform dynamics (classical feel)
TASK1_PERTURB    = 0.08   # minimal noise (AE is deterministic by nature)
 
ae_model.eval()
generated_rolls = []
task1_midi_paths = []
 
with torch.no_grad():
    # Collect reference latents from real training data (the only region AE has learned)
    real_Xs = []
    for X_batch, _ in single_train_loader:
        real_Xs.append(X_batch)
        if sum(x.size(0) for x in real_Xs) >= 30:
            break
    real_X = torch.cat(real_Xs, dim=0)[:30].to(cfg.DEVICE)
    z_real = ae_model.encode(real_X)
    z_std  = z_real.std(dim=0, keepdim=True)
 
    # Pick 5 distinct real latents (spread apart) for maximum variety
    # Use k-means-like selection: pick furthest-apart latents
    chosen_indices = [0]
    for _ in range(4):
        dists = torch.cdist(z_real, z_real[chosen_indices]).min(dim=1).values
        chosen_indices.append(int(torch.argmax(dists).item()))
 
    for i, idx in enumerate(chosen_indices):
        z_base = z_real[idx].unsqueeze(0)
        # Small perturbation keeps it near the learned manifold
        z_use  = z_base + TASK1_PERTURB * z_std * torch.randn_like(z_base)
        roll   = ae_model.decode(z_use, TASK1_SEQ_LEN).cpu().numpy()[0]
        generated_rolls.append(roll)
 
        # Convert to MIDI with TASK 1's signature low-velocity, high-threshold character
        pr = roll.T  # (128, seq_len)
 
        # Adaptive threshold: relax only if we have < 8 notes
        threshold = TASK1_THRESHOLD
        if (pr > threshold).sum() < 8:
            threshold = float(np.percentile(pr, 93))
 
        # Build MIDI with fixed soft velocity (Task 1's signature dynamics)
        pm = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)   # Acoustic Grand Piano
        pr_bin = (pr > threshold).astype(np.int8)
        for pitch in range(pr_bin.shape[0]):
            active = False; start = 0
            for t in range(pr_bin.shape[1]):
                if pr_bin[pitch, t] == 1 and not active:
                    active = True; start = t
                elif pr_bin[pitch, t] == 0 and active:
                    active = False
                    piano.notes.append(pretty_midi.Note(
                        velocity=TASK1_VELOCITY,
                        pitch=int(pitch),
                        start=start/cfg.FS,
                        end=t/cfg.FS))
            if active:
                piano.notes.append(pretty_midi.Note(
                    velocity=TASK1_VELOCITY,
                    pitch=int(pitch),
                    start=start/cfg.FS,
                    end=pr_bin.shape[1]/cfg.FS))
        pm.instruments.append(piano)
 
        out_path = f'{cfg.OUTPUT_DIR}/generated_midis/task1_sample_{i+1}.mid'
        pm.write(out_path)
        task1_midi_paths.append(out_path)
 
        # Verify and report
        pm_check = pretty_midi.PrettyMIDI(out_path)
        n_notes = sum(len(inst.notes) for inst in pm_check.instruments)
        duration = pm_check.get_end_time()
        print(f'  ({i+1}/5) {os.path.basename(out_path)} '
              f'| vel={TASK1_VELOCITY} | thresh={threshold:.2f} '
              f'| {n_notes} notes | {duration:.1f}s')
 
# Visualize all 5 piano-rolls
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(generated_rolls[i].T, aspect='auto', origin='lower', cmap='hot')
    ax.set_title(f'AE Sample {i+1}')
    ax.set_xlabel('Time')
    if i == 0: ax.set_ylabel('MIDI Pitch')
plt.suptitle(f'Task 1: 5 Generated Piano-Rolls from LSTM Autoencoder ({cfg.GENRES[single_genre_idx]})',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{cfg.OUTPUT_DIR}/plots/task1_five_samples_viz.png', dpi=130, bbox_inches='tight')
plt.show()
 
print("\n" + "="*60)
print("TASK 1 COMPLETE — All 3 deliverables produced:")
print(f"  [1] Autoencoder code     → models/task1_lstm_ae.pt + task1_architecture.txt")
print(f"  [2] Reconstruction curve → plots/task1_reconstruction_loss_curve.png")
print(f"  [3] 5 MIDI samples       → generated_midis/task1_sample_1..5.mid")
print(f"  Signature: Single-genre, short (8s), soft (vel 70), sparse (thresh 0.55)")
print("="*60)



# ---------- Task2: 8 Multi-Genre Samples ----------
print("\n" + "="*60)
print("TASK 2 — DELIVERABLE 2: 8 Multi-Genre Generated Samples")
print("="*60)
print("VAE signature: longer sequences, denser chords, genre-specific timbre")
print()
 
# === TASK 2 SIGNATURE SETTINGS — deliberately DIFFERENT from Task 1 ===
TASK2_SEQ_LEN      = 192    # LONGER than Task 1's 128 (12s vs 8s)
TASK2_THRESHOLD    = 0.38   # LOWER than Task 1's 0.55 (denser chords)
TASK2_TEMPERATURE  = 1.2    # Sampling temperature (higher = more variety)
 
# Genre-specific MIDI settings — gives each genre unique character
GENRE_SETTINGS = {
    'Classical':  {'program':  0,  'velocity':  65, 'name': 'Acoustic Grand Piano'},
    'Jazz':       {'program':  3,  'velocity':  90, 'name': 'Honky-tonk Piano'},
    'Rock':       {'program': 29,  'velocity': 110, 'name': 'Overdriven Guitar'},
    'Pop':        {'program':  5,  'velocity':  85, 'name': 'Electric Piano'},
    'Electronic': {'program': 81,  'velocity': 100, 'name': 'Synth Lead (Sawtooth)'},
}
 
vae_model.eval()
generated_rolls_vae = []
task2_midi_paths = []
 
with torch.no_grad():
    # Learn genre-specific posterior statistics from real training data
    ref_samples_by_genre = {g: [] for g in range(cfg.N_GENRES)}
    for X_batch, y_batch in pr_train_loader:
        for j in range(X_batch.size(0)):
            g = int(y_batch[j].item())
            if len(ref_samples_by_genre[g]) < 10:
                ref_samples_by_genre[g].append(X_batch[j])
 
    genre_latents = {}
    for g, samples in ref_samples_by_genre.items():
        if len(samples) == 0: continue
        real_X = torch.stack(samples).to(cfg.DEVICE)
        real_y = torch.tensor([g]*len(samples)).to(cfg.DEVICE)
        mu, logvar = vae_model.encode(real_X, real_y)
        genre_latents[g] = {
            'mean': mu.mean(dim=0, keepdim=True),
            'std':  mu.std(dim=0, keepdim=True) + 1e-3,
        }
 
    # Generate 8 samples — cycle through genres for diversity
    for i in range(8):
        g_idx = i % cfg.N_GENRES
        g_name = cfg.GENRES[g_idx]
        setting = GENRE_SETTINGS[g_name]
        genre = torch.tensor([g_idx]).to(cfg.DEVICE)
 
        # Sample from learned genre-specific posterior N(μ_g, (T·σ_g)²)
        if g_idx in genre_latents:
            stats = genre_latents[g_idx]
            z = stats['mean'] + TASK2_TEMPERATURE * stats['std'] * torch.randn(1, cfg.LATENT_DIM).to(cfg.DEVICE)
        else:
            z = TASK2_TEMPERATURE * torch.randn(1, cfg.LATENT_DIM).to(cfg.DEVICE)
 
        gen = vae_model.decode(z, genre, TASK2_SEQ_LEN).cpu().numpy()[0]
        generated_rolls_vae.append((gen, g_idx))
 
        # Adaptive threshold — relax only if too sparse
        threshold = TASK2_THRESHOLD
        if (gen > threshold).sum() < 20:
            threshold = float(np.percentile(gen, 80))
 
        # Build MIDI with genre-specific instrument and velocity
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=setting['program'])
        pr_bin = (gen.T > threshold).astype(np.int8)
        for pitch in range(pr_bin.shape[0]):
            active = False; start = 0
            for t in range(pr_bin.shape[1]):
                if pr_bin[pitch, t] == 1 and not active:
                    active = True; start = t
                elif pr_bin[pitch, t] == 0 and active:
                    active = False
                    inst.notes.append(pretty_midi.Note(
                        velocity=setting['velocity'],
                        pitch=int(pitch),
                        start=start/cfg.FS,
                        end=t/cfg.FS))
            if active:
                inst.notes.append(pretty_midi.Note(
                    velocity=setting['velocity'],
                    pitch=int(pitch),
                    start=start/cfg.FS,
                    end=pr_bin.shape[1]/cfg.FS))
        pm.instruments.append(inst)
 
        out_path = f'{cfg.OUTPUT_DIR}/generated_midis/task2_{g_name}_sample{i+1}.mid'
        pm.write(out_path)
        task2_midi_paths.append(out_path)
 
        # Verify
        pm_check = pretty_midi.PrettyMIDI(out_path)
        n_notes = sum(len(x.notes) for x in pm_check.instruments)
        duration = pm_check.get_end_time()
        print(f'  ({i+1}/8) {g_name:<10s} | {setting["name"]:<22s} | vel={setting["velocity"]:3d} | {n_notes:3d} notes | {duration:.1f}s')
 
# Visualize
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for i, (roll, g_idx) in enumerate(generated_rolls_vae):
    ax = axes[i // 4, i % 4]
    ax.imshow(roll.T, aspect='auto', origin='lower', cmap='hot')
    ax.set_title(f'VAE {i+1}: {cfg.GENRES[g_idx]}')
    ax.set_xlabel('Time'); ax.set_ylabel('Pitch' if i%4==0 else '')
plt.suptitle('Task 2: 8 Multi-Genre Samples from VAE (denser, longer, genre-specific)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{cfg.OUTPUT_DIR}/plots/task2_eight_samples_viz.png', dpi=130, bbox_inches='tight')
plt.show()



# ---------- Task 3: 10 Long Generated Compositions ----------
print("\n" + "="*60)
print("TASK 3 — DELIVERABLE 3: 10 Long Generated Compositions")
print("="*60)

transformer.eval()
generation_metadata = []

for i in range(10):
    g_idx = i % cfg.N_GENRES
    prompt = torch.randint(0, 128, (1, 8)).to(cfg.DEVICE)
    genre  = torch.tensor([g_idx]).to(cfg.DEVICE)

    with torch.no_grad():
        gen = transformer.generate(prompt, genre, max_new=500, temperature=1.0, top_k=40)
    toks = gen[0].cpu().numpy()
    out_path = f'{cfg.OUTPUT_DIR}/generated_midis/task3_{cfg.GENRES[g_idx]}_composition{i+1}.mid'
    tokens_to_midi(toks, out_path)

    try:
        pm = pretty_midi.PrettyMIDI(out_path)
        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        duration = pm.get_end_time()
    except Exception:
        n_notes, duration = 0, 0.0

    generation_metadata.append({
        'idx': i+1, 'genre': cfg.GENRES[g_idx],
        'num_tokens': len(toks), 'num_notes': n_notes,
        'duration_sec': round(duration, 2),
        'file': os.path.basename(out_path),
    })
    print(f"  ({i+1:2d}/10) {cfg.GENRES[g_idx]:<10s} | {len(toks)} tokens | {n_notes} notes | {duration:.1f}s")

pd.DataFrame(generation_metadata).to_csv(
    f'{cfg.OUTPUT_DIR}/task3_generation_metadata.csv', index=False)