# ---------- DELIVERABLE 2: Reward Scoring Function ----------
print("\n" + "="*60)
print("TASK 4 — DELIVERABLE 2: Reward Scoring Function")
print("="*60)

class RewardModel:
    '''Weighted combination of musical features.
       r(X_gen) = Σ_i w_i · f_i(X_gen)
    '''
    def __init__(self):
        self.weights = {
            'pitch_variety':    0.25,
            'non_repetition':   0.20,
            'pitch_range':      0.15,
            'valid_tokens':     0.15,
            'rhythm_presence':  0.10,
            'note_density':     0.15,
        }

    def __call__(self, tokens):
        return self.score(tokens)

    def score(self, tokens):
        tokens = np.asarray(tokens)
        if len(tokens) < 10:
            return 0.0

        note_on = tokens[(tokens >= 0)   & (tokens < 128)]
        times   = tokens[(tokens >= 288) & (tokens < 413)]

        pitch_variety = len(np.unique(note_on)) / max(len(note_on), 1) if len(note_on) else 0

        if len(note_on) > 1:
            non_rep = float(np.mean(np.diff(note_on) != 0))
        else:
            non_rep = 0.0

        in_range = float(np.mean((note_on >= 36) & (note_on <= 96))) if len(note_on) else 0
        valid    = float(np.mean(tokens < cfg.VOCAB_SIZE))
        rhythm_p = min(len(times) / max(len(tokens), 1) * 3, 1.0)

        density  = len(note_on) / max(len(tokens), 1)
        note_den = max(0, 1 - abs(density - 0.3) * 3)

        return float(
            self.weights['pitch_variety']   * pitch_variety +
            self.weights['non_repetition']  * non_rep       +
            self.weights['pitch_range']     * in_range      +
            self.weights['valid_tokens']    * valid         +
            self.weights['rhythm_presence'] * rhythm_p      +
            self.weights['note_density']    * note_den
        )

    def describe(self):
        s = "Reward function: weighted sum of musical features.\n"
        for k, v in self.weights.items():
            s += f"  {k:<18s} weight = {v:.2f}\n"
        return s

reward_model = RewardModel()
print(reward_model.describe())

with open(f'{cfg.OUTPUT_DIR}/task4_reward_function.txt', 'w') as f:
    f.write("TASK 4: REWARD SCORING FUNCTION\n" + "="*50 + "\n")
    f.write(reward_model.describe())
    f.write("\nMath: r(X_gen) = Σ_i w_i · f_i(X_gen)\n")

# Pre-RLHF baseline
print("Computing pre-RLHF reward baseline...")
pre_rewards = []
transformer.eval()
with torch.no_grad():
    for _ in range(30):
        p = torch.randint(0, 128, (1, 8)).to(cfg.DEVICE)
        g = torch.randint(0, cfg.N_GENRES, (1,)).to(cfg.DEVICE)
        gen = transformer.generate(p, g, max_new=200, temperature=1.0, top_k=40)
        pre_rewards.append(reward_model(gen[0].cpu().numpy()))
print(f"Pre-RLHF avg reward: {np.mean(pre_rewards):.4f} ± {np.std(pre_rewards):.4f}")