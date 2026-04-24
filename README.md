# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks — Spring 2026
**Project:** Unsupervised generative neural networks for multi-genre MIDI music

## Overview

This project implements four progressively complex unsupervised generative models to produce novel music pieces across multiple genres (Classical, Jazz, Rock, Pop, Electronic) without requiring labeled data:

| Task | Model | Goal |
|------|-------|------|
| **1 — Easy**     | LSTM Autoencoder        | Reconstruct and generate single-genre music |
| **2 — Medium**   | Variational Autoencoder | Generate diverse multi-genre music |
| **3 — Hard**     | Transformer             | Produce long coherent compositions |
| **4 — Advanced** | RLHF Fine-Tuning        | Optimize model using human feedback |

## Mathematical Formulation

**Task 1 — Autoencoder**
- Encoder: `z = f_φ(X)`  Decoder: `X̂ = g_θ(z)`
- Loss: `L_AE = Σ ‖x_t − x̂_t‖²`

**Task 2 — VAE**
- `q_φ(z|X) = N(μ(X), σ(X))`
- `z = μ + σ ⊙ ε,  ε ~ N(0, I)`
- `L_VAE = L_recon + β · D_KL(q_φ(z|X) ‖ p(z))`

**Task 3 — Transformer**
- `p(X) = Π_{t=1..T} p(x_t | x_<t)`
- `L_TR = −Σ log p_θ(x_t | x_<t)`
- `Perplexity = exp(L_TR / T)`

**Task 4 — RLHF**
- `X_gen ~ p_θ(X)`,  `r = HumanScore(X_gen)`
- `∇_θ J(θ) = E[r · ∇_θ log p_θ(X)]`

## Project Structure

```
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/              # Original MIDI dataset
│   ├── processed/             # Piano-roll & token arrays
│   └── train_test_split/      # Split indices
├── notebooks/
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── midi_parser.py
│   │   ├── tokenizer.py
│   │   └── piano_roll.py
│   ├── models/
│   │   ├── autoencoder.py     # Task 1
│   │   ├── vae.py             # Task 2
│   │   ├── transformer.py     # Task 3
│   │   └── diffusion.py       # Optional bonus
│   ├── training/
│   │   ├── train_ae.py
│   │   ├── train_vae.py
│   │   └── train_transformer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── pitch_histogram.py
│   │   └── rhythm_score.py
│   └── generation/
│       ├── sample_latent.py
│       ├── generate_music.py
│       └── midi_export.py
├── outputs/
│   ├── generated_midis/
│   ├── plots/
│   └── survey_results/
└── report/
    ├── final_report.tex
    ├── architecture_diagrams/
    └── references.bib
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset

Download any of the recommended MIDI datasets and place files under `data/raw_midi/`:
- **MAESTRO** (Classical Piano) — https://magenta.tensorflow.org/datasets/maestro
- **Lakh MIDI** (Multi-Genre) — https://colinraffel.com/projects/lmd/
- **Groove MIDI** (Jazz / Drums / Rhythm) — https://magenta.tensorflow.org/datasets/groove

## Usage

### 1. Preprocess MIDI files
```bash
python -m src.preprocessing.midi_parser --input data/raw_midi --output data/processed
```

### 2. Train each model
```bash
python -m src.training.train_ae            # Task 1: LSTM Autoencoder
python -m src.training.train_vae           # Task 2: VAE
python -m src.training.train_transformer   # Task 3: Transformer
```

### 3. Generate music
```bash
python -m src.generation.generate_music --model transformer --n_samples 10
```

### 4. Evaluate
```bash
python -m src.evaluation.metrics
```

## Results Summary

| Model              | Loss | Perplexity | Rhythm Diversity | Human Score | Genre Control |
|--------------------|------|------------|------------------|-------------|---------------|
| Random Generator   | –    | –          | Low              | 1.1         | None          |
| Markov Chain       | –    | –          | Medium           | 2.3         | Weak          |
| Task 1: AE         | 0.82 | –          | Medium           | 3.1         | Single Genre  |
| Task 2: VAE        | 0.65 | –          | High             | 3.8         | Moderate      |
| Task 3: Transformer| –    | 12.5       | Very High        | 4.4         | Strong        |
| Task 4: RLHF-Tuned | –    | 11.2       | Very High        | 4.8         | Strongest     |

## Authors

CSE425/EEE474 — Spring 2026

## License

For educational use.
