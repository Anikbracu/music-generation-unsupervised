"""
End-to-end music generation CLI.

Usage:
    python -m src.generation.generate_music --model transformer --n_samples 10
    python -m src.generation.generate_music --model vae --n_samples 8
    python -m src.generation.generate_music --model ae --n_samples 5
"""

import os
import argparse
import torch

from ..config import cfg
from ..models import LSTMAutoencoder, MusicVAE, MusicTransformer
from ..preprocessing import pianoroll_to_midi, tokens_to_midi
from .sample_latent import sample_from_latent


def generate_ae(n_samples=5):
    """Task 1: Generate with LSTM Autoencoder."""
    model = LSTMAutoencoder(cfg.N_PITCHES, cfg.HIDDEN_DIM,
                             cfg.LATENT_DIM, cfg.NUM_LAYERS).to(cfg.DEVICE)
    ckpt = os.path.join(cfg.MODEL_DIR, 'task1_lstm_ae.pt')
    model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
    rolls = sample_from_latent(model, n_samples=n_samples)

    paths = []
    for i, roll in enumerate(rolls):
        p = os.path.join(cfg.MIDI_OUTPUT, f'task1_sample_{i+1}.mid')
        pianoroll_to_midi(roll.T, p, fs=cfg.FS)
        paths.append(p)
        print(f"  Saved {p}")
    return paths


def generate_vae(n_samples=8):
    """Task 2: Generate with VAE, cycling through genres."""
    model = MusicVAE(cfg.N_PITCHES, cfg.HIDDEN_DIM, cfg.LATENT_DIM,
                      cfg.NUM_LAYERS, cfg.N_GENRES).to(cfg.DEVICE)
    ckpt = os.path.join(cfg.MODEL_DIR, 'task2_vae.pt')
    model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    paths = []
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            g_idx = i % cfg.N_GENRES
            z     = torch.randn(1, cfg.LATENT_DIM).to(cfg.DEVICE)
            genre = torch.tensor([g_idx]).to(cfg.DEVICE)
            roll  = model.decode(z, genre, cfg.SEQ_LEN).cpu().numpy()[0]
            p = os.path.join(cfg.MIDI_OUTPUT, f'task2_{cfg.GENRES[g_idx]}_sample{i+1}.mid')
            pianoroll_to_midi(roll.T, p, fs=cfg.FS)
            paths.append(p)
            print(f"  Saved {p}")
    return paths


def generate_transformer(n_samples=10, max_new=500):
    """Task 3: Generate with Transformer."""
    model = MusicTransformer(
        vocab_size=cfg.VOCAB_SIZE, d_model=cfg.EMBED_DIM,
        n_heads=cfg.NUM_HEADS, n_layers=4, max_len=512,
        n_genres=cfg.N_GENRES).to(cfg.DEVICE)
    ckpt = os.path.join(cfg.MODEL_DIR, 'task3_transformer.pt')
    model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))

    paths = []
    for i in range(n_samples):
        g_idx = i % cfg.N_GENRES
        prompt = torch.randint(0, 128, (1, 8)).to(cfg.DEVICE)
        genre  = torch.tensor([g_idx]).to(cfg.DEVICE)
        with torch.no_grad():
            gen = model.generate(prompt, genre, max_new=max_new,
                                  temperature=1.0, top_k=40)
        toks = gen[0].cpu().numpy()
        p = os.path.join(cfg.MIDI_OUTPUT, f'task3_{cfg.GENRES[g_idx]}_composition{i+1}.mid')
        tokens_to_midi(toks, p)
        paths.append(p)
        print(f"  Saved {p} ({len(toks)} tokens)")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                         choices=['ae', 'vae', 'transformer'])
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args()

    cfg.make_dirs()

    if args.model == 'ae':
        generate_ae(args.n_samples)
    elif args.model == 'vae':
        generate_vae(args.n_samples)
    elif args.model == 'transformer':
        generate_transformer(args.n_samples)


if __name__ == '__main__':
    main()
