"""
Piano-roll representation: MIDI ↔ (128, T) binary matrix.
"""

import os
import random
import numpy as np
from tqdm import tqdm
import pretty_midi

from ..config import cfg


def midi_to_pianoroll(midi_path, fs=None):
    """Convert MIDI to binary piano-roll: shape (128, T)."""
    fs = fs or cfg.FS
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        pr = pm.get_piano_roll(fs=fs)
        return (pr > 0).astype(np.float32)
    except Exception:
        return None


def pianoroll_to_midi(pr, out_path, fs=None, threshold=0.5):
    """Convert piano-roll (128, T) back to a MIDI file."""
    fs = fs or cfg.FS
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    pr_bin = (pr > threshold).astype(np.int8)

    for pitch in range(pr_bin.shape[0]):
        active = False
        start = 0
        for t in range(pr_bin.shape[1]):
            if pr_bin[pitch, t] == 1 and not active:
                active = True
                start = t
            elif pr_bin[pitch, t] == 0 and active:
                active = False
                piano.notes.append(pretty_midi.Note(
                    velocity=80, pitch=pitch,
                    start=start/fs, end=t/fs))
        if active:
            piano.notes.append(pretty_midi.Note(
                velocity=80, pitch=pitch,
                start=start/fs, end=pr_bin.shape[1]/fs))

    pm.instruments.append(piano)
    pm.write(out_path)
    return out_path


def build_pianoroll_dataset(files, seq_len=None, max_per_file=8):
    """Extract fixed-length piano-roll windows with 50% overlap stride."""
    seq_len = seq_len or cfg.SEQ_LEN
    segments, genres = [], []
    skipped = 0
    for f in tqdm(files, desc='Piano-roll'):
        pr = midi_to_pianoroll(f)
        if pr is None or pr.shape[1] < seq_len:
            skipped += 1
            continue
        g_name = os.path.basename(f).split('_')[0]
        g_idx  = cfg.GENRES.index(g_name) if g_name in cfg.GENRES else random.randint(0, cfg.N_GENRES - 1)
        n = 0
        for start in range(0, pr.shape[1] - seq_len, seq_len // 2):
            segments.append(pr[:, start:start + seq_len].T)  # (seq_len, 128)
            genres.append(g_idx)
            n += 1
            if n >= max_per_file:
                break
    print(f"Piano-roll: {len(segments)} segments | {skipped} files skipped")
    if len(segments) == 0:
        raise RuntimeError("No piano-roll segments extracted! Check data directory.")
    return np.array(segments, dtype=np.float32), np.array(genres, dtype=np.int64)
