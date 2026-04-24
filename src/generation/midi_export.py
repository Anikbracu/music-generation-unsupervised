"""
Thin wrappers that re-export MIDI utilities for generation scripts.
"""

from ..preprocessing.piano_roll import pianoroll_to_midi
from ..preprocessing.tokenizer import tokens_to_midi


def export_midi(data, out_path, mode='pianoroll', fs=16, threshold=0.5):
    """Unified MIDI export.

    mode='pianoroll' → `data` is (128, T) piano-roll array
    mode='tokens'    → `data` is 1-D token sequence
    """
    if mode == 'pianoroll':
        return pianoroll_to_midi(data, out_path, fs=fs, threshold=threshold)
    elif mode == 'tokens':
        return tokens_to_midi(data, out_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
