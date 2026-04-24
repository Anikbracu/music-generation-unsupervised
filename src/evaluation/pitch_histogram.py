"""
Pitch-class histogram similarity.

    H(p, q) = Σ |p_i − q_i|    (lower = more similar)
"""

import numpy as np
import pretty_midi


def pitch_histogram(midi_path):
    """Return normalized 12-bin pitch-class histogram."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    hist = np.zeros(12)
    for inst in pm.instruments:
        for n in inst.notes:
            hist[n.pitch % 12] += 1
    s = hist.sum()
    return hist / s if s > 0 else hist


def pitch_histogram_similarity(p, q):
    """L1 distance between two pitch histograms (0 = identical)."""
    return float(np.sum(np.abs(p - q)))
