"""
Rhythm metrics.

    Rhythm Diversity = # unique durations / # total notes
    Repetition Ratio = # repeated patterns / # total patterns
"""

from collections import Counter
import pretty_midi


def rhythm_diversity(midi_path):
    """Fraction of unique note durations in the piece."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return 0.0
    durs = [round(n.end - n.start, 2)
            for inst in pm.instruments for n in inst.notes]
    if not durs:
        return 0.0
    return len(set(durs)) / len(durs)


def repetition_ratio(midi_path, window=4):
    """Fraction of pitch-patterns that repeat elsewhere in the piece."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return 0.0
    pitches = [n.pitch
               for inst in pm.instruments
               for n in sorted(inst.notes, key=lambda x: x.start)]
    if len(pitches) < window * 2:
        return 0.0
    patterns = [tuple(pitches[i:i + window])
                for i in range(len(pitches) - window)]
    if not patterns:
        return 0.0
    c = Counter(patterns)
    repeats = sum(v for v in c.values() if v > 1)
    return repeats / len(patterns)
