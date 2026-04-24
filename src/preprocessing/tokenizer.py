"""
Event-based MIDI tokenizer.

Vocabulary (413 tokens):
    [0..127]   = NOTE_ON
    [128..255] = NOTE_OFF
    [256..287] = VELOCITY (32 bins)
    [288..412] = TIME_SHIFT (125 bins × 10 ms)
"""

import os
import random
import numpy as np
from tqdm import tqdm
import pretty_midi

from ..config import cfg


def midi_to_tokens(midi_path, max_tokens=8192):
    """Convert MIDI to token sequence."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None

    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            events.append((note.start, 'on',  note.pitch, note.velocity))
            events.append((note.end,   'off', note.pitch, 0))

    if len(events) < 10:
        return None

    # sort by time, with NOTE_OFF before NOTE_ON at same timestamp
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

    tokens, last_t = [], 0.0
    for t, kind, pitch, vel in events:
        dt = t - last_t
        while dt > 0:
            steps = min(int(dt * 100), 124)
            if steps <= 0:
                break
            tokens.append(288 + steps)
            dt -= steps / 100.0

        if kind == 'on':
            vel_bin = min(int(vel / 4), 31)
            tokens.append(256 + vel_bin)
            tokens.append(int(pitch))
        else:
            tokens.append(128 + int(pitch))
        last_t = t

        if len(tokens) >= max_tokens:
            break

    return np.array(tokens, dtype=np.int64)


def tokens_to_midi(tokens, out_path, tempo=120):
    """Reconstruct MIDI from a token sequence."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    current_t = 0.0
    current_vel = 64
    active = {}

    for tok in tokens:
        tok = int(tok)
        if 0 <= tok < 128:            # NOTE_ON
            active[tok] = (current_t, current_vel)
        elif 128 <= tok < 256:        # NOTE_OFF
            p = tok - 128
            if p in active:
                start, vel = active.pop(p)
                if current_t > start:
                    piano.notes.append(pretty_midi.Note(
                        velocity=int(vel), pitch=int(p),
                        start=start, end=current_t))
        elif 256 <= tok < 288:        # VELOCITY
            current_vel = max(1, min(127, (tok - 256) * 4))
        elif 288 <= tok < 413:        # TIME_SHIFT
            current_t += (tok - 288) / 100.0

    # close any still-active notes
    for p, (start, vel) in active.items():
        piano.notes.append(pretty_midi.Note(
            velocity=int(vel), pitch=int(p),
            start=start, end=current_t + 0.25))

    pm.instruments.append(piano)
    pm.write(out_path)
    return out_path


def build_token_dataset(files, seq_len=256, max_per_file=6):
    """Extract fixed-length token windows with 50% overlap stride."""
    sequences, genres = [], []
    skipped_none = skipped_short = 0
    token_counts = []

    for f in tqdm(files, desc='Tokenization'):
        toks = midi_to_tokens(f, max_tokens=8192)
        if toks is None:
            skipped_none += 1
            continue
        token_counts.append(len(toks))
        if len(toks) < seq_len + 1:
            skipped_short += 1
            continue

        g_name = os.path.basename(f).split('_')[0]
        g_idx  = cfg.GENRES.index(g_name) if g_name in cfg.GENRES else random.randint(0, cfg.N_GENRES - 1)
        n = 0
        for start in range(0, len(toks) - seq_len, seq_len // 2):
            sequences.append(toks[start:start + seq_len])
            genres.append(g_idx)
            n += 1
            if n >= max_per_file:
                break

    avg = np.mean(token_counts) if token_counts else 0
    print(f"Tokens: {len(sequences)} seqs | avg {avg:.0f} tokens/file | skipped {skipped_none} unparseable, {skipped_short} too short")

    if len(sequences) == 0:
        raise RuntimeError("No token sequences extracted!")

    return np.array(sequences, dtype=np.int64), np.array(genres, dtype=np.int64)
