"""
Unified evaluation: compute all metrics on a folder of generated MIDIs.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

from ..config import cfg
from .pitch_histogram import pitch_histogram, pitch_histogram_similarity
from .rhythm_score import rhythm_diversity, repetition_ratio


def build_reference_histogram(ref_dir, max_files=50):
    """Aggregate reference pitch histogram from training corpus."""
    files = glob.glob(os.path.join(ref_dir, '**', '*.mid'), recursive=True)[:max_files]
    ref = np.zeros(12); count = 0
    for f in files:
        h = pitch_histogram(f)
        if h is not None:
            ref += h; count += 1
    return ref / count if count else ref


def evaluate_folder(folder, ref_hist):
    """Compute per-file metrics for all MIDIs in a folder."""
    results = []
    for f in sorted(glob.glob(os.path.join(folder, '*.mid'))):
        h = pitch_histogram(f)
        results.append({
            'file':            os.path.basename(f),
            'pitch_hist_dist': pitch_histogram_similarity(h, ref_hist) if h is not None else np.nan,
            'rhythm_div':      rhythm_diversity(f),
            'repetition':      repetition_ratio(f),
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', default=cfg.MIDI_OUTPUT,
                         help='Folder with generated MIDI files')
    parser.add_argument('--ref_dir', default=cfg.RAW_MIDI_DIR,
                         help='Reference training MIDI folder')
    args = parser.parse_args()

    print(f"Building reference histogram from {args.ref_dir}...")
    ref = build_reference_histogram(args.ref_dir)

    print(f"Evaluating generated MIDIs in {args.gen_dir}...")
    df = evaluate_folder(args.gen_dir, ref)
    if df.empty:
        print("No generated MIDIs found.")
        return

    print("\n=== Per-file metrics ===")
    print(df.round(4).to_string(index=False))

    # Group by task prefix (task1_, task2_, etc.)
    df['task'] = df['file'].apply(lambda x: x.split('_')[0])
    summary = df.groupby('task').agg({
        'pitch_hist_dist': 'mean',
        'rhythm_div':      'mean',
        'repetition':      'mean',
    }).round(4)
    print("\n=== Summary per task ===")
    print(summary)

    out_csv = os.path.join(cfg.OUTPUT_DIR, 'evaluation_metrics.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == '__main__':
    main()
