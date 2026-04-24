"""
MIDI file discovery and batch parsing utilities.
"""

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pretty_midi

from ..config import cfg


def find_midi_files(root_dir):
    """Recursively discover all .mid/.midi files under root_dir."""
    if not os.path.exists(root_dir):
        return []
    files  = glob.glob(os.path.join(root_dir, '**', '*.mid'),  recursive=True)
    files += glob.glob(os.path.join(root_dir, '**', '*.midi'), recursive=True)
    return sorted(files)


def parse_midi_dataset(root_dir, max_files=None):
    """Parse all MIDI files and return valid PrettyMIDI objects + their paths."""
    files = find_midi_files(root_dir)
    if max_files is not None:
        files = files[:max_files]

    parsed = []
    for f in tqdm(files, desc='Parsing MIDI'):
        try:
            pm = pretty_midi.PrettyMIDI(f)
            if len(pm.instruments) == 0:
                continue
            parsed.append((f, pm))
        except Exception:
            continue
    print(f"Successfully parsed {len(parsed)}/{len(files)} MIDI files.")
    return parsed


def infer_genre(filename, genres=None):
    """Heuristic: genre inferred from filename prefix (e.g., 'Classical_01.mid')."""
    if genres is None:
        genres = cfg.GENRES
    basename = os.path.basename(filename).split('_')[0]
    return genres.index(basename) if basename in genres else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default=cfg.RAW_MIDI_DIR)
    parser.add_argument('--output', default=cfg.PROCESSED_DIR)
    parser.add_argument('--max_files', type=int, default=cfg.MAX_FILES)
    args = parser.parse_args()

    cfg.make_dirs()
    parsed = parse_midi_dataset(args.input, max_files=args.max_files)

    # Save the list of successfully parsed file paths
    out_list = os.path.join(args.output, 'parsed_files.txt')
    with open(out_list, 'w') as f:
        for path, _ in parsed:
            f.write(path + '\n')
    print(f"Saved parsed file list → {out_list}")


if __name__ == '__main__':
    main()
