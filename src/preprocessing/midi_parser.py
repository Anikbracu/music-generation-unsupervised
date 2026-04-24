def find_midi_files(root):
    if not os.path.exists(root):
        return []
    files  = glob.glob(os.path.join(root, '**', '*.mid'),  recursive=True)
    files += glob.glob(os.path.join(root, '**', '*.midi'), recursive=True)
    return files


def generate_synthetic_midi(n_files=200, out_dir='/kaggle/working/synthetic_midi'):
    '''Fallback: generate simple synthetic MIDIs across pseudo-genres.'''
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    scales = {
        'Classical':  [60, 62, 64, 65, 67, 69, 71, 72],
        'Jazz':       [60, 63, 65, 66, 67, 70, 72, 75],
        'Rock':       [60, 62, 64, 67, 69, 72, 74, 76],
        'Pop':        [60, 62, 64, 65, 67, 69, 71, 72],
        'Electronic': [60, 61, 64, 66, 68, 70, 72, 74],
    }
    genres = list(scales.keys())
    for i in range(n_files):
        g  = genres[i % len(genres)]
        pm = pretty_midi.PrettyMIDI(initial_tempo=random.choice([90,110,120,140]))
        inst = pretty_midi.Instrument(program=0)
        t = 0.0
        for _ in range(80):
            pitch = random.choice(scales[g]) + random.choice([-12, 0, 0, 12])
            dur   = random.choice([0.25, 0.5, 0.5, 1.0])
            inst.notes.append(pretty_midi.Note(
                velocity=random.randint(60,100),
                pitch=max(0,min(127,pitch)),
                start=t, end=t+dur))
            t += dur
        pm.instruments.append(inst)
        p = os.path.join(out_dir, f'{g}_{i}.mid')
        pm.write(p)
        paths.append(p)
    return paths


# ----- Locate MIDI files -----
midi_files = find_midi_files(cfg.MIDI_DIR)
if len(midi_files) < 10:
    print(f"[!] No real dataset at {cfg.MIDI_DIR}. Using synthetic MIDIs.")
    midi_files = generate_synthetic_midi(n_files=200)
else:
    midi_files = midi_files[:cfg.MAX_FILES]

print(f"Using {len(midi_files)} MIDI files.")
print(f"Sample: {midi_files[:2]}")