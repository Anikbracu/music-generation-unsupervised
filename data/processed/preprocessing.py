def midi_to_pianoroll(midi_path, fs=16):
    '''Convert MIDI to binary piano-roll matrix: shape (128, T).'''
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        pr = pm.get_piano_roll(fs=fs)
        return (pr > 0).astype(np.float32)
    except Exception:
        return None


def midi_to_tokens(midi_path, max_tokens=8192):
    '''Event-based tokenizer:
       [0..127]   = NOTE_ON
       [128..255] = NOTE_OFF
       [256..287] = VELOCITY (32 bins)
       [288..412] = TIME_SHIFT (125 bins of 10ms each)
    '''
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

    events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

    tokens, last_t = [], 0.0
    for t, kind, pitch, vel in events:
        dt = t - last_t
        # Break long time gaps into multiple time-shift tokens
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
    '''Reconstruct MIDI file from token sequence.'''
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    current_t  = 0.0
    current_vel = 64
    active = {}

    for tok in tokens:
        tok = int(tok)
        if 0 <= tok < 128:        # NOTE_ON
            active[tok] = (current_t, current_vel)
        elif 128 <= tok < 256:    # NOTE_OFF
            p = tok - 128
            if p in active:
                start, vel = active.pop(p)
                if current_t > start:
                    piano.notes.append(pretty_midi.Note(
                        velocity=int(vel), pitch=int(p),
                        start=start, end=current_t))
        elif 256 <= tok < 288:    # VELOCITY
            current_vel = max(1, min(127, (tok - 256) * 4))
        elif 288 <= tok < 413:    # TIME_SHIFT
            current_t += (tok - 288) / 100.0

    # Close any still-active notes
    for p, (start, vel) in active.items():
        piano.notes.append(pretty_midi.Note(
            velocity=int(vel), pitch=int(p),
            start=start, end=current_t + 0.25))

    pm.instruments.append(piano)
    pm.write(out_path)
    return out_path


def pianoroll_to_midi(pr, out_path, fs=16, threshold=0.5):
    '''Convert piano-roll (128, T) to MIDI file.'''
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    pr_bin = (pr > threshold).astype(np.int8)

    for pitch in range(pr_bin.shape[0]):
        active = False
        start  = 0
        for t in range(pr_bin.shape[1]):
            if pr_bin[pitch, t] == 1 and not active:
                active = True
                start  = t
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


print("Preprocessing functions ready.")