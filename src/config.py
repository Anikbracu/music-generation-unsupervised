"""
Global configuration for the music generation project.
"""

import os
import torch


class Config:
    # ---- Paths ----
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR        = os.path.join(PROJECT_ROOT, 'data')
    RAW_MIDI_DIR    = os.path.join(DATA_DIR, 'raw_midi')
    PROCESSED_DIR   = os.path.join(DATA_DIR, 'processed')
    SPLIT_DIR       = os.path.join(DATA_DIR, 'train_test_split')
    OUTPUT_DIR      = os.path.join(PROJECT_ROOT, 'outputs')
    MIDI_OUTPUT     = os.path.join(OUTPUT_DIR, 'generated_midis')
    PLOT_OUTPUT     = os.path.join(OUTPUT_DIR, 'plots')
    SURVEY_OUTPUT   = os.path.join(OUTPUT_DIR, 'survey_results')
    MODEL_DIR       = os.path.join(OUTPUT_DIR, 'models')

    # ---- Data processing ----
    SEQ_LEN     = 128           # piano-roll window length
    N_PITCHES   = 128           # MIDI pitch range
    FS          = 16            # 16 time-steps per bar
    MAX_FILES   = 150           # cap for training (increase for production)

    # Tokenizer vocab: NOTE_ON(128) + NOTE_OFF(128) + VELOCITY(32) + TIME_SHIFT(125) = 413
    VOCAB_SIZE  = 413

    # ---- Training hyperparameters ----
    BATCH_SIZE  = 32
    LR          = 1e-3
    EPOCHS_AE   = 15
    EPOCHS_VAE  = 20
    EPOCHS_TR   = 25
    EPOCHS_RL   = 10

    # ---- Model architecture ----
    HIDDEN_DIM  = 256
    LATENT_DIM  = 128
    NUM_LAYERS  = 2
    EMBED_DIM   = 256
    NUM_HEADS   = 4

    # ---- VAE specific ----
    BETA        = 0.5           # KL weight

    # ---- Genres ----
    GENRES      = ['Classical', 'Jazz', 'Rock', 'Pop', 'Electronic']
    N_GENRES    = 5

    # ---- Misc ----
    SEED        = 42
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def make_dirs(cls):
        """Create all output directories if they don't exist."""
        for d in (cls.PROCESSED_DIR, cls.SPLIT_DIR,
                  cls.MIDI_OUTPUT, cls.PLOT_OUTPUT,
                  cls.SURVEY_OUTPUT, cls.MODEL_DIR):
            os.makedirs(d, exist_ok=True)


cfg = Config()
