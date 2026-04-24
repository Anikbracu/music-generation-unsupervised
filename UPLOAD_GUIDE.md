# GitHub Upload Guide

Complete step-by-step instructions for uploading this project to GitHub.
Choose **ONE** of the three methods below.

---

## Method 1 — GitHub Web UI (easiest, no tools needed)

### Steps
1. Go to **https://github.com/new**
2. Repository name: `music-generation-unsupervised`
3. Description: *"Unsupervised Neural Network for Multi-Genre Music Generation — CSE425/EEE474"*
4. Set to **Public** (or Private if required by your course)
5. **Do NOT** check "Initialize with README" (we already have one)
6. Click **Create repository**

### Upload files
1. On the empty repository page, click **"uploading an existing file"**
2. Drag and drop the **entire** `music-generation-unsupervised` folder contents
   - Keeps folder structure automatically
   - GitHub may require you to upload in chunks if > 100 files
3. Scroll down, write commit message: `Initial project upload`
4. Click **Commit changes**

### Add each folder separately (if drag-drop fails)
Use **Add file → Create new file**, type `folder-name/filename.ext` in the filename box (GitHub auto-creates nested folders from the slash).

---

## Method 2 — Git command line (recommended)

### Install Git
- **Windows**: https://git-scm.com/download/win
- **Mac**: `brew install git`
- **Linux**: `sudo apt install git`

### One-time setup
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Upload commands
```bash
# 1. Navigate to the project folder
cd path/to/music-generation-unsupervised

# 2. Initialize git and make first commit
git init
git add .
git commit -m "Initial project upload"

# 3. Create an empty repo on GitHub (github.com/new), then link it
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/music-generation-unsupervised.git

# 4. Push
git push -u origin main
```

If it asks for login, use a **Personal Access Token** (not password):
1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Tick **repo** scope → Generate → copy the token
4. Paste it as the password when prompted

---

## Method 3 — GitHub Desktop (GUI)

1. Download and install **GitHub Desktop** from https://desktop.github.com
2. Sign in with your GitHub account
3. **File → Add local repository** → select the `music-generation-unsupervised` folder
4. When prompted, click **"create a repository"**
5. Fill in the name, description, leave defaults → **Create repository**
6. In the main window, review changes → write summary `Initial project upload`
7. Click **Commit to main**
8. Click **Publish repository** (top bar) → choose public/private → **Publish**

---

## What gets uploaded vs ignored

### Uploaded (tracked in git)
- All `.py` source files
- `.ipynb` notebooks
- `README.md`, `requirements.txt`, `.gitignore`
- `report/final_report.tex`, `report/references.bib`
- Generated CSV files in `outputs/survey_results/`

### NOT uploaded (in `.gitignore`)
- Raw MIDI files in `data/raw_midi/` (too large)
- Trained model files `.pt` (too large, can exceed GitHub's 100 MB limit)
- Generated MIDI files (regeneratable)
- Python cache directories
- LaTeX build artifacts

### Why this matters
GitHub **rejects single files > 100 MB**. Trained `.pt` checkpoints and large MIDI corpora often exceed this. Keep them out of git and reference them in the README (Kaggle Datasets, Hugging Face, Google Drive).

---

## After pushing: verify on GitHub

Your repo structure should look exactly like this on github.com:

```
music-generation-unsupervised/
├── .gitignore
├── README.md
├── requirements.txt
├── UPLOAD_GUIDE.md
├── data/
│   ├── raw_midi/            (empty, has .gitkeep)
│   ├── processed/
│   └── train_test_split/
├── notebooks/
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── generation/
├── outputs/
│   ├── generated_midis/
│   ├── plots/
│   └── survey_results/
│       ├── task4_listening_survey_template.csv
│       ├── task4_listening_survey_SIMULATED.csv
│       └── task4_before_vs_after_comparison.csv
└── report/
    ├── final_report.tex
    ├── references.bib
    └── architecture_diagrams/
```

---

## Final checks before submission

1. ✅ Repository is **public** (or shared with your instructor if private)
2. ✅ `README.md` shows the correct structure on the GitHub landing page
3. ✅ All `.py` files are syntactically valid (they are — already verified)
4. ✅ CSV survey data is present under `outputs/survey_results/`
5. ✅ Share the repo URL in your final report and submission form

---

## Commands cheat sheet (for updates after first push)

```bash
# Add new/changed files
git add .
git commit -m "short description of change"
git push

# Check what's changed
git status

# View history
git log --oneline
```

---

## Troubleshooting

**"Large file" error on push** — a file exceeds GitHub's 100 MB limit.
```bash
# Find large files
find . -size +50M
# Remove from tracking (keeps local copy)
git rm --cached path/to/big/file
# Add the path to .gitignore, commit, push again
```

**Authentication rejected** — password auth is disabled on GitHub since 2021. Use a Personal Access Token as described under Method 2.

**Push rejected: "refusing to merge unrelated histories"** — happens if you initialized the GitHub repo with a README. Fix:
```bash
git pull origin main --allow-unrelated-histories
git push
```
