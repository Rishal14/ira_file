# EchoFind — Query-by-Example Speech Retrieval

A minimal Query-by-Example (QbE) speech retrieval demo using Wav2Vec2 embeddings and FAISS.

This repository contains a small Streamlit app that lets you upload a short audio clip and returns the most similar audio files from a small dataset.

Contents

- `utils.py` — audio loading and Wav2Vec2 embedding utilities
- `extract_and_index.py` — compute embeddings for files in `data/wavs/` and build a FAISS index
- `search_app.py` — Streamlit search UI (upload query and view matches)
- `generate_sample_wavs.py` — creates 3 short synthetic WAV files for testing
- `data/wavs/` — put your dataset audio files here (wav, mp3, flac, m4a)
- `index_files/` — FAISS index and embeddings (created by the indexer)
- `requirements.txt` — Python dependencies (see notes below)

Quick start (Windows PowerShell)

1) (Recommended) Use Python 3.11

Many ML packages (PyTorch, FAISS) are best supported on Python 3.11. If you have Python 3.13 you may run into wheel/build issues. Install Python 3.11 if possible.

2) Create & activate a virtual environment

```powershell
cd C:\Users\dell\ira_file\echo_find
python -m venv venv
# activate venv in PowerShell
.\venv\Scripts\Activate.ps1
```

3) Install dependencies (pip)

```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Note about FAISS on Windows

- `faiss-cpu` may not have a compatible pre-built wheel for every Python+OS combination on PyPI. If `pip install -r requirements.txt` fails when trying to install `faiss-cpu`, use conda (recommended on Windows):

```powershell
# if you have conda/miniconda installed
conda create -n echo_find python=3.11 -y
conda activate echo_find
conda install -c conda-forge faiss-cpu -y
# then install the rest with pip
pip install -r requirements.txt
```

Or try lower/higher `faiss-cpu` versions if needed. Conda-forge is the most reliable source for FAISS on Windows.

4) (Optional) Generate sample WAVs for quick testing

```powershell
python generate_sample_wavs.py
# this writes three files to data/wavs/
Get-ChildItem .\data\wavs\
```

5) Build embeddings & FAISS index

```powershell
python extract_and_index.py
```

This script will:
- Walk `data/wavs/` and compute embeddings using the model specified in `utils.py`.
- Save embeddings to `index_files/embeddings.npy`
- Write metadata to `data/metadata.json`
- Write FAISS index to `index_files/faiss.index`

If the script prints `No embeddings computed`, make sure your audio files are in `data/wavs/` and readable.

6) Run the Streamlit search app

```powershell
streamlit run search_app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501). Upload a short query clip (0.5–6s recommended) and view the top matches.

Model and embedding notes

- Default model: `facebook/wav2vec2-base-960h` (set in `utils.py` via `MODEL_NAME`).
- You may see a harmless warning like:

  > Some weights of Wav2Vec2Model were not initialized from the model checkpoint ...

  This means a small task head was randomly initialized; it does NOT prevent using the model as a feature extractor. The last_hidden_state outputs are valid and can be pooled into embeddings.

- For better embeddings, consider swapping the model in `utils.py` to:

```python
MODEL_NAME = "microsoft/wavlm-base-plus"
# or
MODEL_NAME = "microsoft/wavlm-base"
```

These variants often produce higher-quality speech representations but may be larger and slower.

Troubleshooting

- `torchaudio` / `speechbrain` errors:
  - If SpeechBrain or torchaudio raise backend errors (like `list_audio_backends` missing), ensure you installed compatible PyTorch/torchaudio wheels for your Python version. Using Python 3.11 and installing PyTorch from the official wheel or from the PyTorch CPU index helps.

- `librosa.resample` TypeError:
  - Some librosa versions changed the resample signature. The code in `utils.py` calls `librosa.resample(y=..., orig_sr=..., target_sr=...)` to be robust. If you still see an error, upgrade/downgrade librosa to match the wheel in `requirements.txt`.

- If `extract_and_index.py` completes but `search_app.py` can't find index files:
  - Ensure `index_files/faiss.index` and `data/metadata.json` exist. Re-run the indexer if necessary.

- Large model download:
  - The first time a Hugging Face model is used it will download weights (internet required). If you're behind a proxy, set `HUGGINGFACE_HUB_TOKEN` or configure networking accordingly.

Extending the project

- Emotion detection: integrate SpeechBrain's pretrained emotion classifier to compute and store emotion labels in `metadata.json`.
- Segment-level indexing: for long recordings, compute embeddings on overlapping windows (e.g., 1.5s windows with 0.75s stride) and store start/end times in metadata.
- Approximate search: replace `faiss.IndexFlatIP` with an IVF or HNSW index for large datasets.

Contact / Next steps

If you'd like, I can:
- Add emotion detection to the indexer and show emotion labels in the Streamlit UI.
- Add waveform/spectrogram thumbnails in `search_app.py`.
- Swap to a WavLM default and re-run a small embedding test in this environment.

Happy testing — run `python generate_sample_wavs.py`, then `python extract_and_index.py`, then `streamlit run search_app.py`. If anything fails, paste the traceback here and I will help fix it.