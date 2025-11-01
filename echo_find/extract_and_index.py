import os
import json
import numpy as np
import faiss
from utils import load_audio, compute_embedding_from_array
from tqdm import tqdm

DATA_DIR = "data/wavs"
INDEX_DIR = "index_files"
META_PATH = "data/metadata.json"
EMBED_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

def build_index():
    files = []
    for root, _, fnames in os.walk(DATA_DIR):
        for f in fnames:
            if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
                files.append(os.path.join(root, f))
    files.sort()
    embeddings = []
    metadata = []

    print(f"Found {len(files)} audio files. Computing embeddings...")
    for path in tqdm(files):
        try:
            wav, sr = load_audio(path, sr=16000)
            emb = compute_embedding_from_array(wav, sr=sr)
            embeddings.append(emb)
            metadata.append({"path": path, "filename": os.path.basename(path)})
        except Exception as e:
            print("Error processing", path, e)

    if len(embeddings) == 0:
        raise ValueError("No embeddings computed. Put .wav files in data/wavs/ and try again.")

    X = np.vstack(embeddings).astype("float32")
    np.save(EMBED_PATH, X)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # FAISS index - inner product on normalized vectors => cosine similarity
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, FAISS_PATH)
    print(f"Index built with {index.ntotal} vectors. Saved to {FAISS_PATH}")

if __name__ == "__main__":
    build_index()
