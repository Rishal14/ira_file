import streamlit as st
import numpy as np
import faiss
import json
import os
from utils import read_audio_file_bytes, compute_embedding_from_array

INDEX_DIR = "index_files"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
EMBED_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
META_PATH = "data/metadata.json"

st.set_page_config(page_title="EchoFind - QbE Speech Retrieval", layout="wide")


def load_index():
    if not os.path.exists(FAISS_PATH):
        st.error("FAISS index not found. Run extract_and_index.py first.")
        st.stop()
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_index()

st.title("🎧 EchoFind — Query-by-Example Speech Retrieval")
st.markdown(
    "Upload a short audio clip (wav/mp3/flac) as a query. The system will find similar audio files from the dataset."
)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload a query audio file (wav/mp3/flac)", type=["wav", "mp3", "flac", "m4a"])
    st.write("Recommendations: keep queries between 0.5s and 6s for best results.")
    k = st.slider("Number of results (k)", 1, 20, 5)

    if uploaded is not None:
        file_bytes = uploaded.read()
        waveform, sr = read_audio_file_bytes(file_bytes, sr=16000)
        st.audio(file_bytes, format="audio/wav")
        query_emb = compute_embedding_from_array(waveform, sr=sr)
        D, I = index.search(np.array([query_emb]).astype("float32"), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            meta = metadata[idx]
            results.append((meta, float(score)))
        st.session_state['results'] = results

with col2:
    st.header("Top matches")
    results = st.session_state.get('results', None)
    if results is None:
        st.info("Upload a query audio on the left to see matches.")
    else:
        for meta, score in results:
            path = meta["path"]
            filename = meta.get("filename", os.path.basename(path))
            st.subheader(f"{filename} — score: {score:.4f}")
            try:
                # read bytes to play
                with open(path, "rb") as f:
                    data = f.read()
                st.audio(data, format="audio/wav")
            except Exception as e:
                st.write("Error playing file:", e)
            st.write("Path:", path)
            st.markdown("---")
