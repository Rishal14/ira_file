import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
import matplotlib.pyplot as plt
from utils.feature_extractor import FeatureExtractor
from utils.similarity_computer import SimilarityComputer

# Set page config
st.set_page_config(
    page_title="🎙️ Query-by-Example Speech Retrieval System",
    layout="wide"
)

# Initialize session state
if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = FeatureExtractor(feature_type='mfcc')
if 'similarity_computer' not in st.session_state:
    st.session_state.similarity_computer = SimilarityComputer()

def load_precomputed_features():
    """Load precomputed features and emotions from files"""
    features = np.load('features/dataset_features.npy', allow_pickle=True).item()
    emotions = np.load('features/dataset_emotions.npy', allow_pickle=True).item()
    return features, emotions

def plot_waveform(audio, sr):
    """Plot waveform of audio signal"""
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    return fig

def main():
    st.title("🎙️ Query-by-Example Speech Retrieval System")
    
    # Load precomputed features
    try:
        dataset_features, dataset_emotions = load_precomputed_features()
        st.sidebar.success("✅ Dataset features loaded successfully!")
    except:
        st.sidebar.error("❌ No precomputed features found. Please process the dataset first.")
        return

    # File uploader
    st.subheader("Upload Query Audio")
    query_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if query_file:
        # Save uploaded file temporarily
        with open('temp_query.wav', 'wb') as f:
            f.write(query_file.getvalue())
        
        # Extract features from query
        query_features = st.session_state.feature_extractor.extract_features('temp_query.wav')
        query_emotion = st.session_state.feature_extractor.detect_emotion('temp_query.wav')
        
        # Display query audio info
        col1, col2 = st.columns(2)
        with col1:
            st.audio('temp_query.wav')
        with col2:
            st.info(f"Detected Emotion: {query_emotion}")
            
        # Compute similarities
        similarities = st.session_state.similarity_computer.compute_similarities(
            query_features, dataset_features
        )
        
        # Display top 3 matches
        st.subheader("Top 3 Similar Audio Clips")
        
        for i, (filename, score) in enumerate(similarities[:3], 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.audio(os.path.join('data', filename))
                audio, sr = librosa.load(os.path.join('data', filename))
                st.pyplot(plot_waveform(audio, sr))
                
            with col2:
                st.metric(
                    label=f"Match #{i}",
                    value=f"Score: {score:.3f}",
                    delta=f"Emotion: {dataset_emotions[filename]}"
                )
                
            with col3:
                if st.button(f"Download Match #{i}", key=f"download_{i}"):
                    with open(os.path.join('data', filename), 'rb') as f:
                        st.download_button(
                            label=f"Save Audio #{i}",
                            data=f,
                            file_name=filename,
                            mime='audio/wav'
                        )
        
        # Cleanup
        os.remove('temp_query.wav')

    # Sidebar filters
    st.sidebar.subheader("Filters")
    selected_emotion = st.sidebar.selectbox(
        "Filter by emotion",
        ['All'] + list(set(dataset_emotions.values()))
    )

if __name__ == "__main__":
    main()