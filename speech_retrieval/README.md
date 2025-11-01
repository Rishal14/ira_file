# 🎙️ Intelligent Query-by-Example Speech Retrieval System

This system allows users to upload a speech audio clip and find similar audio samples from a dataset using feature extraction and cosine similarity.

## Features

- Audio similarity search using MFCC or Wav2Vec2.0 features
- Emotion detection using SpeechBrain
- Interactive Streamlit web interface
- Waveform visualization
- Audio playback and download options
- Emotion-based filtering

## Directory Structure

```
speech_retrieval/
├── data/               # Dataset directory for .wav files
├── features/          # Precomputed features storage
├── utils/             # Utility modules
│   ├── feature_extractor.py
│   └── similarity_computer.py
├── app.py            # Main Streamlit application
└── requirements.txt  # Project dependencies
```

## Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
- Place .wav audio files in the `data/` directory
- Supported format: 16kHz mono WAV files

4. Process the dataset:
```python
from utils.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
extractor.process_dataset('data', 'features/dataset')
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Open the web interface (default: http://localhost:8501)
2. Upload a query audio file (.wav format)
3. View the top 3 similar audio matches
4. Listen to or download matching audio clips
5. Use the emotion filter in the sidebar to filter results

## Technical Details

- Audio Features: MFCC (default) or Wav2Vec2.0 embeddings
- Similarity Metric: Cosine Similarity
- Emotion Detection: SpeechBrain pretrained model
- Supported Audio Format: WAV (16kHz, mono)

## Dependencies

- librosa
- numpy
- streamlit
- pandas
- scikit-learn
- torch
- torchaudio
- transformers
- speechbrain
- soundfile
- matplotlib
- seaborn

## Note

Make sure to process your dataset before running the application. The system needs precomputed features to function properly.