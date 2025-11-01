"""Process dataset audio files and save features and emotion labels.
Run from project root after activating the virtual environment:
    python process_dataset.py
This will read WAV files from data/ and write features to features/dataset_features.npy
and emotions to features/dataset_emotions.npy
"""
import os
from utils.feature_extractor import FeatureExtractor

DATA_DIR = "data"
FEATURES_DIR = "features"
SAVE_PREFIX = os.path.join(FEATURES_DIR, "dataset")

if __name__ == "__main__":
    os.makedirs(FEATURES_DIR, exist_ok=True)
    extractor = FeatureExtractor(feature_type='mfcc')
    print(f"Processing dataset in: {DATA_DIR}")
    features, emotions = extractor.process_dataset(DATA_DIR, SAVE_PREFIX)
    print(f"Saved features to: {SAVE_PREFIX}_features.npy")
    print(f"Saved emotions to: {SAVE_PREFIX}_emotions.npy")
    print(f"Processed {len(features)} files.")
