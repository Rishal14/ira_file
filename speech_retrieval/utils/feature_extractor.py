import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from speechbrain.pretrained import EncoderClassifier
import os

class FeatureExtractor:
    def __init__(self, feature_type='mfcc', sample_rate=16000):
        """
        Initialize feature extractor with specified type
        Args:
            feature_type (str): Type of features to extract ('mfcc' or 'wav2vec')
            sample_rate (int): Audio sample rate
        """
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        
        if feature_type == 'wav2vec':
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Initialize emotion classifier
        self.emotion_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir="tmpdir"
        )

    def extract_features(self, audio_path):
        """
        Extract audio features from file
        Args:
            audio_path (str): Path to audio file
        Returns:
            np.ndarray: Extracted features
        """
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        if self.feature_type == 'mfcc':
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
            return np.mean(mfccs, axis=1)  # Average over time
        
        else:  # wav2vec
            # Process audio for wav2vec
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use mean pooling over time steps
            features = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
            return features

    def detect_emotion(self, audio_path):
        """
        Detect emotion in audio file using SpeechBrain
        Args:
            audio_path (str): Path to audio file
        Returns:
            str: Detected emotion
        """
        signal = self.emotion_classifier.load_audio(audio_path)
        emotion_prob = self.emotion_classifier.classify_batch(signal)
        emotion_idx = emotion_prob[0].argmax().item()
        emotions = ['angry', 'happy', 'sad', 'neutral']
        return emotions[emotion_idx]

    def process_dataset(self, data_dir, save_path):
        """
        Process all audio files in dataset and save features
        Args:
            data_dir (str): Directory containing audio files
            save_path (str): Path to save features
        """
        features = {}
        emotions = {}
        
        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(data_dir, file)
                features[file] = self.extract_features(file_path)
                emotions[file] = self.detect_emotion(file_path)
        
        # Save features and emotions
        np.save(save_path + '_features.npy', features)
        np.save(save_path + '_emotions.npy', emotions)
        return features, emotions