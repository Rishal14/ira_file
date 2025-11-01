import librosa
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Model choice - change for higher accuracy if needed (WavLM / Whisper encoders)
MODEL_NAME = "facebook/wav2vec2-base-960h"

# Lazy-loaded model & processor
_processor = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(_device).eval()
    return _processor, _model


def load_audio(path, sr=16000):
    """
    Load audio from disk and resample to sr (default 16k).
    Returns numpy float32 mono waveform in range [-1, 1] and sr.
    """
    audio, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        # use keyword args to avoid signature dispatch issues across librosa versions
        audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32), sr


def read_audio_file_bytes(file_bytes, sr=16000):
    """
    Read uploaded file bytes (e.g. from Streamlit uploader) and return waveform, sr.
    """
    import io
    data, samplerate = sf.read(io.BytesIO(file_bytes))
    # convert multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if samplerate != sr:
        # handle multi-channel by converting to mono earlier; use keyword args
        data = librosa.resample(y=data.astype(np.float32), orig_sr=samplerate, target_sr=sr)
    return data.astype(np.float32), sr


def compute_embedding_from_array(waveform, sr=16000, pooling="mean"):
    """
    waveform: 1D numpy float array sampled at sr
    returns: 1D numpy vector (float32) normalized to unit length
    """
    processor, model = get_model()
    # processor expects a list of arrays or array and sampling_rate
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(_device)
    attention_mask = inputs.attention_mask.to(_device) if "attention_mask" in inputs else None
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, dim)
        if pooling == "mean":
            # mask-aware mean pooling:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
                masked = hidden_states * mask
                summed = masked.sum(dim=1)
                lens = mask.sum(dim=1).clamp(min=1e-9)
                emb = (summed / lens).squeeze().cpu().numpy()
            else:
                emb = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        else:
            emb = hidden_states.mean(dim=1).squeeze().cpu().numpy()
    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb
