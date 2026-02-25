import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)

def compute_melodic_score(waveform: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> float:
    """
    Computes a 'melodicity' score for an audio segment.
    Higher score means more likely to be melodic (vocals/violin) and less likely to be silence or pure percussion.
    
    Score is based on:
    1. Spectral Flatness: Lower is better (more tonal).
    2. RMS Energy: Higher is better (less likely to be silence).
    
    Score = (1 - mean_flatness) * log(mean_energy + epsilon)
    """
    if len(waveform) == 0:
        return -np.inf
        
    # Ensure waveform is float32
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

    # 1. Spectral Flatness
    # Percussion/Noise -> High Flatness (~1.0)
    # Tonal/Melodic -> Low Flatness (~0.0)
    flatness = librosa.feature.spectral_flatness(y=waveform, n_fft=n_fft, hop_length=hop_length)
    mean_flatness = np.mean(flatness)
    
    # 2. Energy
    # Silence -> Low Energy
    rms = librosa.feature.rms(y=waveform, frame_length=n_fft, hop_length=hop_length)
    mean_rms = np.mean(rms)
    
    # Combine
    # We want low flatness (1 - flatness is high) and reasonable energy.
    # We use log energy to dampen the effect of very loud percussion spikes.
    epsilon = 1e-6
    score = (1.0 - mean_flatness) * np.log(mean_rms + epsilon)
    
    return float(score)
