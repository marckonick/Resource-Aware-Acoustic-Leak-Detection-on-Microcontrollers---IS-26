# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 01:23:53 2026

@author: nikola.markovic
"""

import numpy as np 
import pandas as pd
from scipy.io import wavfile
import os 
import librosa 
from scipy.signal import resample_poly
import yaml



def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def _summarize(x, how, axis = -1):
    """
    Reduce a time/frequency axis to fixed-size statistics.

    Used to convert variable-length or high-dimensional per-frame features
    into a fixed-length vector per clip.
    """
    
    if how is None or how == 'none':
        return x
    if how == 'mean':
        return x.mean(axis=axis)
    if how == 'std':
        return x.std(axis=axis)
    if how == 'mean_std':
        return np.concatenate([x.mean(axis=axis), x.std(axis=axis)], axis=-1)
    if how == 'stats':
        # mean, std, min, max, median — a richer fixed summary
        return np.concatenate([
            x.mean(axis=axis),
            x.std(axis=axis),
            x.min(axis=axis),
            x.max(axis=axis),
            np.median(x, axis=axis),
        ], axis=-1)
    raise ValueError(f"Unknown summarize mode: {how!r}")
    
    
def get_frames(x_ex, frame_size, frame_stride, samplerate):

    frame_length, frame_step = frame_size * samplerate, frame_stride * samplerate  # Convert from seconds to samples
    signal_length = len(x_ex)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    # Pad signal to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(x_ex, z)

    # Slice the signal into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply Hamming window
    frames *= np.hanning(frame_length)

    return frames


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """Promote a 1D clip to a (1, L) batch. Leave (N, L) untouched."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[np.newaxis, :]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected input of shape (L,) or (N, L), got {x.shape}")



def extract_spectral_summary(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    summarize = 'mean_std',
) -> np.ndarray:
    """
    Interpretable hand-crafted spectral / temporal features.

    Computes per-frame:
        - spectral centroid    (where the spectrum's "center of mass" is)
        - spectral bandwidth   (spread around the centroid)
        - spectral rolloff     (frequency below which 85% of energy lies)
        - spectral flatness    (tonal vs. noisy)
        - zero-crossing rate   (rough proxy for high-frequency content)
        - RMS energy           (loudness)

    Then summarizes each one across time. Defaults give a 12-dim vector
    per clip (6 features x mean+std), which is small enough for a workshop
    scatter plot but already separates coughs from breaths fairly well.

    Returns
    -------
    np.ndarray of shape (N, D) if summarize is set, else (N, 6, T).
    """
    x = _ensure_2d(audio)
    N = x.shape[0]

    feats = []
    for i in range(N):
        y = x[i].astype(np.float32)

        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
        )
        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=n_fft, hop_length=hop_length
        )
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

        # Each is shape (1, T) — stack to (6, T)
        stacked = np.concatenate([centroid, bandwidth, rolloff,
                                  flatness, zcr, rms], axis=0)
        feats.append(stacked)

    feats = np.stack(feats, axis=0)  # (N, 6, T)

    if summarize is not None:
        return _summarize(feats, summarize, axis=-1).reshape(N, -1)
    return feats


#* S = Recording session (1, 2, or 3)
#* L = Annotation (_iO_ = no leak present, _niO_ = leak present)
#* N = Number of knob rotations (0.0 - 9.0)
#* M = Mic configuration (1 - 4, m = max volume, l = low volume)



def get_raw_state_data(fault_type, noise, take, microphone, base_folder = "", knobs=["0.0", "1.0", "2.0", "3.0",  "5.5",  "6.0",  "6.5",  "7.0", "7.5",  "8.0",  "9.0"]):


       signals_normal = []
       signals_fault = []

       T_samples = int(30*48000)

          
       for knob in knobs:
               filename_2_load_normal = base_folder  + fault_type + "/" + noise + "/" + take + "/" + take + "_iO_" + knob + "n_" + microphone + "_.wav" 
               filename_2_load_fault = base_folder   + fault_type + "/" + noise + "/" + take + "/" + take + "_niO_" + knob + "n_" + microphone + "_.wav" 

               audio_data = []
               if os.path.isfile(filename_2_load_normal):
                   sample_rate, audio_data = wavfile.read(filename_2_load_normal)
                   
                   if len(audio_data) < T_samples:
                       audio_data = np.concatenate((audio_data, np.zeros(T_samples - len(audio_data), dtype = np.int32)))
                   else:
                       audio_data = audio_data[0:T_samples]
                   
                   signals_normal.append(audio_data)
                
               if os.path.isfile(filename_2_load_fault):
                       sample_rate, audio_data = wavfile.read(filename_2_load_fault)
                       
                       if len(audio_data) < T_samples:
                           audio_data = np.concatenate((audio_data, np.zeros(T_samples - len(audio_data), dtype = np.int32)))
                       else:
                           audio_data = audio_data[0:T_samples]
                                                   
                       signals_fault.append(audio_data)             
            
               #print(sample_rate)
      
        
       signals_normal = np.array(signals_normal)  
       signals_fault = np.array(signals_fault)
 
       return signals_normal, signals_fault
        
   
    
_EXTRACTORS = {
   #'fft': extract_fft,
   #'mfcc': extract_mfcc,
   # 'mfcc_manual': extract_mfcc_manual,
    'spectral_summary': extract_spectral_summary,
    'spectral': extract_spectral_summary,  # alias
}


def extract_features(
    audio: np.ndarray,
    feature: str,
    sr: int = 16000,
    *args,
    **kwargs,
) -> np.ndarray:

    key = feature.lower().strip()
    if key not in _EXTRACTORS:
        raise ValueError(
            f"Unknown feature {feature!r}. "
            f"Available: {sorted(_EXTRACTORS)}"
        )
    return _EXTRACTORS[key](audio, sr=sr, *args, **kwargs)     
        
feature_save_name = "" # tubeleak_lab_fault_take_1, tubeleak_lab_norm_take_1
        
# %%

      
def perform_feature_extraction(sel_state, sel_feature, **kwargs):
            
        noises = ["lab"]
        takes = ["1"]
        microphones=["1l", "1m"]
        
        
        downsaple_rate = kwargs["downsample"]
        Fs = int(kwargs["Fs"]/downsaple_rate)
        base_folder = ""
        
        
        for noise in noises:
            for take in takes:
                for microphone in microphones:
        
                    signals_normal, signals_fault = get_raw_state_data(sel_state, noise, take, microphone, base_folder)
                    
                    Xn = []
                    Xf = []
                    
                    if sel_feature == 'spectral_summary':
                        for j in range(signals_normal.shape[0]):
                            
                            x_cur = resample_poly(signals_normal[j:j+1], up=1, down=downsaple_rate, axis=1) # resample signal
                            x_cur = extract_features(x_cur, sel_feature, sr=Fs, summarize= None, n_fft = 1024) #  summarize=None                         
                            Xn.append(x_cur)
        
                            x_cur = resample_poly(signals_fault[j:j+1], up=1, down=downsaple_rate, axis=1) # resample signal
                            x_cur = extract_features(x_cur, sel_feature, sr=Fs, summarize= None, n_fft = 1024) #  summarize=None                         
                            Xf.append(x_cur)
                            
                        Xn = np.concatenate(Xn, 0)
                        Xf = np.concatenate(Xf, 0)
                        
                        feat_savename_n = "saved_features/" + sel_state + "_normal_" + sel_feature + "_" + noise + "_t" + take + "_m" + microphone + ".npy"    
                        np.save(feat_savename_n, Xn)
                        feat_savename_f = "saved_features/" + sel_state + "_fault_" + sel_feature + "_" + noise + "_t" + take + "_m" + microphone + ".npy"    
                        np.save(feat_savename_f, Xf)

                        
                        print(f"Extracted and saved {feat_savename_n} and \n {feat_savename_f}")
                              
                            
                            
                            

                                             
config = load_config("config_feat_extraction.yaml")  
df_config = config['data_and_features']
kwargs_arguments = {'Fs':df_config['Fs'], 'downsample':df_config["downsample"]}


perform_feature_extraction("tubeleak", "spectral_summary", **kwargs_arguments)






