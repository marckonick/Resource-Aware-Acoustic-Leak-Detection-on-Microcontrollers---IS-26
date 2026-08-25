import numpy as np
from scipy.io import wavfile
import os
import librosa
from scipy.signal import resample_poly




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


# ---------------------------------------------------------------------------
# Feature tier T0/T1: interpretable spectral + temporal summaries
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature tier T1/T2: band-power descriptors
# ---------------------------------------------------------------------------

def _band_edges(n_bands, fmin, fmax, scale):
    """Frequency band edges (n_bands + 1 values) on a log or linear grid."""
    if scale == 'log':
        return np.logspace(np.log10(max(fmin, 1.0)), np.log10(fmax), n_bands + 1)
    if scale in ('linear', 'lin'):
        return np.linspace(fmin, fmax, n_bands + 1)
    raise ValueError(f"Unknown band_scale: {scale!r} (use 'log' or 'linear')")


def extract_bandpower(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    n_bands: int = 8,
    fmin: float = 50.0,
    fmax: float = None,
    band_scale: str = 'log',
    log_power: bool = True,
    ref_band: int = None,
    eps: float = 1e-10,
    summarize = 'mean_std',
) -> np.ndarray:
    """
    Band-power descriptors: STFT power aggregated into a handful of
    frequency bands per frame.

    This is the cheap, embedded-friendly spectral tier. Instead of the full
    magnitude spectrum, each frame is reduced to the energy in `n_bands`
    (log-spaced by default, which gives finer resolution at low frequency and
    coarse bands over the high-frequency hiss where compressed-air leaks live).
    On a microcontroller these same band energies are what a bank of Goertzel
    filters would produce, so the representation maps directly onto a
    deployable feature extractor.

    Parameters
    ----------
    n_bands : number of contiguous frequency bands.
    fmin, fmax : band range in Hz. fmax defaults to the Nyquist rate (sr / 2).
    band_scale : 'log' (default) or 'linear' spacing of the band edges.
    log_power : if True, return log-energy (dB-like); else linear power.
    ref_band : if set, subtract this band's log-energy from every band,
        yielding scale-invariant band *ratios* (only used when log_power=True).
        Useful because leak vs. no-leak often differs more in spectral shape
        than in absolute level, and ratios cancel per-clip gain differences.

    Returns
    -------
    np.ndarray of shape (N, n_bands, T) if summarize is None,
    else (N, n_bands * k) with k set by the summarize mode.
    """
    x = _ensure_2d(audio)
    N = x.shape[0]
    if fmax is None:
        fmax = sr / 2.0

    edges = _band_edges(n_bands, fmin, fmax, band_scale)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)      # (n_fft // 2 + 1,)

    # Precompute the STFT-bin mask for each band once (bands are clip-independent).
    band_masks = []
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        band_masks.append(mask)

    feats = []
    for i in range(N):
        y = x[i].astype(np.float32)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2  # power (F, T)
        T = S.shape[1]

        bp = np.empty((n_bands, T), dtype=np.float32)
        for b, mask in enumerate(band_masks):
            bp[b] = S[mask].sum(axis=0) if mask.any() else 0.0

        if log_power:
            bp = np.log(bp + eps)
            if ref_band is not None:
                bp = bp - bp[ref_band:ref_band + 1]     # broadcast subtract reference

        feats.append(bp)

    feats = np.stack(feats, axis=0)                          # (N, n_bands, T)

    if summarize is not None:
        return _summarize(feats, summarize, axis=-1).reshape(N, -1)
    return feats


# ---------------------------------------------------------------------------
# Feature tier T2: MFCCs
# ---------------------------------------------------------------------------

def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    n_mels: int = 40,
    n_mfcc: int = 13,
    fmin: float = 0.0,
    fmax: float = None,
    deltas: bool = False,
    summarize = 'mean_std',
) -> np.ndarray:
    """
    Mel-frequency cepstral coefficients per frame.

    A compact, decorrelated summary of the log-mel spectral envelope. With
    `deltas=True` the first- and second-order temporal derivatives are stacked
    on, giving 3 * n_mfcc channels and capturing local dynamics — cheap extra
    context that sometimes helps distinguish a steady leak hiss from transient
    background events.

    Returns
    -------
    np.ndarray of shape (N, D, T) if summarize is None (D = n_mfcc, or
    3 * n_mfcc when deltas=True), else (N, D * k).
    """
    x = _ensure_2d(audio)
    N = x.shape[0]
    if fmax is None:
        fmax = sr / 2.0

    feats = []
    for i in range(N):
        y = x[i].astype(np.float32)
        m = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax,
        )                                                    # (n_mfcc, T)
        if deltas:
            d1 = librosa.feature.delta(m)
            d2 = librosa.feature.delta(m, order=2)
            m = np.concatenate([m, d1, d2], axis=0)          # (3 * n_mfcc, T)
        feats.append(m)

    feats = np.stack(feats, axis=0)                          # (N, D, T)

    if summarize is not None:
        return _summarize(feats, summarize, axis=-1).reshape(N, -1)
    return feats


# ---------------------------------------------------------------------------
# Feature tier T3: compact log-mel spectrograms / patches
# ---------------------------------------------------------------------------

def _cut_patches(S, patch_frames, patch_hop):
    """
    Slide a fixed window over the time axis of a (n_mels, T) spectrogram.

    Returns (num_patches, n_mels, patch_frames). Trailing frames that do not
    fill a whole patch are dropped. patch_hop=None -> non-overlapping patches.
    """
    if patch_hop is None:
        patch_hop = patch_frames
    n_mels, T = S.shape
    if T < patch_frames:
        return np.empty((0, n_mels, patch_frames), dtype=S.dtype)
    starts = range(0, T - patch_frames + 1, patch_hop)
    patches = [S[:, s:s + patch_frames] for s in starts]
    return np.stack(patches, axis=0)


def extract_logmel(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float = None,
    power: float = 2.0,
    top_db: float = 80.0,
    patch_frames: int = None,
    patch_hop: int = None,
    summarize = None,
) -> np.ndarray:
    """
    Log-mel spectrogram — the 2D-CNN input tier.

    Computes a mel power spectrogram and converts it to a log (dB) scale via
    librosa.power_to_db with per-clip reference to the clip maximum, then
    clipped to a `top_db` dynamic range. This is the highest-cost feature tier
    and the natural input for a small 2D-CNN.

    Two output modes:
      * patch_frames is None (default): the full log-mel spectrogram is
        returned as (N, n_mels, T) — patch/window it downstream as needed.
      * patch_frames set: each clip is cut into fixed-size time patches,
        returning (N, num_patches, n_mels, patch_frames). At 16 kHz a hop of
        512 samples is 32 ms/frame, so patch_frames=24 -> ~800 ms of context
        per patch, matching the log-mel patch tier in the paper.

    Returns
    -------
    np.ndarray, shape depending on patch_frames (see above). If patch_frames
    is None and `summarize` is set, the time axis is reduced to (N, n_mels * k).
    """
    x = _ensure_2d(audio)
    N = x.shape[0]
    if fmax is None:
        fmax = sr / 2.0

    logmels = []
    for i in range(N):
        y = x[i].astype(np.float32)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=power,
        )                                                    # (n_mels, T)
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
        logmels.append(log_mel.astype(np.float32))

    if patch_frames is not None:
        # (N, num_patches, n_mels, patch_frames) — patches keep their clip index
        patched = [_cut_patches(lm, patch_frames, patch_hop) for lm in logmels]
        return np.stack(patched, axis=0)

    feats = np.stack(logmels, axis=0)                        # (N, n_mels, T)

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
    'mfcc': extract_mfcc,
   # 'mfcc_manual': extract_mfcc_manual,
    'spectral_summary': extract_spectral_summary,
    'spectral': extract_spectral_summary,   # alias
    'bandpower': extract_bandpower,
    'bandpower_ratio': extract_bandpower,   # alias (set ref_band in feature_params)
    'logmel': extract_logmel,
    'logmel_patch': extract_logmel,         # alias (set patch_frames in feature_params)
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


def _resample_batch(signals, downsample_rate):
    """Anti-alias + downsample a (N, L) int/float batch along the time axis.
    Returns None for an empty batch (e.g. no files matched on disk)."""
    if signals is None or len(signals) == 0:
        return None
    return resample_poly(np.double(signals), up=1, down=downsample_rate, axis=1)


def perform_feature_extraction(sel_states, sel_feature, noises, takes, microphones, **kwargs):
    """
    Feature-agnostic extraction loop.

    For every (noise, take, microphone) combination it loads the normal/fault
    clips, resamples them to the embedded target rate (Fs / downsample), runs
    the selected extractor, and saves one .npy per split.

    Feature-specific parameters (n_fft, hop_length, n_bands, n_mels, ...) are
    passed via the `feature_params` kwarg, which comes straight from the YAML
    block for the selected feature. `summarize` controls whether per-frame
    features (None) or a fixed-length per-clip summary ('mean_std', 'stats', …)
    is saved.
    """

    downsample_rate = kwargs["downsample"]
    Fs = int(kwargs["Fs"] / downsample_rate)
    base_folder = kwargs.get("base_folder", "")
    summarize = kwargs.get("summarize", None)
    feat_params = dict(kwargs.get("feature_params", {}) or {})
    out_dir = kwargs.get("save_folder", "saved_features")

    os.makedirs(out_dir, exist_ok=True)

    for sel_state in sel_states:
      for noise in noises:
        for take in takes:
            for microphone in microphones:

                signals_normal, signals_fault = get_raw_state_data(
                    sel_state, noise, take, microphone, base_folder
                )                                            # each: N_knobs x 1_440_000

                xn = _resample_batch(signals_normal, downsample_rate)
                xf = _resample_batch(signals_fault, downsample_rate)

                tag = f"{noise}_t{take}_m{microphone}"

                if xn is not None:
                    Xn = extract_features(xn, sel_feature, sr=Fs,
                                          summarize=summarize, **feat_params)
                    fn = os.path.join(
                        out_dir, f"{sel_state}_normal_{sel_feature}_{tag}.npy")
                    np.save(fn, Xn)
                    print(f"[normal] {sel_feature:16s} {tag:20s} -> {tuple(Xn.shape)}  ({fn})")

                if xf is not None:
                    Xf = extract_features(xf, sel_feature, sr=Fs,
                                          summarize=summarize, **feat_params)
                    ff = os.path.join(
                        out_dir, f"{sel_state}_fault_{sel_feature}_{tag}.npy")
                    np.save(ff, Xf)
                    print(f"[fault ] {sel_feature:16s} {tag:20s} -> {tuple(Xf.shape)}  ({ff})")












