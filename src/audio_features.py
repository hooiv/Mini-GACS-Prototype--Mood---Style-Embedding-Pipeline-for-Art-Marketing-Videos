"""
audio_features.py
-----------------
Audio spectral feature extraction and cross-modal affective scoring.

Implements §2d "Multi-Modal Extension" (REPORT.md §17).  The pipeline is
purely visual by default; when video files contain an audio track this
module extracts spectral features via scipy (no librosa dependency) and
maps them onto the same six affective axes used by ``AffectiveScorer``
(energy, warmth, complexity, luxury, joy, tension), enabling:

1. **Audio-only affective scores** — characterise the *sound* of a creative.
2. **Cross-modal discord score** — 1 − cosine(audio_aff, visual_aff);
   quantifies audio/visual vibe mismatch, a known predictor of reduced ad
   recall (Brackett & McLeod 2000; North et al. 2004).

Audio extraction from video files requires ``ffmpeg`` on PATH
(``apt install ffmpeg`` or ``brew install ffmpeg``).  All functions that
operate on pre-extracted PCM arrays work without ffmpeg and are fully tested.

Features computed (scipy + numpy only):
- **RMS energy** — root-mean-square signal amplitude.
- **Zero-crossing rate (ZCR)** — sign-change density; proxy for noisiness.
- **Spectral centroid** — power-weighted mean frequency; proxy for brightness.
- **Spectral bandwidth** — power-weighted std around centroid.
- **Spectral rolloff (85th pct)** — frequency below which 85% of energy lies.
- **MFCC-proxy cepstral means** — 13 DCT coefficients of a log-mel filterbank
  (numpy/scipy; no librosa required).

Affective axis mapping heuristic (weights documented in ``_AXIS_WEIGHTS``):
- energy    ← RMS energy (+0.6) + spectral centroid (+0.4)
- warmth    ← spectral centroid (−0.5) + ZCR (−0.5)
- complexity← spectral bandwidth (+0.6) + MFCC variance (+0.4)
- luxury    ← ZCR (−0.6) + spectral bandwidth (−0.4)
- joy       ← RMS energy (+0.5) + spectral centroid (+0.5)
- tension   ← ZCR (+0.6) + spectral rolloff (+0.4)
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import dct, rfft, rfftfreq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

_DEFAULT_SR: int = 22_050          # default extraction sample rate (Hz)
_ROLLOFF_PERCENTILE: float = 0.85  # fraction of energy for spectral rolloff
_MFCC_N_COEFF: int = 13            # number of cepstral coefficients to keep
_N_MEL_BINS: int = 40              # mel filterbank resolution

# Normalisation upper bounds (typical speech/music operational ranges)
_NORM_RMS: float = 0.30
_NORM_ZCR: float = 0.30
_NORM_CENTROID_HZ: float = 8_000.0
_NORM_BANDWIDTH_HZ: float = 4_000.0
_NORM_ROLLOFF_HZ: float = 16_000.0
_NORM_MFCC_VAR: float = 10.0       # empirical max variance across coefficients

# Axis names (must match AffectiveScorer axes)
_DISCORD_AXES: Tuple[str, ...] = (
    "energy", "warmth", "complexity", "luxury", "joy", "tension",
)

# Affective axis weight matrix.
# Column order: [rms_energy, zcr, spectral_centroid, spectral_bandwidth,
#                spectral_rolloff, mfcc_var]
# Each row sums to at most 1.0 in absolute value so the output stays in [-1, 1].
_AXIS_WEIGHTS: Dict[str, Tuple[float, ...]] = {
    "energy":      ( 0.60,  0.00,  0.40,  0.00,  0.00,  0.00),
    "warmth":      ( 0.00, -0.50, -0.50,  0.00,  0.00,  0.00),
    "complexity":  ( 0.00,  0.00,  0.00,  0.60,  0.00,  0.40),
    "luxury":      ( 0.00, -0.60,  0.00, -0.40,  0.00,  0.00),
    "joy":         ( 0.50,  0.00,  0.50,  0.00,  0.00,  0.00),
    "tension":     ( 0.00,  0.60,  0.00,  0.00,  0.40,  0.00),
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg is unavailable or audio extraction fails."""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AudioSpectralFeatures:
    """
    Spectral audio features for one analysis window, normalised to ``[0, 1]``.

    Attributes:
        rms_energy:          Root-mean-square amplitude (0 = silent).
        zcr:                 Zero-crossing rate (0 = DC, 1 = maximal).
        spectral_centroid:   Power-weighted mean frequency, normalised.
        spectral_bandwidth:  Power-weighted std around centroid, normalised.
        spectral_rolloff:    85th-percentile rolloff frequency, normalised.
        mfcc_means:          Shape ``(_MFCC_N_COEFF,)`` cepstral coefficients.
        mfcc_var:            Mean MFCC coefficient variance, normalised to [0, 1].
    """

    rms_energy: float
    zcr: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    mfcc_means: np.ndarray
    mfcc_var: float

    def to_dict(self) -> Dict:
        d = {k: v for k, v in asdict(self).items() if k != "mfcc_means"}
        d["mfcc_means"] = self.mfcc_means.tolist()
        return d


# ---------------------------------------------------------------------------
# Low-level signal processing (pure scipy/numpy — no ffmpeg)
# ---------------------------------------------------------------------------

def compute_rms(audio: np.ndarray) -> float:
    """RMS energy normalised to ``[0, 1]`` by ``_NORM_RMS``."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return min(rms / _NORM_RMS, 1.0)


def compute_zcr(audio: np.ndarray) -> float:
    """Zero-crossing rate normalised to ``[0, 1]`` by ``_NORM_ZCR``."""
    signs = np.sign(audio.astype(np.float64))
    signs[signs == 0] = 1.0
    zcr = float(np.mean(np.abs(np.diff(signs)) / 2.0))
    return min(zcr / _NORM_ZCR, 1.0)


def compute_spectral_features(
    audio: np.ndarray,
    sr: int = _DEFAULT_SR,
) -> Tuple[float, float, float]:
    """
    Compute (spectral_centroid, spectral_bandwidth, spectral_rolloff),
    all normalised to ``[0, 1]``.

    Uses ``scipy.fft.rfft`` for the power spectrum.

    Returns:
        Tuple ``(centroid, bandwidth, rolloff)``.
    """
    n = len(audio)
    if n == 0:
        return 0.0, 0.0, 0.0

    power = np.abs(rfft(audio.astype(np.float64))) ** 2
    freqs = rfftfreq(n, d=1.0 / sr)
    total = float(power.sum())

    if total < 1e-10:
        return 0.0, 0.0, 0.0

    centroid_hz = float(np.dot(freqs, power) / total)
    bandwidth_hz = float(
        np.sqrt(np.dot((freqs - centroid_hz) ** 2, power) / total)
    )

    # Rolloff: cumulative sum → find index where threshold is exceeded
    cumsum = np.cumsum(power)
    threshold = _ROLLOFF_PERCENTILE * cumsum[-1]
    rolloff_idx = int(np.searchsorted(cumsum, threshold))
    rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    return (
        min(centroid_hz / _NORM_CENTROID_HZ, 1.0),
        min(bandwidth_hz / _NORM_BANDWIDTH_HZ, 1.0),
        min(rolloff_hz / _NORM_ROLLOFF_HZ, 1.0),
    )


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def compute_mfcc_proxy(
    audio: np.ndarray,
    sr: int = _DEFAULT_SR,
    n_coeff: int = _MFCC_N_COEFF,
    n_mel: int = _N_MEL_BINS,
) -> np.ndarray:
    """
    MFCC-proxy cepstral coefficients computed with scipy DCT on a log-mel
    filterbank — no librosa dependency.

    Returns:
        Float32 array of shape ``(n_coeff,)``.
    """
    n = len(audio)
    if n == 0:
        return np.zeros(n_coeff, dtype=np.float32)

    power = np.abs(rfft(audio.astype(np.float64))) ** 2
    freqs = rfftfreq(n, d=1.0 / sr)

    # Build mel filterbank
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sr / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_bins = np.clip(
        np.floor((n + 1) * hz_points / sr).astype(int),
        0, len(power) - 1,
    )

    filterbank = np.zeros((n_mel, len(power)))
    for m in range(1, n_mel + 1):
        f_lo, f_cen, f_hi = fft_bins[m - 1], fft_bins[m], fft_bins[m + 1]
        span_up = max(f_cen - f_lo, 1)
        span_dn = max(f_hi - f_cen, 1)
        if f_cen > f_lo:
            filterbank[m - 1, f_lo:f_cen] = (
                np.arange(f_lo, f_cen) - f_lo
            ) / span_up
        if f_hi > f_cen:
            filterbank[m - 1, f_cen:f_hi] = (
                f_hi - np.arange(f_cen, f_hi)
            ) / span_dn

    mel_energies = filterbank @ power
    log_mel = np.log(mel_energies + 1e-10)

    # DCT-II → cepstral domain
    cepstral = dct(log_mel, type=2, norm="ortho")
    return cepstral[:n_coeff].astype(np.float32)


def extract_audio_features(
    audio: np.ndarray,
    sr: int = _DEFAULT_SR,
) -> AudioSpectralFeatures:
    """
    Compute all spectral features for a 1-D PCM audio array.

    Args:
        audio:  1-D array (float or int16).  Int16 values are normalised
                to ``[-1, 1]`` automatically.
        sr:     Sample rate in Hz.

    Returns:
        :class:`AudioSpectralFeatures` with all scalar values in ``[0, 1]``.

    Raises:
        ValueError: if *audio* is not 1-D or is empty.
    """
    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError(
            f"Expected 1-D non-empty audio array; got shape {audio.shape}."
        )

    audio_f = audio.astype(np.float64)
    if np.abs(audio_f).max() > 1.0:
        audio_f = audio_f / 32768.0  # int16 range → [-1, 1]

    rms = compute_rms(audio_f)
    zcr = compute_zcr(audio_f)
    centroid, bandwidth, rolloff = compute_spectral_features(audio_f, sr)
    mfcc = compute_mfcc_proxy(audio_f, sr)
    mfcc_var = float(min(float(np.var(mfcc)) / _NORM_MFCC_VAR, 1.0))

    return AudioSpectralFeatures(
        rms_energy=rms,
        zcr=zcr,
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        spectral_rolloff=rolloff,
        mfcc_means=mfcc,
        mfcc_var=mfcc_var,
    )


# ---------------------------------------------------------------------------
# Affective axis mapping
# ---------------------------------------------------------------------------

def map_to_affective_axes(
    features: AudioSpectralFeatures,
) -> Dict[str, float]:
    """
    Map spectral features to affective axis scores in ``[-1, 1]``.

    Each ``[0, 1]`` feature is first re-scaled to ``[-1, 1]`` via
    ``f_norm = 2*f - 1``, then combined as a weighted sum using the
    coefficient table ``_AXIS_WEIGHTS``.  The result is clipped to
    ``[-1, 1]`` to guard against floating-point edge cases.

    Returns:
        Dict mapping each axis name in :data:`_DISCORD_AXES` to a float.
    """
    feat_vec = np.array(
        [
            2.0 * features.rms_energy - 1.0,
            2.0 * features.zcr - 1.0,
            2.0 * features.spectral_centroid - 1.0,
            2.0 * features.spectral_bandwidth - 1.0,
            2.0 * features.spectral_rolloff - 1.0,
            2.0 * features.mfcc_var - 1.0,
        ],
        dtype=np.float64,
    )

    return {
        axis: float(np.clip(np.dot(np.array(weights), feat_vec), -1.0, 1.0))
        for axis, weights in _AXIS_WEIGHTS.items()
    }


# ---------------------------------------------------------------------------
# Cross-modal discord score
# ---------------------------------------------------------------------------

def cross_modal_discord_score(
    audio_affective: Dict[str, float],
    visual_affective: Dict[str, float],
    axes: Tuple[str, ...] = _DISCORD_AXES,
) -> float:
    """
    Compute the audio-visual affective discord score.

    Defined as::

        discord = 1 − cosine_similarity(audio_vec, visual_vec)

    where both vectors are constructed from the *axes* keys.

    - **0** → perfect alignment (audio and visual vibes match).
    - **1** → orthogonal (no correlation between audio and visual mood).
    - **2** → perfect opposition (contrarian creative).

    Typical well-aligned marketing creatives score < 0.4.

    Args:
        audio_affective:   Dict from :func:`map_to_affective_axes`.
        visual_affective:  Dict from ``AffectiveScorer.score_video_level``
                           (values in ``[-1, 1]``).
        axes:              Axis names to include (must exist in both dicts).

    Returns:
        Float in ``[0, 2]``.

    Raises:
        KeyError: if any axis is missing from either dict.
    """
    a_vec = np.array([audio_affective[ax] for ax in axes], dtype=np.float64)
    v_vec = np.array([visual_affective[ax] for ax in axes], dtype=np.float64)

    norm_a = float(np.linalg.norm(a_vec))
    norm_v = float(np.linalg.norm(v_vec))

    if norm_a < 1e-10 or norm_v < 1e-10:
        logger.warning(
            "cross_modal_discord_score: near-zero affective vector "
            "(norm_a=%.2e, norm_v=%.2e); returning 0.5 (neutral).",
            norm_a,
            norm_v,
        )
        return 0.5

    cos_sim = float(np.clip(np.dot(a_vec, v_vec) / (norm_a * norm_v), -1.0, 1.0))
    return 1.0 - cos_sim


# ---------------------------------------------------------------------------
# ffmpeg extraction (requires ffmpeg on PATH)
# ---------------------------------------------------------------------------

def extract_audio_pcm(
    video_path: str,
    sr: int = _DEFAULT_SR,
    mono: bool = True,
) -> np.ndarray:
    """
    Extract raw PCM audio from a video file using ``ffmpeg``.

    Args:
        video_path:  Path to the video file (any container supported by ffmpeg).
        sr:          Target sample rate in Hz.
        mono:        If *True*, downmix to mono.

    Returns:
        1-D float32 NumPy array, values in ``[-1, 1]``.  Returns one second
        of silence if no audio track is found.

    Raises:
        AudioExtractionError: if ffmpeg is not installed or extraction fails.
        FileNotFoundError:    if *video_path* does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path!r}")

    # Verify ffmpeg availability
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AudioExtractionError(
            "ffmpeg is not installed or not on PATH.  "
            "Install with:  apt install ffmpeg  or  brew install ffmpeg"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                           # no video stream
            "-acodec", "pcm_s16le",          # signed 16-bit PCM
            "-ar", str(sr),                  # resample to target rate
            "-ac", "1" if mono else "2",
            "-f", "s16le",                   # raw headerless PCM
            tmp_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg returned exit code {result.returncode} "
                f"for {video_path!r}:\n"
                + result.stderr.decode(errors="replace")
            )

        with open(tmp_path, "rb") as fh:
            raw = np.frombuffer(fh.read(), dtype=np.int16).astype(np.float32)

        if raw.size == 0:
            logger.warning("No audio data extracted from %s; using silence.", video_path)
            return np.zeros(sr, dtype=np.float32)

        raw /= 32768.0  # int16 → [-1, 1]
        logger.info(
            "Extracted %.1f s of audio from %s at %d Hz.",
            len(raw) / sr,
            os.path.basename(video_path),
            sr,
        )
        return raw

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def generate_synthetic_audio_features(
    n_windows: int = 1,
    seed: Optional[int] = 42,
) -> List[AudioSpectralFeatures]:
    """
    Generate synthetic :class:`AudioSpectralFeatures` for testing / demo.

    Args:
        n_windows:  Number of feature windows to generate.
        seed:       Random seed for reproducibility.

    Returns:
        List of :class:`AudioSpectralFeatures` with values drawn from
        ``Uniform[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    results: List[AudioSpectralFeatures] = []
    for _ in range(n_windows):
        results.append(
            AudioSpectralFeatures(
                rms_energy=float(rng.uniform(0, 1)),
                zcr=float(rng.uniform(0, 1)),
                spectral_centroid=float(rng.uniform(0, 1)),
                spectral_bandwidth=float(rng.uniform(0, 1)),
                spectral_rolloff=float(rng.uniform(0, 1)),
                mfcc_means=rng.uniform(-1, 1, _MFCC_N_COEFF).astype(np.float32),
                mfcc_var=float(rng.uniform(0, 1)),
            )
        )
    return results


def score_video_audio(
    video_path: str,
    sr: int = _DEFAULT_SR,
) -> Optional[Dict[str, float]]:
    """
    High-level helper: extract audio from *video_path* and return its
    affective axis scores.

    Returns *None* (and logs a warning) if ffmpeg is unavailable or the
    video has no audio track.
    """
    try:
        pcm = extract_audio_pcm(video_path, sr=sr)
        feats = extract_audio_features(pcm, sr=sr)
        return map_to_affective_axes(feats)
    except (AudioExtractionError, FileNotFoundError, ValueError) as exc:
        logger.warning("Audio scoring skipped for %s: %s", video_path, exc)
        return None
