import numpy as np
from scipy.signal import welch


FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def compute_bandpower(signal: np.ndarray, fs: int, band: tuple[float, float]) -> float:
    """
    Compute absolute bandpower using Welch's PSD estimate.
    """
    low, high = band

    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    band_mask = (freqs >= low) & (freqs <= high)

    if not np.any(band_mask):
        return 0.0

    band_power = np.trapz(psd[band_mask], freqs[band_mask])

    return float(band_power)


def extract_trial_bandpower_features(
    trial_eeg: np.ndarray,
    fs: int = 128,
    frequency_bands: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """
    Extract channel-wise bandpower features from one EEG trial.

    Input shape:
    channels x samples
    """
    if frequency_bands is None:
        frequency_bands = FREQUENCY_BANDS

    features = {}
    n_channels = trial_eeg.shape[0]

    for ch_idx in range(n_channels):
        signal = trial_eeg[ch_idx]

        for band_name, band_range in frequency_bands.items():
            feature_name = f"ch{ch_idx + 1:02d}_{band_name}"
            features[feature_name] = compute_bandpower(
                signal=signal,
                fs=fs,
                band=band_range,
            )

    return features


def extract_subject_bandpower_features(
    eeg: np.ndarray,
    fs: int = 128,
    frequency_bands: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """
    Extract bandpower features for all trials of one subject.

    Input shape:
    trials x channels x samples
    """
    rows = []

    for trial_idx in range(eeg.shape[0]):
        trial_features = extract_trial_bandpower_features(
            trial_eeg=eeg[trial_idx],
            fs=fs,
            frequency_bands=frequency_bands,
        )
        rows.append(trial_features)

    return rows