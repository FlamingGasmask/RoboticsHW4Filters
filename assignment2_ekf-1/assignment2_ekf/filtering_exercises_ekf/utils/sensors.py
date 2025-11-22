import numpy as np


def add_gaussian_noise(readings: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to sensor readings."""
    return readings + np.random.normal(0, std, size=readings.shape)


def normalize_readings(readings: np.ndarray, max_range: float) -> np.ndarray:
    """Normalize readings to [0, 1] range."""
    return np.clip(readings / max_range, 0, 1)
