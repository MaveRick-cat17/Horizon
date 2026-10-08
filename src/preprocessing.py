import numpy as np
from scipy.signal import savgol_filter


def add_measurement_noise(data, noise_level=0.01):
    """
    Simulate noisy measurements from a real sensor/camera.
    """
    noise = np.random.normal(
        0,
        noise_level,
        size=data.shape
    )

    return data + noise


def smooth_trajectory(data, window_length=31, polyorder=3):
    """
    Remove high-frequency measurement noise
    while preserving the shape of the trajectory.
    """

    if window_length >= len(data):
        window_length = len(data) - 1

    if window_length % 2 == 0:
        window_length -= 1

    return savgol_filter(
        data,
        window_length=window_length,
        polyorder=polyorder
    )