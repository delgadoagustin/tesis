import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.fft import fft, fftfreq

def calculate_features(data_row: np.ndarray) -> dict[str, float]:
    """
    Calculate and print various features from a given row of data.

    Parameters:
        data_row (pd.Series or np.ndarray): A single row of data to analyze.
    """
    
    # Extract data
    x = np.arange(1, len(data_row) + 1)
    y = data_row

    # Interpolation
    p = PchipInterpolator(x, y)
    xnew = np.linspace(1, len(data_row), num=100)
    ynew = p(xnew)

    # Amplitude
    amplitude_max = np.max(ynew)

    # Width
    threshold = 0.35 * amplitude_max
    above_threshold = np.where(ynew > threshold)[0]
    width = xnew[above_threshold[-1]] - xnew[above_threshold[0]] if len(above_threshold) > 0 else 0

    # Rise and Fall time
    amplitude_10 = 0.1 * amplitude_max
    amplitude_90 = 0.9 * amplitude_max
    start_rise_index = np.argmax(ynew > amplitude_10)
    end_rise_index = np.argmax(ynew > amplitude_90)
    rise_time = xnew[end_rise_index] - xnew[start_rise_index]
    start_fall_index = len(ynew) - np.argmax(ynew[::-1] > amplitude_90) - 1
    end_fall_index = len(ynew) - np.argmax(ynew[::-1] > amplitude_10) - 1
    fall_time = xnew[end_fall_index] - xnew[start_fall_index]

    # Area under the curve over threshold
    if len(above_threshold) > 0:
        area_above_threshold = np.trapezoid(ynew[above_threshold], xnew[above_threshold])
    else:
        area_above_threshold = 0

    # Slopes
    ascending_slope = (ynew[end_rise_index] - ynew[start_rise_index]) / (xnew[end_rise_index] - xnew[start_rise_index])
    descending_slope = (ynew[end_fall_index] - ynew[start_fall_index]) / (xnew[end_fall_index] - xnew[start_fall_index])

    # Symmetry
    symmetry = (end_rise_index - start_rise_index) / width if width > 0 else 0

    # Pulse energy
    if len(above_threshold) > 0:
        pulse_energy = np.trapezoid(ynew[start_rise_index:end_fall_index], xnew[start_rise_index:end_fall_index])
    else:
        pulse_energy = 0

    # Max curvature
    curvature = np.diff(ynew, 2)
    max_curvature = np.max(curvature)

    # Center of mass
    if len(above_threshold) > 0:
        center_of_mass = np.sum(xnew[above_threshold] * ynew[above_threshold]) / np.sum(ynew[above_threshold])
    else:
        center_of_mass = 0

    # Skewness
    if len(above_threshold) > 0:
        skewness = np.sum((xnew[above_threshold] - np.mean(xnew[above_threshold]))**3 * ynew[above_threshold]) / (np.sum(ynew[above_threshold]) * np.std(xnew[above_threshold])**3)
    else:
        skewness = 0

    # Kurtosis
    if len(above_threshold) > 0:
        kurtosis = np.sum((xnew[above_threshold] - np.mean(xnew[above_threshold]))**4 * ynew[above_threshold]) / (np.sum(ynew[above_threshold]) * np.std(xnew[above_threshold])**4)
    else:
        kurtosis = 0

    # Dominant frequency
    N = len(ynew)
    T = 1.0 / 100.0
    yf = fft(ynew)
    xf = fftfreq(N, T)[:N//2]
    dominant_frequency = xf[np.argmax(2.0/N * np.abs(yf[0:N//2]))]

    # Bandwidth
    bandwidth = np.sum(np.abs(ynew)) / len(ynew)

    # Total power
    total_power = np.sum(ynew**2)

    # Spectral ratio
    spectral_ratio = np.sum(ynew[0:int(len(ynew)/2)]**2) / np.sum(ynew[int(len(ynew)/2):]**2)

    # build dictionary with all features
    features = {
        "amplitude": amplitude_max,
        "width": width,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "area_above_threshold": area_above_threshold,
        "ascending_slope": ascending_slope,
        "descending_slope": descending_slope,
        "symmetry": symmetry,
        "pulse_energy": pulse_energy,
        "max_curvature": max_curvature,
        "center_of_mass": center_of_mass,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "dominant_frequency": dominant_frequency,
        "bandwidth": bandwidth,
        "total_power": total_power,
        "spectral_ratio": spectral_ratio
    }
    # Print features
    # for key, value in features.items():
    #     print(f"{key}: {value}")
    # Return features
    return features