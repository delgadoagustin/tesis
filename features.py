import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.fft import fft, fftfreq

# Aqui desarrollo cada calculo de las caracteristicas por separado

def amplitude(signal):
    """Calcula la amplitud de la señal."""
    return np.max(signal) - np.min(signal)

def pulse_width(signal, threshold=0.5):
    """Calcula el ancho de pulso de la señal."""
    threshold_value = np.max(signal) * threshold
    above_threshold = np.where(signal >= threshold_value)[0]
    
    if len(above_threshold) == 0:
        return 0
    
    return above_threshold[-1] - above_threshold[0]

def rise_time(signal, threshold=0.1):
    """Calcula el tiempo de subida de la señal."""
    threshold_value = np.max(signal) * threshold
    above_threshold = np.where(signal >= threshold_value)[0]
    
    if len(above_threshold) == 0:
        return 0
    
    return above_threshold[0]
    
def fall_time(signal, threshold=0.9):
    """Calcula el tiempo de bajada de la señal."""
    threshold_value = np.max(signal) * threshold
    above_threshold = np.where(signal >= threshold_value)[0]
    
    if len(above_threshold) == 0:
        return 0
    
    return len(signal) - above_threshold[-1]

def area_under_curve(signal, threshold=0):
    """Calcula el área bajo la curva de la señal."""
    signal = np.array(signal)
    if len(signal) == 0:
        return 0
    
    # Aplicar el umbral
    signal[signal < threshold] = 0
    
    # Calcular el área bajo la curva
    return np.trapezoid(signal)

def ascending_slope(signal):
    """Calcula la pendiente ascendente de la señal."""
    if len(signal) < 2:
        return 0
    
    return np.mean(np.diff(signal))

def descending_slope(signal):
    """Calcula la pendiente descendente de la señal."""
    if len(signal) < 2:
        return 0
    
    return -np.mean(np.diff(signal))

def simetry(signal):
    """Calcula la simetría de la señal."""
    if len(signal) == 0:
        return 0
    
    mid = len(signal) // 2
    left_half = signal[:mid]
    right_half = signal[-mid:][::-1]  # Invertir la segunda mitad
    
    return np.mean(left_half - right_half)

def energy(signal):
    """Calcula la energía de la señal."""
    return np.sum(np.square(signal))

def max_curvature(signal):
    """Calcula la máxima curvatura de la señal."""
    if len(signal) < 3:
        return 0
    
    second_derivative = np.diff(np.diff(signal))
    return np.max(np.abs(second_derivative))

def center_of_mass(signal):
    """Calcula el centro de masa de la señal."""
    if len(signal) == 0:
        return 0
    
    total_area = np.sum(signal)
    if total_area == 0:
        return 0
    
    x = np.arange(len(signal))
    return np.sum(x * signal) / total_area

def skewness(signal):
    """Calcula el sesgo de la señal."""
    if len(signal) == 0:
        return 0
    
    mean = np.mean(signal)
    std_dev = np.std(signal)
    
    if std_dev == 0:
        return 0
    
    return np.mean(((signal - mean) / std_dev) ** 3)

def kurtosis(signal):
    """Calcula la curtosis de la señal."""
    if len(signal) == 0:
        return 0
    
    mean = np.mean(signal)
    std_dev = np.std(signal)
    
    if std_dev == 0:
        return 0
    
    return np.mean(((signal - mean) / std_dev) ** 4) - 3  # Restar 3 para obtener la curtosis centrada

def dominant_frequency(signal, sampling_rate):
    """Calcula la frecuencia dominante de la señal."""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    
    # Tomar solo la parte positiva del espectro
    idx = np.where(xf >= 0)
    xf = xf[idx]
    yf = np.abs(yf[idx])
    
    # Encontrar la frecuencia dominante
    return xf[np.argmax(yf)]

def bandwidth(signal, sampling_rate):
    """Calcula el ancho de banda de la señal."""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    
    # Tomar solo la parte positiva del espectro
    idx = np.where(xf >= 0)
    xf = xf[idx]
    yf = np.abs(yf[idx])
    
    # Normalizar el espectro
    yf /= np.max(yf)
    
    # Encontrar el ancho de banda donde la magnitud es mayor que 0.5
    bandwidth_indices = np.where(yf >= 0.5)[0]
    
    if len(bandwidth_indices) == 0:
        return 0
    
    return xf[bandwidth_indices[-1]] - xf[bandwidth_indices[0]]

def total_power(signal):
    """Calcula la potencia total de la señal."""
    return np.sum(np.square(signal)) / len(signal)

def spectral_ratio(signal, sampling_rate):
    """Calcula la relación espectral de la señal."""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    
    # Tomar solo la parte positiva del espectro
    idx = np.where(xf >= 0)
    xf = xf[idx]
    yf = np.abs(yf[idx])
    
    # Calcular la potencia total
    total_power = np.sum(yf ** 2) / N
    
    # Calcular la potencia en cada banda de frecuencia
    low_freq_power = np.sum(yf[xf < 10] ** 2) / N  # Banda baja
    high_freq_power = np.sum(yf[xf >= 10] ** 2) / N  # Banda alta
    
    if high_freq_power == 0:
        return 0
    
    return low_freq_power / high_freq_power

