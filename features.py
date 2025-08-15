import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis

def amplitude_max(signal):
    """
    Calculate the maximum amplitude of a signal.
    
    Parameters:
    signal (np.ndarray): Input signal.
    
    Returns:
    float: Maximum amplitude of the signal.
    """
    return np.max(signal)

def width(signal, threshold=0.5):
    """
    Calcula el ancho del pulso a un cierto umbral desde el máximo hacia los costados.

    :param signal: Array de la señal.
    :param threshold: Umbral como fracción del máximo (ej: 0.5 para 50%).
    :return: Ancho en número de muestras.
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Buscar hacia la izquierda del máximo
    left_idx = max_idx
    while left_idx > 0 and signal[left_idx] > threshold_value:
        left_idx -= 1

    # Buscar hacia la derecha del máximo
    right_idx = max_idx
    while right_idx < len(signal) - 1 and signal[right_idx] > threshold_value:
        right_idx += 1

    return right_idx - left_idx

def fall_time(signal, threshold=0.1):
    """
    Calculate the fall time of a signal from a certain threshold.

    :param signal: Array of the signal.
    :param threshold: Threshold as a fraction of the maximum (e.g., 0.1 for 10%).
    :return: Fall time in number of samples.
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Find the index where the signal falls below the threshold
    while max_idx < len(signal) - 1 and signal[max_idx] > threshold_value:
        max_idx += 1

    return max_idx - np.argmax(signal)

def rise_time(signal, threshold=0.1):
    """
    Calculate the rise time of a signal from a certain threshold.

    :param signal: Array of the signal.
    :param threshold: Threshold as a fraction of the maximum (e.g., 0.1 for 10%).
    :return: Rise time in number of samples.
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Find the index where the signal rises above the threshold
    while max_idx > 0 and signal[max_idx] > threshold_value:
        max_idx -= 1

    return np.argmax(signal) - max_idx

def area_under_curve(signal, threshold=0.5):
    """
    Calculate the area under the curve of a signal using the trapezoidal rule.

    :param signal: Array of the signal.
    :param threshold: Threshold as a fraction of the maximum (e.g., 0.5 for 50%).
    :return: Area under the curve.
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Buscar hacia la izquierda del máximo
    left_idx = max_idx
    while left_idx > 0 and signal[left_idx] > threshold_value:
        left_idx -= 1

    # Buscar hacia la derecha del máximo
    right_idx = max_idx
    while right_idx < len(signal) - 1 and signal[right_idx] > threshold_value:
        right_idx += 1

    # Calcular el área bajo la curva usando la regla del trapecio
    area = np.trapezoid(signal[left_idx:right_idx + 1])
    return area

def ascendent_slope(signal, threshold=0.1):
    """
    Calcula la pendiente media de la parte ascendente del pulso,
    desde el cruce del umbral hasta el máximo.

    :param signal: Array de la señal.
    :param threshold: Umbral como fracción del máximo (ej: 0.1 para 10%).
    :return: Pendiente media de la subida.
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Buscar el índice donde la señal cruza el umbral antes del máximo
    start_idx = 0
    for i in range(max_idx):
        if signal[i] < threshold_value and signal[i+1] >= threshold_value:
            start_idx = i
            break

    delta_y = signal[max_idx] - signal[start_idx]
    delta_x = max_idx - start_idx if max_idx != start_idx else 1  # evitar división por cero
    return delta_y / delta_x

def descendent_slope(signal, threshold=0.1):
    """
    Calcula la pendiente media de la parte descendente del pulso,
    desde el máximo hasta el cruce del umbral.

    :param signal: Array de la señal.
    :param threshold: Umbral como fracción del máximo (ej: 0.1 para 10%).
    :return: Pendiente media de la bajada.
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Buscar el índice donde la señal cruza el umbral después del máximo
    end_idx = len(signal) - 1
    for i in range(max_idx, len(signal) - 1):
        if signal[i] >= threshold_value and signal[i+1] < threshold_value:
            end_idx = i + 1
            break

    delta_y = signal[max_idx] - signal[end_idx]
    delta_x = max_idx - end_idx if max_idx != end_idx else 1  # evitar división por cero
    return delta_y / delta_x

def pulse_symmetry(signal, threshold=0.5):
    """
    Analiza la simetría/asimetría del pulso comparando la subida y bajada alrededor del máximo.

    1. Encuentra el cruce del umbral (por defecto 50%) en la subida.
    2. Calcula la distancia desde ese punto al máximo.
    3. Extrae el segmento simétrico en la bajada.
    4. Calcula el error cuadrático medio (MSE) entre ambos segmentos.

    :param signal: Array de la señal.
    :param threshold: Umbral como fracción del máximo (ej: 0.5 para 50%).
    :return: mse, segmento_subida, segmento_bajada
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # 1. Encuentra el cruce del umbral en la subida
    start_idx = 0
    for i in range(max_idx):
        if signal[i] < threshold_value and signal[i+1] >= threshold_value:
            start_idx = i + 1
            break

    # 2. Distancia desde el cruce al máximo
    width = max_idx - start_idx

    # 3. Extrae el segmento simétrico en la bajada
    end_idx = max_idx + width
    if end_idx >= len(signal):
        end_idx = len(signal) - 1

    segment_rise = signal[start_idx:max_idx+1]
    segment_fall = signal[max_idx:end_idx+1]

    # Igualar longitudes si es necesario
    min_len = min(len(segment_rise), len(segment_fall))
    segment_rise = segment_rise[:min_len]
    segment_fall = segment_fall[:min_len]

    # 4. Calcula el error cuadrático medio
    mse = np.mean((segment_rise - segment_fall[::-1])**2)  # compara subida vs bajada invertida

    return mse, segment_rise, segment_fall

def max_curvature(signal, threshold=0.5):
    """
    Calculates the maximum curvature (second derivative) within the pulse width at a given threshold.

    :param signal: Array of the signal.
    :param threshold: Threshold as a fraction of the maximum (e.g., 0.5 for 50%).
    :return: Maximum absolute curvature value and its index (relative to the signal).
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Find left crossing
    left_idx = max_idx
    while left_idx > 0 and signal[left_idx] > threshold_value:
        left_idx -= 1

    # Find right crossing
    right_idx = max_idx
    while right_idx < len(signal) - 1 and signal[right_idx] > threshold_value:
        right_idx += 1

    # Compute second derivative (curvature) in the window
    window = signal[left_idx:right_idx+1]
    curvature = np.diff(window, n=2)  # second discrete derivative

    if len(curvature) == 0:
        return np.nan, None

    max_curv_idx = np.argmax(np.abs(curvature))
    max_curv_value = curvature[max_curv_idx]

    # Convert to index in the original signal (offset by left_idx + 1)
    signal_idx = left_idx + max_curv_idx + 1

    return max_curv_value, signal_idx

def pulse_kurtosis(signal, threshold=0.5):
    """
    Calculates the kurtosis within the pulse width at a given threshold.

    :param signal: Array of the signal.
    :param threshold: Threshold as a fraction of the maximum (e.g., 0.5 for 50%).
    :return: Kurtosis value within the pulse width.
    """
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Find left crossing
    left_idx = max_idx
    while left_idx > 0 and signal[left_idx] > threshold_value:
        left_idx -= 1

    # Find right crossing
    right_idx = max_idx
    while right_idx < len(signal) - 1 and signal[right_idx] > threshold_value:
        right_idx += 1

    window = signal[left_idx:right_idx+1]
    if len(window) < 4:
        return np.nan  # Kurtosis is not defined for very short windows

    return kurtosis(window)

def pulse_center_of_mass(signal, x=None, threshold=0.5):
    """
    Calculates the center of mass within the pulse width at a given threshold.

    :param signal: Array of the signal.
    :param x: Array of x values (same length as signal). If None, uses indices.
    :param threshold: Threshold as a fraction of the maximum (e.g., 0.5 for 50%).
    :return: Center of mass within the pulse width.
    """
    if x is None:
        x = np.arange(len(signal))
    max_idx = np.argmax(signal)
    max_value = signal[max_idx]
    threshold_value = max_value * threshold

    # Find left crossing
    left_idx = max_idx
    while left_idx > 0 and signal[left_idx] > threshold_value:
        left_idx -= 1

    # Find right crossing
    right_idx = max_idx
    while right_idx < len(signal) - 1 and signal[right_idx] > threshold_value:
        right_idx += 1

    window_signal = signal[left_idx:right_idx+1]
    window_x = x[left_idx:right_idx+1]

    if np.sum(window_signal) == 0:
        return np.nan

    center_of_mass = np.sum(window_x * window_signal) / np.sum(window_signal)
    return center_of_mass

def energy_above_threshold(signal, x=None, threshold=0.5):
    """
    Computes the energy of the signal above a threshold (as a fraction of the maximum).

    :param signal: Array of the signal.
    :param x: Array of x values (same length as signal). If None, uses indices (spacing=1).
    :param threshold: Threshold as a fraction of the maximum (e.g., 0.5 for 50%).
    :return: Energy above the threshold.
    """
    if x is None:
        spacing = 1.0
    else:
        spacing = np.mean(np.diff(x))
    max_value = np.max(signal)
    threshold_value = max_value * threshold

    above = signal > threshold_value
    if np.sum(above) == 0:
        return 0.0

    # Energy = sum(signal^2) * spacing, only above threshold
    return np.sum(signal[above]**2) * spacing

def interpolated_bandwidth(signal, x):
    """
    Calculates the bandwidth of an interpolated signal using its time axis.

    :param signal: Array of the interpolated signal.
    :param x: Array of time values (same length as signal).
    :return: Bandwidth in Hz.
    """
    N = len(signal)
    # Calculate effective sampling frequency from time axis
    fs = 1 / np.mean(np.diff(x))
    freqs = fftfreq(N, d=1/fs)
    spectrum = np.abs(fft(signal))**2
    spectrum = spectrum[:N//2]
    freqs = freqs[:N//2]

    total_energy = np.sum(spectrum)
    cumulative = np.cumsum(spectrum) / total_energy

    # Bandwidth: frequency where cumulative energy reaches 99%
    idx = np.searchsorted(cumulative, 0.99)
    return freqs[idx]


## ESTA FUNCION COMENTADA SIEMPRE VA RESULTAR 0 PORQUE ES UNA SEÑAL DE PULSO UNICO
## POR LO TANTO, LA FRECUENCIA DOMINANTE SIEMPRE VA A SER 0
# Uncomment if you want to use this function, but note it will always return 0 for a single pulse signal

# def interpolated_dominant_frequency(signal, x):
#     """
#     Calculates the dominant frequency of an interpolated signal using its time axis.

#     :param signal: Array of the interpolated signal.
#     :param x: Array of time values (same length as signal).
#     :return: Dominant frequency in Hz.
#     """
#     N = len(signal)
#     fs = 1 / np.mean(np.diff(x))
#     freqs = fftfreq(N, d=1/fs)
#     spectrum = np.abs(fft(signal))**2
#     spectrum = spectrum[:N//2]
#     freqs = freqs[:N//2]

#     idx = np.argmax(spectrum)
#     return freqs[idx]