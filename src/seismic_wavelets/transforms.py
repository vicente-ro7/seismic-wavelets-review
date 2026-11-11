"""Representaciones y potencia por ventana para el piloto comparativo."""

import numpy as np
import pywt
from scipy import fft, signal


def fourier_power(frames, fs, *, nfft=None):
    """Calcula potencia unilateral por ventana con una Hann periódica.

    El último eje contiene las muestras. La salida conserva los ejes de lote
    y termina en bins de frecuencia en Hz. Se divide |FFT|² por el producto
    de NFFT y la energía de la ventana. Por Parseval, al sumar los bins se
    recupera la media cuadrática ponderada por Hann². Se duplican los bins
    interiores para incluir sus conjugados negativos. Aumentar NFFT densifica
    la rejilla de frecuencias y conserva la duración efectiva de la ventana.
    """
    width = frames.shape[-1]
    nfft = width if nfft is None else nfft
    taper = signal.get_window("hann", width, fftbins=True)
    power = np.abs(fft.rfft(frames * taper, n=nfft, axis=-1)) ** 2 / (nfft * np.sum(taper**2))
    power[..., 1 : -1 if nfft % 2 == 0 else None] *= 2
    return fft.rfftfreq(nfft, 1 / fs), power


def frame_ratios(samples, fs, width, hop, background, band, method):
    """Devuelve índices de cierre y razones adimensionales de potencia.

    La potencia corta se calcula sobre width muestras y se evalúa cada hop
    muestras. El fondo promedia background potencias previas. Su última
    ventana termina antes de que empiece la ventana corta actual, aunque
    las ventanas que componen el fondo pueden solaparse entre sí.

    La entrada multicanal tiene forma (componentes, muestras) y unidades
    compatibles. Se suman las potencias antes de formar la razón. Los
    cierres se miden desde la primera muestra de la entrada. El ejecutor
    agrega el desplazamiento del preprocesamiento al regresar a la traza.
    """
    frames = np.lib.stride_tricks.sliding_window_view(samples, width, axis=-1)[..., ::hop, :]
    match method:
        case "STA/LTA":
            power = np.mean(frames**2, axis=-1)
        case "STFT":
            frequencies, spectrum = fourier_power(frames, fs)
            power = spectrum[..., (frequencies >= band[0]) & (frequencies < band[1])].sum(axis=-1)
        case "DWT":
            _, *details = pywt.wavedec(frames, "haar", mode="periodization", level=4, axis=-1)
            # Haar ortonormal en una ventana diádica. La energía de los detalles
            # 4, 3 y 2 se divide por el número de muestras de entrada.
            power = sum(np.sum(detail**2, axis=-1) for detail in details[:-1]) / width
        case _:
            raise ValueError(f"Método de detección desconocido: {method}")
    if power.ndim == 2:
        power = power.sum(axis=0)
    frame_count = len(power)
    # Redondear hacia arriba en saltos garantiza que las dos ventanas no compartan muestras.
    separation = (width + hop - 1) // hop
    first = background - 1 + separation
    if frame_count <= first:
        raise ValueError("La señal es demasiado corta para construir el fondo.")
    baseline = np.lib.stride_tricks.sliding_window_view(power, background)
    baseline = baseline[: frame_count - first].mean(axis=1)
    recent = power[first:]
    # Una ventana de fondo exactamente nula carece de una razón interpretable.
    ratios = np.divide(recent, baseline, out=np.zeros_like(recent), where=baseline > 0)
    ends = width - 1 + hop * np.arange(first, frame_count)
    return ends, ratios


def wavelet_map(samples, fs, frequencies, wavelet_name, precision):
    """Devuelve pseudofrecuencias en Hz y |CWT|² por escala y muestra.

    La conversión de frecuencia a escala usa la frecuencia central de la
    wavelet y el intervalo de muestreo 1/fs. Se excluyen con NaN los centros
    cuyo soporte digital cruza un borde. El margen crece al bajar la
    frecuencia. El cálculo centrado sirve para descripción retrospectiva
    y puede incorporar muestras posteriores al tiempo del centro.
    """
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    scales = pywt.frequency2scale(wavelet, frequencies / fs, precision=precision)
    coefficients, actual_frequencies = pywt.cwt(
        samples, scales, wavelet, sampling_period=1 / fs, method="fft", precision=precision
    )
    # La cota cubre ambos lados del soporte tabulado y el paso de diferenciación.
    margin = np.ceil(scales * max(abs(wavelet.lower_bound), abs(wavelet.upper_bound))).astype(int)
    indices = np.arange(len(samples))
    valid = (indices >= (margin + 1)[:, None]) & (indices < len(samples) - (margin + 1)[:, None])
    return actual_frequencies, np.where(valid, np.abs(coefficients) ** 2, np.nan)


def fourier_amplitude(samples, fs):
    """Devuelve el espectro global unilateral en Hz y en las unidades de entrada.

    Normaliza por el número de muestras y duplica los bins con pareja
    conjugada. La magnitud resume el tramo completo sin localizar eventos.
    """
    amplitude = np.abs(fft.rfft(samples)) / len(samples)
    amplitude[1 : -1 if len(samples) % 2 == 0 else None] *= 2
    return fft.rfftfreq(len(samples), 1 / fs), amplitude
