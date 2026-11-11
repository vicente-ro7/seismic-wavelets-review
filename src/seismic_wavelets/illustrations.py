"""Rutinas numéricas para las figuras didácticas de Fourier y wavelets.

Las entradas son arreglos NumPy; fs está en Hz y los tiempos en segundos. Cada representación
tiene una convención propia de amplitud y soporte. Las comprobaciones fallan si la
implementación se aparta de la convención de la figura. Los parámetros ilustrativos no
constituyen un protocolo de detección.
"""

import hashlib

import numpy as np
import pywt
from scipy import fft, signal, special


def array_sha256(array):
    """Identifica un arreglo por tipo, forma y bytes en orden contiguo.

    Permite rastrear la misma señal entre figuras sin redondear valores. Arreglos de igual
    valor con otro tipo o representación binaria pueden tener un hash diferente.
    """
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def packet(time, amplitude, frequency_hz, support_s):
    """Construye un coseno con envolvente seno cuadrado y soporte [inicio, fin).

    Fuera del soporte la envolvente es cero. Dentro, la fase del coseno se mide desde el
    inicio del paquete. Devuelve la señal modulada y la envolvente por separado, en las mismas
    unidades que amplitude.
    """
    start, end = support_s
    if end <= start:
        raise ValueError("Soporte vacío.")
    mask = (time >= start) & (time < end)
    envelope = np.zeros_like(time)
    envelope[mask] = amplitude * np.sin(np.pi * (time[mask] - start) / (end - start)) ** 2
    carrier = np.cos(2 * np.pi * frequency_hz * (time - start))
    return envelope * carrier, envelope


def shared_signal(settings):
    """Genera una rejilla uniforme y suma los paquetes del contrato sin ruido.

    Con N muestras a fs Hz, la duración nominal es N/fs y la última muestra cae en (N−1)/fs.
    Devuelve tiempo, señal total, paquetes y envolventes, estos últimos con un paquete por
    fila.
    """
    fs, count = settings["fs_hz"], settings["n_samples"]
    if count != round(fs * settings["nominal_duration_s"]) or settings["noise"] is not None:
        raise ValueError("El contrato requiere la señal limpia de longitud exacta.")
    time = np.arange(count, dtype=float) / fs
    pieces, envelopes = [], []
    for item in settings["packets"]:
        piece, envelope = packet(time, item["amplitude"], item["frequency_hz"], item["support_s"])
        pieces.append(piece)
        envelopes.append(envelope)
    return time, np.sum(pieces, axis=0), np.array(pieces), np.array(envelopes)


def one_sided_amplitude(samples, fs):
    """Devuelve frecuencia en Hz y amplitud unilateral de una señal real.

    Divide el módulo de la DFT por N y suma la contribución del bin negativo conjugado
    duplicando los bins interiores. DC y Nyquist (solo si N es par) quedan sin duplicar. El
    resultado es amplitud, con las unidades de la entrada, sin ventana adicional.
    """
    amplitude = np.abs(fft.rfft(samples)) / len(samples)
    amplitude[1 : -1 if len(samples) % 2 == 0 else None] *= 2
    return fft.rfftfreq(len(samples), 1 / fs), amplitude


def stft_illustration(samples, fs, length, hop, nfft):
    """Calcula la STFT unilateral de ventanas completas con Hann periódica.

    length, hop y nfft se expresan en muestras. Las columnas son centros (inicio +
    length/2)/fs y las filas son frecuencias separadas por fs/nfft. El relleno interno de cada
    FFT cambia el espaciado de bins, no la resolución física de la ventana. Devuelve
    frecuencias, tiempos, coeficientes complejos y la convención comprobada.
    """
    window = signal.get_window("hann", length, fftbins=True)
    frequencies, times, coefficients = signal.stft(
        samples,
        fs=fs,
        window=window,
        nperseg=length,
        noverlap=length - hop,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    starts = np.arange(len(times)) * hop
    expected_centers = (starts + length / 2) / fs
    np.testing.assert_allclose(times, expected_centers, rtol=0, atol=1e-13)
    if coefficients.shape[1] != (len(samples) - length) // hop + 1:
        raise RuntimeError("Número de ventanas STFT inesperado.")
    # Referencia independiente: aplicar la ventana y dividir por su suma reproduce el escalado
    # espectral de SciPy sin confundirlo con una PSD.
    direct = np.array(
        [
            fft.rfft(samples[start : start + length] * window, n=nfft) / window.sum()
            for start in starts
        ]
    ).T
    np.testing.assert_allclose(coefficients, direct, rtol=1e-11, atol=1e-13)
    metadata = {
        "window": "periodic_hann",
        "nperseg": length,
        "hop_samples": hop,
        "noverlap": length - hop,
        "nfft": nfft,
        "detrend": False,
        "boundary": None,
        "padded": False,
        "return_onesided": True,
        "scaling": "spectrum",
        "window_duration_s": length / fs,
        "frequency_spacing_hz": fs / nfft,
        "center_convention": "SciPy: (start_sample + L/2)/fs; not (L-1)/2",
        "first_center_s": float(times[0]),
        "last_center_s": float(times[-1]),
        "frame_count": len(times),
        "first_covered_sample_s": float(starts[0] / fs),
        "last_covered_sample_s": float((starts[-1] + length - 1) / fs),
        "direct_windowed_fft_verified": True,
    }
    return frequencies, times, coefficients, metadata


def corrected_morlet(settings):
    """Construye una Morlet compleja de media casi nula y norma L² unitaria.

    Resta la componente constante de la portadora antes de multiplicar por la gaussiana.
    Integra por trapecios en una rejilla finita para estimar tanto la media compleja como la
    integral de |ψ|². Devuelve una función evaluable en u adimensional y las comprobaciones de
    esa rejilla.
    """
    grid = np.linspace(*settings["domain_u"], settings["domain_samples"])
    omega = settings["omega"]
    raw = (np.exp(1j * omega * grid) - np.exp(-(omega**2) / 2)) * np.exp(-(grid**2) / 2)
    norm = np.sqrt(np.trapezoid(np.abs(raw) ** 2, grid))

    def prototype(u):
        """Evalúa la Morlet corregida usando la misma constante de normalización."""
        return (np.exp(1j * omega * u) - np.exp(-(omega**2) / 2)) * np.exp(-(u**2) / 2) / norm

    mean = np.trapezoid(prototype(grid), grid)
    measured_norm = float(np.trapezoid(np.abs(prototype(grid)) ** 2, grid))
    if (
        abs(mean) > settings["mean_tolerance"]
        or abs(measured_norm - 1) > settings["norm_tolerance"]
    ):
        raise RuntimeError("El prototipo no satisface las comprobaciones de media/norma.")
    return prototype, {
        "omega": omega,
        "normalization": "numerical_complex_L2",
        "normalization_constant": float(norm),
        "domain_u": settings["domain_u"],
        "domain_samples": len(grid),
        "mean_abs": float(abs(mean)),
        "norm_squared": measured_norm,
        "mean_tolerance": settings["mean_tolerance"],
        "norm_tolerance": settings["norm_tolerance"],
    }


def effective_cwt_kernels(settings, fs):
    """Audita los filtros discretos que PyWavelets aplica en la rejilla CWT.

    La frecuencia en Hz se divide por fs antes de convertirla a escala. A partir de la wavelet
    integrada se reconstruye el kernel equivalente por muestreo, inversión y diferencia
    finita. Su pico y la fracción de potencia cercana a Nyquist se contrastan con la
    configuración. Las guardas en muestras dependen del soporte discreto; no son un cono de
    influencia estadístico.
    """
    wavelet = pywt.ContinuousWavelet(settings["wavelet"])
    frequencies = np.geomspace(*settings["frequency_hz"], settings["number_of_frequencies"])
    scales = pywt.frequency2scale(wavelet, frequencies / fs, precision=settings["precision"])
    integrated, grid = pywt.integrate_wavelet(wavelet, precision=settings["precision"])
    integrated = integrated.conj()
    step = grid[1] - grid[0]
    checks, kernels = [], []
    support = max(abs(wavelet.lower_bound), abs(wavelet.upper_bound))
    guards = np.ceil(support * scales).astype(int) + settings["guard_extra_samples"]
    for frequency, scale, guard in zip(frequencies, scales, guards, strict=True):
        # PyWavelets aproxima la dilatación con índices enteros sobre la integral tabulada; la
        # diferencia finita recupera el kernel de convolución equivalente.
        indices = (np.arange(scale * (grid[-1] - grid[0]) + 1) / (scale * step)).astype(int)
        indices = indices[indices < len(integrated)]
        sampled = integrated[indices][::-1]
        kernel = -np.sqrt(scale) * np.diff(sampled)
        kernels.append(kernel)
        # Usar ambos signos porque la wavelet es compleja. La razón de potencias cancela la
        # constante común de normalización de la FFT.
        bins = np.fft.fftfreq(settings["kernel_fft_samples"], 1 / fs)
        power = abs(np.fft.fft(kernel, settings["kernel_fft_samples"])) ** 2
        peak = abs(float(bins[np.argmax(power)]))
        fraction = float(
            power[abs(bins) >= settings["near_nyquist_fraction"] * fs / 2].sum() / power.sum()
        )
        # Dos colas del espectro continuo de cmor: control complementario, no prueba universal.
        alpha = np.pi * np.sqrt(2 * float(wavelet.bandwidth_frequency))
        center = float(wavelet.center_frequency)
        tail = float(
            0.5
            * (
                special.erfc(alpha * (scale / 2 - center))
                + special.erfc(alpha * (scale / 2 + center))
            )
        )
        crop_left = int(np.floor((len(sampled) - 2) / 2))
        crop_right = int(np.ceil((len(sampled) - 2) / 2))
        if guard < max(crop_left + 1, crop_right + 1):
            raise RuntimeError("La guarda no cubre el kernel real y su diferencia.")
        if abs(peak / frequency - 1) > settings["maximum_peak_relative_error"]:
            raise RuntimeError("Pico del kernel incompatible con la frecuencia declarada.")
        if fraction > settings["maximum_near_nyquist_power_fraction"]:
            raise RuntimeError("Kernel con potencia excesiva cerca de Nyquist.")
        checks.append(
            {
                "frequency_hz": float(frequency),
                "scale_samples": float(scale),
                "kernel_length": len(kernel),
                "peak_hz": peak,
                "near_nyquist_power_fraction": fraction,
                "continuous_outside_nyquist_power_fraction": tail,
                "guard_samples": int(guard),
            }
        )
    return (
        frequencies,
        scales,
        guards,
        kernels,
        {
            "wavelet": settings["wavelet"],
            "precision": settings["precision"],
            "bounds": [wavelet.lower_bound, wavelet.upper_bound],
            "grid": "geomspace_hz",
            "discrete_kernel_rule": "same integrate_wavelet, floor indices and "
            "difference as pywt.cwt",
            "guards": "ceil(max(abs(bounds))*scale)+2; support-based, not statistical COI",
            "kernel_checks": checks,
            "maximum_near_nyquist_power_fraction": settings["maximum_near_nyquist_power_fraction"],
            "maximum_peak_relative_error": settings["maximum_peak_relative_error"],
            "limitation": "Numerical adequacy checks; not a proof of alias absence for "
            "arbitrary wavelets/scales",
        },
    )


def cwt_illustration(samples, fs, settings):
    """Devuelve la CWT compleja en una matriz de frecuencia por tiempo.

    Usa las escalas y guardas ya auditadas y pasa 1/fs como período de muestreo. Contrasta las
    frecuencias devueltas por PyWavelets con la rejilla solicitada; las filas representan
    pseudofrecuencias ligadas a la familia y a la escala.
    """
    frequencies, scales, guards, kernels, metadata = effective_cwt_kernels(settings, fs)
    coefficients, actual = pywt.cwt(
        samples,
        scales,
        settings["wavelet"],
        sampling_period=1 / fs,
        method=settings["method"],
        precision=settings["precision"],
    )
    np.testing.assert_allclose(actual, frequencies, rtol=1e-12, atol=1e-12)
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Coeficientes CWT no finitos.")
    metadata.update(
        method=settings["method"],
        sampling_period_s=1 / fs,
        scale_frequency_conversion_verified=True,
    )
    return actual, scales, guards, coefficients, metadata


def dwt_reconstructions(samples, fs, settings):
    """Reconstruye aproximación y detalles DWT en la rejilla de la señal.

    Para cada subbanda conserva solo sus coeficientes y pone las otras a cero antes de la
    síntesis. Por linealidad, la suma debe reproducir la señal; se mide el residuo con norma
    L² relativa. Las bandas en Hz son nominales y la comprobación de norma está restringida al
    caso ortonormal con periodización que acepta esta rutina.
    """
    wavelet = pywt.Wavelet(settings["wavelet"])
    level = settings["level"]
    maximum = pywt.dwt_max_level(len(samples), wavelet.dec_len)
    if level > maximum:
        raise ValueError("Nivel DWT superior al permitido.")
    coefficients = pywt.wavedec(samples, wavelet, mode=settings["mode"], level=level)
    labels = [f"A{level}"] + [f"D{j}" for j in range(level, 0, -1)]
    reconstructions = {}
    for index, label in enumerate(labels):
        # La síntesis aislada permite comparar componentes sobre los mismos instantes, pese al
        # diezmado de los coeficientes.
        selection = [np.zeros_like(values) for values in coefficients]
        selection[index] = coefficients[index].copy()
        recovered = pywt.waverec(selection, wavelet, mode=settings["mode"])
        if len(recovered) != len(samples):
            raise RuntimeError("Longitud de reconstrucción diferente de la rejilla original.")
        reconstructions[label] = recovered
    total = np.sum(list(reconstructions.values()), axis=0)
    relative = float(np.linalg.norm(total - samples) / np.linalg.norm(samples))
    if relative > settings["relative_error_tolerance"]:
        raise RuntimeError("Error relativo de reconstrucción DWT fuera de tolerancia.")
    metadata = {
        **settings,
        "maximum_allowed_level": maximum,
        "coefficient_lengths": [len(x) for x in coefficients],
        "full_grid_reconstruction_lengths": {
            key: len(value) for key, value in reconstructions.items()
        },
        "relative_reconstruction_error": relative,
        "orthogonal": wavelet.orthogonal,
        "nominal_bands_hz": {f"D{j}": [fs / 2 ** (j + 1), fs / 2**j] for j in range(1, level + 1)},
        "nominal_approximation_band_hz": [0, fs / 2 ** (level + 1)],
    }
    if wavelet.orthogonal and settings["mode"] == "periodization":
        energy = float(sum(np.sum(abs(values) ** 2) for values in coefficients))
        reference = float(np.sum(abs(samples) ** 2))
        metadata["discrete_squared_norm_relative_error"] = abs(energy / reference - 1)
        if metadata["discrete_squared_norm_relative_error"] > settings["relative_error_tolerance"]:
            raise RuntimeError("La norma discreta no se conserva en el caso ortonormal.")
    return reconstructions, total, metadata


def relative_db(coefficients, reference=None, floor=-40):
    """Convierte módulo a dB relativos con un piso para la visualización.

    Se calcula 20 log10(|coeficiente|/referencia), donde la referencia es un máximo propio o
    compartido explícitamente. El piso evita log10(0). Normalizar dos métodos por máximos
    distintos no vuelve comparables sus energías.
    """
    magnitude = np.abs(coefficients)
    maximum = float(magnitude.max()) if reference is None else float(reference)
    if maximum <= 0:
        raise ValueError("Referencia dB nula.")
    values = 20 * np.log10(np.maximum(magnitude / maximum, 10 ** (floor / 20)))
    return np.maximum(values, floor), maximum


def minmax_bins(samples, bins):
    """Devuelve centro, mínimo y máximo de cada intervalo de muestras para graficar.

    Los bordes son índices enteros y cada bloque es semiabierto. Se conservan impulsos de
    ambos signos dentro del bloque. Este resumen gráfico no se usa como señal remuestreada.
    """
    edges = np.linspace(0, len(samples), min(bins, len(samples)) + 1, dtype=int)
    minimum = np.array(
        [samples[left:right].min() for left, right in zip(edges[:-1], edges[1:], strict=True)]
    )
    maximum = np.array(
        [samples[left:right].max() for left, right in zip(edges[:-1], edges[1:], strict=True)]
    )
    centers = (edges[:-1] + edges[1:] - 1) / 2
    return centers, minimum, maximum, edges
