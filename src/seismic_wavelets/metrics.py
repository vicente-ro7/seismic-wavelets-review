"""Reglas de asociación y resúmenes del piloto exploratorio."""

import numpy as np

from seismic_wavelets.detection import CausalWindowDetector


def select_triggers(scores, ends, threshold, holdoff):
    """Selecciona cruces ascendentes y aplica una espera en muestras de la traza.

    El detector recibe un indicador por cierre. ends traduce el índice en
    esa rejilla a la muestra de la forma de onda. La espera se mide entre
    disparos aceptados, por lo que un cruce descartado no reinicia el plazo.
    """
    detector = CausalWindowDetector(1, threshold, lambda window: window[0])
    crossings = detector.process(scores, start_sample=0)
    accepted = []
    for crossing in crossings:
        sample = int(ends[crossing.available_sample])
        if not accepted or sample - accepted[-1] >= holdoff:
            accepted.append(sample)
    return accepted


def associate_p(samples, fs, pick, early, late, next_phase=None):
    """Devuelve el primer disparo dentro de la tolerancia respecto a P.

    La referencia y ambas tolerancias están en segundos desde el inicio
    de la traza. Si se conoce la fase siguiente, su instante excluye ese
    disparo y los posteriores. El desfase es positivo después de P.
    """
    for sample in samples:
        seconds = sample / fs
        if pick - early <= seconds <= pick + late and (next_phase is None or seconds < next_phase):
            return {
                "available_sample": sample,
                "available_s": seconds,
                "relative_to_p_s": seconds - pick,
            }
    return None


def describe_packet(power, times, frequencies, start, end):
    """Mide la cresta en el centro conocido y la duración a media potencia.

    Selecciona la fila de mayor potencia en el centro del paquete.
    En esa fila busca el máximo dentro del soporte de referencia y mide
    el tramo contiguo que supera la mitad de ese máximo. El resultado
    multiplica el número de centros por el salto temporal en segundos.
    """
    inside = (times >= start) & (times < end)
    center = np.argmin(abs(times - (start + end) / 2))
    row = np.nanargmax(power[:, center])
    profile = power[row]
    peak = int(np.nanargmax(np.where(inside, profile, np.nan)))
    above = profile >= profile[peak] / 2
    first = last = peak
    while first > 0 and above[first - 1]:
        first -= 1
    while last < len(above) - 1 and above[last + 1]:
        last += 1
    # Cada centro representa una celda de ancho igual al salto de la rejilla.
    return {
        "ridge_hz": float(frequencies[row]),
        "half_power_width_s": float((last - first + 1) * (times[1] - times[0])),
    }
