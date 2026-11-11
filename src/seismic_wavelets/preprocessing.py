"""Preparación mínima de copias; filtros y calibración siguen pendientes."""

import numpy as np
from obspy import UTCDateTime


def trim_requested_copy(stream, start, end, grid_tolerance_samples=0.1):
    """Recortar sin interpolar, remuestrear, rellenar ni modificar el stream de entrada.

    Los registros miniSEED pueden traer márgenes distintos en la misma rejilla. Se
    requiere continuidad, una traza por canal y rejillas compatibles antes de recortar.
    Una desalineación real se rechaza: el recorte no la corrige.
    """
    if not stream or len({trace.id for trace in stream}) != len(stream) or stream.get_gaps():
        raise ValueError("Recorte requiere una traza continua por canal.")
    first = stream[0]
    fs = first.stats.sampling_rate
    if fs <= 0:
        raise ValueError("Frecuencia de muestreo inválida.")
    for trace in stream:
        if not np.isclose(trace.stats.sampling_rate, fs, rtol=1e-8, atol=0):
            raise ValueError("Componentes con frecuencias de muestreo distintas.")
        # Medir el desfase en unidades de muestra. Un desplazamiento entero admite recorte
        # común; la parte fraccionaria mide la incompatibilidad de las rejillas.
        offset = (trace.stats.starttime - first.stats.starttime) * fs
        if abs(offset - round(offset)) > grid_tolerance_samples:
            raise ValueError("Rejillas de muestreo no coinciden; no se interpolan.")
    left, right = UTCDateTime(start), UTCDateTime(end)
    if any(trace.stats.starttime > left or trace.stats.endtime < right for trace in stream):
        raise ValueError("Datos insuficientes para cubrir la ventana solicitada.")
    # Elegir muestras interiores a los límites solicitados sobre una copia, sin extender el
    # registro.
    return stream.copy().trim(left, right, pad=False, nearest_sample=False)
