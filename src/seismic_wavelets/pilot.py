"""Signal preparation, causal scoring and timing for the Mugello example."""

from time import perf_counter_ns

import numpy as np

from seismic_wavelets.metrics import associate_p, select_triggers
from seismic_wavelets.transforms import frame_ratios

METHODS = ("STA/LTA", "STFT", "DWT")
COLORS = ("#146078", "#b24721", "#645296")
PROTOCOL = {
    "fs_hz": 100,
    "offset_samples": 100,
    "frame_samples": 64,
    "hop_samples": 5,
    "background_frames": 100,
    "band_hz": (3.125, 25),
    "holdoff_s": 1.0,
    "early_s": 0.1,
    "late_s": 1.5,
}
FS = PROTOCOL["fs_hz"]
OFFSET = PROTOCOL["offset_samples"]
WIDTH = PROTOCOL["frame_samples"]
HOP = PROTOCOL["hop_samples"]
TIMING = {"clock": "perf_counter_ns", "warmups": 2, "repetitions": 20}


def prepare(samples):
    """Centra la traza con la media del primer segundo y descarta ese tramo.

    El centrado se puede calcular antes de emitir cualquier disparo. El
    llamador agrega OFFSET a los índices para recuperar la rejilla original.
    """
    return samples[OFFSET:] - np.mean(samples[:OFFSET])


def score(samples, method):
    """Devuelve marcas de disponibilidad sobre el eje temporal original."""
    ends, ratios = frame_ratios(
        samples, FS, WIDTH, HOP, PROTOCOL["background_frames"], PROTOCOL["band_hz"], method
    )
    return ends + OFFSET, ratios


def detect(samples, method, threshold):
    """Aplica el umbral y la espera y devuelve la serie para la auditoría local."""
    ends, ratios = score(samples, method)
    triggers = select_triggers(ratios, ends, threshold, round(PROTOCOL["holdoff_s"] * FS))
    return triggers, ends, ratios


def match_p(triggers, pick, s=None):
    """Aplica una sola regla de asociación a señales sintéticas y reales."""
    return associate_p(triggers, FS, pick, PROTOCOL["early_s"], PROTOCOL["late_s"], s)


def measure_batch(operation, *args, sample_count):
    """Mide tiempo transcurrido de una operación en memoria y devuelve su último resultado."""
    for _ in range(TIMING["warmups"]):
        operation(*args)
    elapsed_ns = []
    for _ in range(TIMING["repetitions"]):
        start = perf_counter_ns()
        result = operation(*args)
        elapsed_ns.append(perf_counter_ns() - start)
    # El denominador es la duración nominal de la entrada al cálculo, después del centrado.
    median_ms = float(np.median(elapsed_ns)) / 1e6
    return result, {
        "samples_processed": sample_count,
        "elapsed_ns": elapsed_ns,
        "median_ms": median_ms,
        "p95_ms": float(np.quantile(elapsed_ns, 0.95)) / 1e6,
        "median_ms_per_signal_s": median_ms / (sample_count / FS),
    }
