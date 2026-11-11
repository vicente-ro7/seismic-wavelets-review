"""Read a verified MiniSEED original and a sample-preserving crop."""

import numpy as np
from obspy import read

from seismic_wavelets.data import project_path, sha256_file


def load_trace(root, case):
    """Comprueba hashes, canal, continuidad y conservación de las muestras del recorte."""
    raw = case["waveform_request"]
    trimmed = case["derived_artifact"]
    raw_path = project_path(root, raw["path"])
    trimmed_path = project_path(root, trimmed["path"])
    for path, expected in ((raw_path, raw["sha256"]), (trimmed_path, trimmed["sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"SHA-256 inesperado: {path}")
    (original,) = read(raw_path, format="MSEED")
    (trace,) = read(trimmed_path, format="MSEED")
    rate = trace.stats.sampling_rate
    offset = (trace.stats.starttime - original.stats.starttime) * rate
    first = round(offset)
    if (
        original.id != trace.id
        or original.stats.sampling_rate != rate
        or abs(first - offset) > 0.01
        or first < 0
    ):
        raise ValueError(f"Original y recorte incompatibles: {trace.id}")
    np.testing.assert_array_equal(original.data[first : first + trace.stats.npts], trace.data)
    return trace
