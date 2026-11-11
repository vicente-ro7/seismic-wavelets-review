"""Pruebas pequeñas de instalación y convenciones DSP.

Un tono con frecuencia conocida controla el eje FFT. La reconstrucción DWT en longitudes pares
e impares controla la extensión y el recorte de síntesis.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest
import pywt
from scipy import fft

import seismic_wavelets


def test_project_imports_from_source():
    root = Path(__file__).resolve().parents[1]
    assert Path(seismic_wavelets.__file__).is_relative_to(root / "src")
    for name in ("data", "synthetic", "preprocessing", "transforms", "metrics", "figures"):
        importlib.import_module(f"seismic_wavelets.{name}")


def test_fft_known_frequency():
    # Tono en un bin exacto: comprobación de muestreo y convención de frecuencia.
    fs = 128.0
    samples = np.sin(2 * np.pi * 8.0 * np.arange(512) / fs)
    frequencies = fft.rfftfreq(samples.size, d=1 / fs)
    assert frequencies[np.argmax(np.abs(fft.rfft(samples)))] == 8.0


@pytest.mark.parametrize("length", [511, 512])
def test_dwt_reconstruction_preserves_samples(length):
    # La extensión puede generar una muestra extra para longitudes impares.
    samples = np.random.default_rng(20261012).standard_normal(length)
    coeffs = pywt.wavedec(samples, "db2", mode="symmetric", level=3)
    recovered = pywt.waverec(coeffs, "db2", mode="symmetric")[:length]
    np.testing.assert_allclose(recovered, samples, rtol=1e-11, atol=1e-11)
