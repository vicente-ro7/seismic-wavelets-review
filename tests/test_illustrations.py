"""Contrasta las ilustraciones con referencias analíticas y numéricas.

Los controles cubren amplitud y rejillas de FFT/STFT, norma de Morlet, frecuencias y guardas
CWT, reconstrucción DWT y preservación de extremos visuales.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from seismic_wavelets.illustrations import (
    corrected_morlet,
    cwt_illustration,
    dwt_reconstructions,
    effective_cwt_kernels,
    minmax_bins,
    one_sided_amplitude,
    packet,
    shared_signal,
    stft_illustration,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/illustrations.json").read_text(encoding="utf-8"))


def test_common_signal_support_and_sampling():
    time, samples, pieces, envelopes = shared_signal(CONFIG["shared_synthetic"])
    assert len(time) == 1600 and time[-1] == 7.995
    np.testing.assert_array_equal(samples, pieces.sum(axis=0))
    for index, support in enumerate([(1, 2.5), (4, 4.75)]):
        outside = (time <= support[0]) | (time >= support[1])
        assert np.all(pieces[index, outside] == 0)
        assert np.all(envelopes[index, outside] == 0)
    np.testing.assert_allclose(envelopes.max(axis=1), [1, 0.8], atol=1e-14)
    with pytest.raises(ValueError, match="Soporte vacío"):
        packet(time, 1, 8, [2, 1])


@pytest.mark.parametrize("count", [200, 201])
def test_fft_amplitudes_dc_even_odd_and_nyquist(count):
    index = np.arange(count)
    samples = 3 + 2 * np.cos(2 * np.pi * 7 * index / count)
    frequencies, amplitude = one_sided_amplitude(samples, count)
    assert frequencies[7] == 7
    np.testing.assert_allclose(amplitude[[0, 7]], [3, 2], atol=1e-13)
    if count % 2 == 0:
        _, nyquist = one_sided_amplitude((-1.0) ** index, count)
        assert abs(nyquist[-1] - 1) < 1e-14
    else:
        _, last = one_sided_amplitude(2 * np.cos(2 * np.pi * (count // 2) * index / count), count)
        assert abs(last[-1] - 2) < 1e-12


def test_corrected_morlet_mean_and_complex_norm_under_affine_changes():
    prototype, _ = corrected_morlet(CONFIG["corrected_morlet"])
    grid = np.linspace(-12, 12, 48001)
    assert abs(np.trapezoid(prototype(grid), grid)) < 1e-10
    for scale, translation in [(0.25, 0), (0.5, -1.5), (1, 1.5)]:
        values = prototype((grid - translation) / scale) / np.sqrt(scale)
        np.testing.assert_allclose(np.trapezoid(abs(values) ** 2, grid), 1, atol=1e-12)
    assert np.trapezoid(prototype(grid).real ** 2, grid) < 0.51


@pytest.mark.parametrize("length,frames", [(64, 154), (100, 151), (256, 135)])
def test_stft_centers_and_amplitude_against_known_tone(length, frames):
    time = np.arange(1600) / 200
    # Portadora compleja no se usa; coseno exacto de 25 Hz, múltiples enteros de ciclos.
    samples = 2 * np.cos(2 * np.pi * 25 * time)
    frequencies, centers, coefficients, metadata = stft_illustration(samples, 200, length, 10, 512)
    assert len(centers) == frames
    assert centers[0] == length / 400
    assert centers[-1] == ((frames - 1) * 10 + length / 2) / 200
    peak = np.flatnonzero(frequencies == 25)[0]
    np.testing.assert_allclose(abs(coefficients[peak]), 1, atol=2e-13)
    assert metadata["last_covered_sample_s"] <= time[-1]


def test_cwt_frequency_kernel_guard_and_interior_tone():
    settings = CONFIG["cwt"]
    frequencies, scales, guards, kernels, metadata = effective_cwt_kernels(settings, 200)
    assert len(frequencies) == 96 and frequencies[0] == 4 and frequencies[-1] == 40
    np.testing.assert_allclose(frequencies * scales, 200, atol=1e-12)
    assert all(2 * guard > len(kernel) for guard, kernel in zip(guards, kernels, strict=True))
    assert max(item["near_nyquist_power_fraction"] for item in metadata["kernel_checks"]) < 1e-4
    time = np.arange(4096) / 200
    actual, _, _, coefficients, _ = cwt_illustration(np.cos(2 * np.pi * 12 * time), 200, settings)
    peak_frequency = actual[np.argmax(np.mean(abs(coefficients[:, 800:-800]) ** 2, axis=1))]
    assert abs(peak_frequency / 12 - 1) < 0.04
    impulse = np.zeros(4096)
    impulse[2048] = 1
    _, _, _, response, _ = cwt_illustration(impulse, 200, settings)
    for row, guard in zip(response, guards, strict=True):
        outside = abs(np.arange(len(row)) - 2048) > guard
        assert abs(row[outside]).max() < 1e-12


def test_cwt_rejects_detected_discretization_artifact():
    """Prueba que una precisión insuficiente dispara el control del kernel discreto."""
    settings = deepcopy(CONFIG["cwt"])
    settings["precision"] = 12
    with pytest.raises(RuntimeError, match="Nyquist"):
        effective_cwt_kernels(settings, 200)


def test_dwt_full_grid_reconstruction_and_shift_sensitivity():
    """Comprueba síntesis en la rejilla original y sensibilidad al desplazamiento."""
    _, samples, _, _ = shared_signal(CONFIG["shared_synthetic"])
    original, total, metadata = dwt_reconstructions(samples, 200, CONFIG["dwt"])
    np.testing.assert_allclose(total, samples, atol=1e-14)
    assert all(values.shape == samples.shape for values in original.values())
    assert metadata["nominal_bands_hz"]["D4"] == [6.25, 12.5]
    assert metadata["maximum_allowed_level"] == 7
    assert metadata["discrete_squared_norm_relative_error"] < 1e-12
    assert samples[0] == samples[-1] == 0
    shifted, _, _ = dwt_reconstructions(np.roll(samples, 1), 200, CONFIG["dwt"])
    difference = np.roll(shifted["D4"], -1) - original["D4"]
    assert np.linalg.norm(difference[211:-211]) > 1e-5
    invalid = {**CONFIG["dwt"], "level": 8}
    with pytest.raises(ValueError, match="Nivel DWT"):
        dwt_reconstructions(samples, 200, invalid)


def test_visual_bins_preserve_positive_and_negative_impulses():
    samples = np.zeros(101)
    samples[[8, 9, 55, 100]] = [9, -12, 20, -30]
    _, low, high, edges = minmax_bins(samples, 10)
    assert low.min() == -30 and high.max() == 20
    assert edges[0] == 0 and edges[-1] == len(samples)
    assert low[0] == -12 and high[0] == 9
