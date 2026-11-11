"""Comprueba disponibilidad causal y conservación de potencia del piloto."""

import numpy as np
import pytest
from scipy.signal import get_window

from seismic_wavelets.metrics import associate_p, select_triggers
from seismic_wavelets.transforms import fourier_power, frame_ratios


@pytest.mark.parametrize("method", ["STA/LTA", "STFT", "DWT"])
def test_frame_ratios_have_no_lookahead(method):
    samples = np.random.default_rng(80).normal(size=2400)
    # Comparar contra el prefijo que termina dentro de una ventana del lote completo.
    ends, ratios = frame_ratios(samples, 100, 64, 5, 100, (3.125, 25), method)
    short_ends, short_ratios = frame_ratios(samples[:1172], 100, 64, 5, 100, (3.125, 25), method)
    np.testing.assert_array_equal(short_ends, ends[ends < 1172])
    np.testing.assert_allclose(short_ratios, ratios[ends < 1172], rtol=1e-12)


def test_background_ends_before_recent_window():
    samples = np.ones(200)
    samples[65:] = 3
    ends, ratios = frame_ratios(samples, 100, 64, 5, 1, (3.125, 25), "STA/LTA")
    assert ends[0] == 128
    assert ratios[0] == pytest.approx(9)


@pytest.mark.parametrize("method", ["STA/LTA", "STFT", "DWT"])
def test_horizontal_power_is_summed_before_taking_the_ratio(method):
    rng = np.random.default_rng(2026)
    north, east = rng.normal(size=(2, 130))
    # Los dos fondos tienen potencias distintas y los cambios se oponen.
    # El resultado es una media ponderada, no la media de razones por canal.
    east *= 4
    samples = np.stack([north, east])
    samples[:, 65:] *= np.array([[3], [0.5]])
    ends, ratios = frame_ratios(samples, 100, 64, 5, 1, (3.125, 25), method)
    # La rotación ortonormal mezcla los canales y conserva su potencia total.
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    rotated_ends, rotated = frame_ratios(rotation @ samples, 100, 64, 5, 1, (3.125, 25), method)
    np.testing.assert_array_equal(rotated_ends, ends)
    np.testing.assert_allclose(rotated, ratios, rtol=1e-12)
    individual = [
        frame_ratios(channel, 100, 64, 5, 1, (3.125, 25), method)[1] for channel in samples
    ]
    assert not np.allclose(ratios, np.mean(individual, axis=0))


@pytest.mark.parametrize("nfft", [64, 512])
def test_fourier_frame_power_preserves_tapered_mean_square(nfft):
    samples = np.random.default_rng(99).normal(size=(3, 64))
    _, powers = fourier_power(samples, 100, nfft=nfft)
    taper = get_window("hann", 64, fftbins=True)
    expected = np.sum((samples * taper) ** 2, axis=1) / np.sum(taper**2)
    np.testing.assert_allclose(powers.sum(axis=1), expected)


def test_first_p_association_uses_emission_time_and_excludes_s():
    ends = np.array([90, 95, 100, 105, 110, 115])
    triggers = select_triggers(np.array([0, 3, 0, 3, 0, 3]), ends, 2, 11)
    assert triggers == [95, 115]
    assert associate_p(triggers, 100, 1, 0.1, 1.5, 1.2)["available_sample"] == 95
    assert associate_p(triggers, 100, 1, 0.04, 1.5, 1.15) is None
