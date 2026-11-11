"""Invariantes de la interfaz causal, con un indicador de prueba en vez de un método DSP."""

import numpy as np
import pytest

from seismic_wavelets.detection import CausalWindowDetector, Trigger


def test_detector_has_same_triggers_with_different_blocks_and_future_suffixes():
    samples = [0, 1, 2, 1, 0, 2, 3, 0]

    def detector():
        return CausalWindowDetector(window_samples=2, threshold=3, score=np.sum)

    complete = detector().process(samples, start_sample=100)
    split = detector()
    chunks = (
        split.process(samples[:1], start_sample=100)
        + split.process(samples[1:5], start_sample=101)
        + split.process(samples[5:], start_sample=105)
    )
    assert complete == chunks == [Trigger(102, 3), Trigger(106, 5)]
    assert detector().process(samples[:5] + [1000], start_sample=100)[:1] == [Trigger(102, 3)]


def test_detector_restarts_warmup_and_threshold_after_gap():
    detector = CausalWindowDetector(window_samples=2, threshold=2, score=np.sum)
    assert detector.process([2, 2], start_sample=0) == [Trigger(1, 4)]
    assert detector.process([2], start_sample=4) == []
    assert detector.process([2], start_sample=5) == [Trigger(5, 4)]
    with pytest.raises(ValueError, match="solapado"):
        detector.process([2], start_sample=5)
    assert detector.process([2], start_sample=6) == []


def test_invalid_block_or_score_does_not_advance_detector():
    def score(window):
        return float("nan") if window[-1] == 99 else float(window.sum())

    detector = CausalWindowDetector(window_samples=2, threshold=3, score=score)
    assert detector.process([1], start_sample=0) == []
    with pytest.raises(ValueError, match="no finitas"):
        detector.process([float("nan")], start_sample=1)
    with pytest.raises(ValueError, match="no finito"):
        detector.process([2, 99], start_sample=1)
    assert detector.process([2], start_sample=1) == [Trigger(1, 3)]
