"""Check the attributed review fixture and its causal trigger times."""

import json
from pathlib import Path

import pytest
from obspy import UTCDateTime

from seismic_wavelets.pilot import detect, match_p, prepare
from seismic_wavelets.waveforms import load_trace


@pytest.mark.parametrize(("method", "sample"), [("STA/LTA", 5013), ("STFT", 5023), ("DWT", 5008)])
def test_bundled_mugello_case(method, sample):
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / "data/mugello/case.json").read_text(encoding="utf-8"))
    trace = load_trace(root, case)
    assert (trace.id, trace.stats.npts, trace.stats.sampling_rate) == ("IV.FNVD..HHZ", 9000, 100)
    picks = {
        ref["phase"]: UTCDateTime(ref["pick_time_utc"]) - trace.stats.starttime
        for ref in case["arrival_references"]
    }
    triggers, _, _ = detect(prepare(trace.data), method, case["thresholds"][method])
    match = match_p(triggers, picks["P"], picks["S"])
    assert match is not None
    assert match["available_sample"] == sample
