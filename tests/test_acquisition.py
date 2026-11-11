"""Ejercita adquisición y calidad mediante inventarios y trazas fabricados.

Las sesiones falsas controlan estados HTTP, caché y límites sin usar proveedores externos. Las
muestras y metadatos prueban procedencia, sincronía y recorte.
"""

import copy
import json
from io import BytesIO

import numpy as np
import pytest
import requests
from obspy import Stream, Trace, UTCDateTime, read, read_inventory
from obspy.core.event import Arrival, Catalog, Event, Magnitude, Origin, Pick, WaveformStreamID
from obspy.core.inventory import Channel, Inventory, Network, Response, Station

from seismic_wavelets.acquisition import (
    DEFAULT_LIMITS,
    DEFAULT_QUALITY,
    inventory_channels,
    load_config,
    run_acquisition,
)
from seismic_wavelets.catalog import load_catalog, normalize_event, utc_time
from seismic_wavelets.data import fetch_resource, new_budget, sha256_file
from seismic_wavelets.preprocessing import trim_requested_copy
from seismic_wavelets.quality import inspect_stream
from seismic_wavelets.references import confirm_arrival_coverage, screen_arrival_references


def test_trim_copies_record_margins_on_common_grid_without_changing_original():
    stream = read(BytesIO(waveform_payload()), format="MSEED")
    start = stream[0].stats.starttime
    for index, trace in enumerate(stream):
        trace.stats.starttime = start - (index + 1) / 100
        trace.data = np.arange(110 + index, dtype=np.int32)
    copies = stream.copy()
    trimmed = trim_requested_copy(stream, start, start + 1)
    assert (
        len({(str(tr.stats.starttime), str(tr.stats.endtime), tr.stats.npts) for tr in trimmed})
        == 1
    )
    for original, unchanged in zip(stream, copies, strict=True):
        assert original.stats == unchanged.stats
        np.testing.assert_array_equal(original.data, unchanged.data)
    trimmed[0].data[0] = -123
    assert stream[0].data[1] != -123


def test_trim_refuses_real_grid_shift_and_missing_coverage():
    stream = read(BytesIO(waveform_payload()), format="MSEED")
    start = stream[0].stats.starttime
    stream[1].stats.starttime += 0.005
    with pytest.raises(ValueError, match="Rejillas"):
        trim_requested_copy(stream, start, start + 1)
    with pytest.raises(ValueError):
        trim_requested_copy(stream, start - 1, start + 1)


def inventory_payload(station="TEST", response=True):
    """Construye un StationXML sintético para controlar selección y unidades."""
    channels = []
    for component, azimuth, dip in [("E", 90, 0), ("N", 0, 0), ("Z", 0, -90)]:
        instrument = (
            Response.from_paz(
                zeros=[0j],
                poles=[-1 + 1j, -1 - 1j],
                stage_gain=1,
                input_units="M/S",
                output_units="COUNTS",
            )
            if response
            else None
        )
        channels.append(
            Channel(
                code=f"HH{component}",
                location_code="00",
                latitude=5,
                longitude=-74,
                elevation=0,
                depth=0,
                azimuth=azimuth,
                dip=dip,
                sample_rate=100,
                start_date=UTCDateTime("2020-01-01"),
                response=instrument,
            )
        )
    inventory = Inventory(
        [
            Network(
                "XX",
                stations=[
                    Station(
                        code=station,
                        latitude=5,
                        longitude=-74,
                        elevation=0,
                        channels=channels,
                    )
                ],
            )
        ],
        source="SYNTHETIC_TEST_ONLY",
    )
    output = BytesIO()
    inventory.write(output, format="STATIONXML")
    return output.getvalue()


def waveform_payload(station="TEST", start="2026-08-10T12:33:27Z"):
    """Serializa trazas sintéticas en miniSEED para ejercitar el flujo completo."""
    data = np.round(1000 * np.sin(np.arange(101) / 7)).astype(np.int32)
    traces = [
        Trace(
            data.copy(),
            header={
                "network": "XX",
                "station": station,
                "location": "00",
                "channel": f"HH{component}",
                "starttime": UTCDateTime(start),
                "sampling_rate": 100,
            },
        )
        for component in "ENZ"
    ]
    output = BytesIO()
    Stream(traces).write(output, format="MSEED")
    return output.getvalue()


class FakeResponse:
    """Emula la parte de la respuesta HTTP que usa el cliente de adquisición."""

    def __init__(self, payload, status=200, url="https://example.test/query"):
        self.payload = payload
        self.status_code = status
        self.url = url
        self.headers = {"Content-Type": "application/octet-stream"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        yield self.payload


class FakeSession:
    """Entrega respuestas controladas por prueba sin abrir conexiones de red."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, **kwargs):
        assert kwargs["verify"] is True
        self.calls.append((url, kwargs["params"]))
        return self.handler(url, kwargs["params"])

    def close(self):
        pass


def config_fixture(root):
    """Construye una configuración de prueba para la adquisición simulada."""
    config = {
        "provider": {
            "station_url": "https://example.test",
            "waveform_url": "https://example.test",
            "event_url": "https://example.test",
            "allow_http": False,
        },
        "catalog": {"mode": "none"},
        "selection": {"network": "XX", "stations": ["TEST"], "location": "00", "channel": "HH*"},
        "technical_windows": [
            {
                "window_id": "SYNTHETIC_TEST_ONLY",
                "starttime_utc": "2026-08-10T12:33:27Z",
                "endtime_utc": "2026-08-10T12:33:28Z",
                "source": "generated_test_fixture",
            }
        ],
        "limits": {"retries": 0, "pause_between_batches_seconds": 0, "max_figures": 0},
    }
    path = root / "config.json"
    path.write_text(json.dumps(config))
    return config, path


def response_handler(url, params):
    """Responde con artefactos de prueba según la consulta recibida."""
    if "/station/" in url:
        return FakeResponse(inventory_payload(params["station"]), url=url)
    return FakeResponse(waveform_payload(params["station"], params["starttime"]), url=url)


def test_cache_repeated_reads_and_originals_unchanged(tmp_path, monkeypatch):
    _, config_path = config_fixture(tmp_path)
    session = FakeSession(response_handler)
    monkeypatch.setattr(requests, "Session", lambda: session)
    first = run_acquisition(tmp_path, config_path, download=True, make_figures=False)
    assert first["all_requested_cases_successful"]
    assert first["technical_windows"] == 1
    assert first["distinct_catalog_event_count"] == 0
    assert first["acquired_trace_count"] == 3
    second = run_acquisition(tmp_path, config_path, download=True, make_figures=False)
    assert len(session.calls) == 2  # StationXML + miniSEED, sin red en la repetición.
    assert second["requests_and_volume"]["requests"] == 0
    assert second["waveform_cache_hits"] == 1
    manifest = json.loads((tmp_path / second["manifest_path"]).read_text(encoding="utf-8"))
    case = manifest[0]
    path = tmp_path / case["waveform_request"]["path"]
    digest = sha256_file(path)
    original = read(path, format="MSEED")
    again = read(path, format="MSEED")
    for left, right in zip(original, again, strict=True):
        assert left.stats == right.stats
        np.testing.assert_array_equal(left.data, right.data)
    modified = original.copy().detrend("demean")
    assert not np.array_equal(original[0].data, modified[0].data)
    assert sha256_file(path) == digest
    np.testing.assert_array_equal(read(path)[0].data, original[0].data)
    quality = json.loads((tmp_path / case["quality_path"]).read_text(encoding="utf-8"))
    assert quality["original_sha256_before_and_after"] == digest
    assert quality["repeated_read_metadata_equal"]


def test_one_failed_request_never_marks_batch_success(tmp_path, monkeypatch):
    config, path = config_fixture(tmp_path)
    config["selection"]["stations"].append("FAIL")
    path.write_text(json.dumps(config))

    def handler(url, params):
        if "/dataselect/" in url and params["station"] == "FAIL":
            return FakeResponse(b"", status=204, url=url)
        return response_handler(url, params)

    monkeypatch.setattr(requests, "Session", lambda: FakeSession(handler))
    result = run_acquisition(tmp_path, path, download=True, make_figures=False)
    assert result["status"] == "partial_or_blocked"
    assert result["acquired_station_cases"] == 1
    assert result["failed_or_deferred_case_count"] == 1
    assert not result["all_requested_cases_successful"]
    assert result["missing_files_or_requests"][0]["status"] == "no_data"


def test_invalid_cache_is_not_overwritten(tmp_path):
    session = FakeSession(lambda url, params: FakeResponse(b"original", url=url))
    first = fetch_resource(
        tmp_path,
        "https://example.test/query",
        {},
        "waveform",
        new_budget(DEFAULT_LIMITS),
        session=session,
    )
    path = tmp_path / first["path"]
    path.write_bytes(b"altered_for_test")
    second = fetch_resource(
        tmp_path,
        "https://example.test/query",
        {},
        "waveform",
        new_budget(DEFAULT_LIMITS),
        session=session,
    )
    assert second["status"] == "cache_invalid"
    assert len(session.calls) == 1
    assert path.read_bytes() == b"altered_for_test"


@pytest.mark.parametrize(
    "code,expected", [(204, "no_data"), (404, "http_not_found"), (400, "bad_request")]
)
def test_http_errors_remain_distinct(tmp_path, code, expected):
    session = FakeSession(lambda url, params: FakeResponse(b"error", code, url))
    result = fetch_resource(
        tmp_path,
        "https://example.test/query",
        {},
        "diagnostic",
        new_budget(DEFAULT_LIMITS),
        session=session,
    )
    assert result["status"] == expected
    assert len(session.calls) == 1


def test_tls_failure_is_not_bypassed_or_retried(tmp_path):
    def handler(url, params):
        raise requests.exceptions.SSLError("expired_test_certificate")

    session = FakeSession(handler)
    result = fetch_resource(
        tmp_path,
        "https://example.test/query",
        {},
        "diagnostic",
        new_budget(DEFAULT_LIMITS),
        session=session,
    )
    assert result["status"] == "tls_error"
    assert len(session.calls) == 1


def test_volume_limit_does_not_create_original(tmp_path):
    limits = {**DEFAULT_LIMITS, "max_request_bytes": 10}
    session = FakeSession(lambda url, params: FakeResponse(b"x" * 11, url=url))
    result = fetch_resource(
        tmp_path,
        "https://example.test/query",
        {},
        "waveform",
        new_budget(limits),
        session=session,
    )
    assert result["status"] == "volume_limit"
    assert not (tmp_path / result["path"]).exists()


def test_retries_are_bounded_and_can_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("seismic_wavelets.data.time.sleep", lambda value: None)
    replies = iter([500, 200])
    session = FakeSession(lambda url, params: FakeResponse(b"result", next(replies), url))
    result = fetch_resource(
        tmp_path,
        "https://example.test/query",
        {},
        "diagnostic",
        new_budget(DEFAULT_LIMITS),
        session=session,
    )
    assert result["status"] == "downloaded"
    assert len(result["attempts"]) == 2


def test_quality_marks_unknown_units_gaps_nonfinite_and_desynchronization():
    stream = read(BytesIO(waveform_payload()), format="MSEED")
    metadata = inventory_channels(
        read_inventory(BytesIO(inventory_payload(response=False))),
        "2026-08-10T12:33:27Z",
        "2026-08-10T12:33:28Z",
        "2026-08-10T12:33:27Z",
    )
    stream[0].data = stream[0].data.astype(float)
    stream[0].data[20] = np.nan
    stream[1].stats.starttime += 0.02
    extra = stream[2].copy()
    extra.stats.starttime += 2
    stream += extra
    quality = inspect_stream(
        stream,
        metadata,
        "2026-08-10T12:33:27Z",
        "2026-08-10T12:33:28Z",
        DEFAULT_QUALITY,
    )
    assert quality["status"] == "rejected_technical"
    assert "components_not_synchronized" in quality["reject_reasons"]
    assert "gaps_or_overlaps" in quality["reject_reasons"]
    assert any("nonfinite" in item for item in quality["reject_reasons"])
    assert all(item["raw_units"] == "unknown" for item in quality["traces"])
    assert any("raw_units_unknown" in item for item in quality["warnings"])


def test_multiple_events_and_stations_keep_event_groups(tmp_path, monkeypatch):
    config, path = config_fixture(tmp_path)
    config["technical_windows"] = []
    config["selection"]["stations"] = ["TEST", "OTHER"]
    config["selection"]["sort_by_distance"] = True
    config["catalog"] = {
        "mode": "explicit",
        "events": [
            {
                "event_id": f"SYNTHETIC_TEST_EVENT_{index}",
                "source": "generated_test_fixture",
                "origin_time_utc": f"2026-08-{10 + index}T12:33:27Z",
                "latitude": 6,
                "longitude": -75,
                "depth_km": 20,
                "magnitude": 3,
            }
            for index in range(2)
        ],
    }
    config["event_window"] = {"seconds_before_origin": 0, "seconds_after_origin": 1}
    config["event_roles"] = {
        "SYNTHETIC_TEST_EVENT_0": "pilot",
        "SYNTHETIC_TEST_EVENT_1": "evaluation_reserved",
    }
    path.write_text(json.dumps(config))
    monkeypatch.setattr(requests, "Session", lambda: FakeSession(response_handler))
    summary = run_acquisition(tmp_path, path, download=True, make_figures=False)
    assert summary["all_requested_cases_successful"]
    assert summary["candidate_events"] == 2
    assert summary["acquired_station_cases"] == 4
    assert summary["distinct_catalog_event_count"] == 2
    manifest = json.loads((tmp_path / summary["manifest_path"]).read_text(encoding="utf-8"))
    for event_id in summary["acquired_event_ids"]:
        roles = {case["partition"] for case in manifest if case["event_id"] == event_id}
        assert len(roles) == 1
    assert all(
        case["distance_km"] > 0 and case["arrival_reference"] == "absent" for case in manifest
    )


def test_csv_catalog_preserved_and_missing_values_do_not_pass_filters(tmp_path):
    config, path = config_fixture(tmp_path)
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "event_id,origin_time_utc,magnitude,latitude,longitude\n"
        "TEST_ONLY_1,2026-08-10T12:33:27Z,3,5,-74\n"
        "TEST_ONLY_2,2026-08-11T12:33:27Z,,6,-73\n"
    )
    config["catalog"] = {
        "mode": "csv",
        "path": "catalog.csv",
        "source": "generated_test_fixture",
        "query": {"minmagnitude": 2},
    }
    config["event_window"] = {"seconds_before_origin": 0, "seconds_after_origin": 1}
    path.write_text(json.dumps(config))
    config = load_config(tmp_path, path)
    events, rejected, status = load_catalog(tmp_path, config, new_budget(DEFAULT_LIMITS))
    assert len(events) == 1
    assert rejected[0]["event_id"] == "TEST_ONLY_2"
    assert sha256_file(tmp_path / status["artifact"]["path"]) == sha256_file(csv_path)


def test_config_rejects_wildcards_unjustified_http_and_ambiguous_time(tmp_path):
    original, path = config_fixture(tmp_path)
    for replacement in [
        {"selection": {**original["selection"], "stations": ["*"]}},
        {"provider": {**original["provider"], "allow_http": True}},
    ]:
        bad = copy.deepcopy(original)
        bad.update(replacement)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_config(tmp_path, path)
    with pytest.raises(ValueError):
        utc_time("2026-08-10T12:33:27")
    with pytest.raises(ValueError):
        normalize_event({"origin_time_utc": "2026-08-10T12:33:27Z", "source": "fixture"})


def test_quakeml_retains_reported_pick_and_depth_units(tmp_path):
    config, path = config_fixture(tmp_path)
    origin = Origin(
        time=UTCDateTime("2026-08-10T12:33:27Z"), latitude=5, longitude=-74, depth=25000
    )
    pick = Pick(
        time=UTCDateTime("2026-08-10T12:33:27.5Z"),
        phase_hint="P",
        evaluation_mode="manual",
        waveform_id=WaveformStreamID(
            network_code="XX", station_code="TEST", location_code="00", channel_code="HHZ"
        ),
    )
    event = Event(
        resource_id="smi:local/SYNTHETIC_TEST_ONLY",
        origins=[origin],
        magnitudes=[Magnitude(mag=3.2, magnitude_type="ML")],
        picks=[pick],
    )
    Catalog([event]).write(tmp_path / "catalog.xml", format="QUAKEML")
    config["catalog"] = {
        "mode": "quakeml",
        "path": "catalog.xml",
        "source": "generated_test_fixture",
    }
    config["event_window"] = {"seconds_before_origin": 0, "seconds_after_origin": 1}
    path.write_text(json.dumps(config))
    rows, rejected, status = load_catalog(
        tmp_path, load_config(tmp_path, path), new_budget(DEFAULT_LIMITS)
    )
    assert not rejected
    assert rows[0]["depth_km"] == 25
    assert rows[0]["picks"][0]["seed_id"] == "XX.TEST.00.HHZ"
    assert rows[0]["picks"][0]["reference_type"] == "catalog_reported_pick"
    assert sha256_file(tmp_path / status["artifact"]["path"]) == sha256_file(
        tmp_path / "catalog.xml"
    )


def test_quakeml_only_confirms_selected_origin_picks_on_the_effective_trace(tmp_path, monkeypatch):
    config, path = config_fixture(tmp_path)
    config["technical_windows"] = []
    config["event_window"] = {"seconds_before_origin": 0, "seconds_after_origin": 1}
    config["catalog"] = {
        "mode": "quakeml",
        "path": "catalog.xml",
        "source": "generated_test_fixture",
    }
    path.write_text(json.dumps(config))
    start = UTCDateTime("2026-08-10T12:33:27Z")
    picks = []
    for name, channel, offset in [
        ("linked", "HHZ", 0.4),
        ("other_origin", "HHZ", 0.5),
        ("unsupported_phase", "HHZ", 0.6),
        ("other_channel", "BHX", 0.6),
        ("outside_window", "HHZ", 1.1),
        ("ambiguous", "HHZ", 0.8),
    ]:
        picks.append(
            Pick(
                resource_id=f"smi:local/{name}",
                time=start + offset,
                phase_hint="S",
                waveform_id=WaveformStreamID(
                    network_code="XX", station_code="TEST", location_code="00", channel_code=channel
                ),
            )
        )
    selected = Origin(
        resource_id="smi:local/selected_origin",
        time=start,
        latitude=5,
        longitude=-74,
        arrivals=[
            Arrival(pick_id=pick.resource_id, phase=phase)
            for pick, phase in [
                (picks[0], "P"),
                (picks[2], "Pg"),
                (picks[3], "P"),
                (picks[4], "P"),
                (picks[5], "P"),
                (picks[5], "S"),
            ]
        ],
    )
    other = Origin(
        time=start,
        latitude=5,
        longitude=-74,
        arrivals=[Arrival(pick_id=picks[1].resource_id, phase="S")],
    )
    event = Event(origins=[other, selected], picks=picks, preferred_origin_id=selected.resource_id)
    Catalog([event]).write(tmp_path / "catalog.xml", format="QUAKEML")
    monkeypatch.setattr(requests, "Session", lambda: FakeSession(response_handler))

    planning = run_acquisition(tmp_path, path, download=False, make_figures=False)
    (planned,) = json.loads((tmp_path / planning["manifest_path"]).read_text(encoding="utf-8"))
    assert planned["arrival_reference"] == "linked_candidates_pending_waveform"
    assert planned["arrival_references"] == []
    (candidate,) = planned["arrival_candidates"]
    assert candidate["pick_id"] == str(picks[0].resource_id)
    assert candidate["origin_id"] == str(selected.resource_id)
    assert candidate["phase"] == "P" and candidate["phase_hint"] == "S"
    assert {entry["reason"] for entry in planned["arrival_reference_exclusions"]} == {
        "not_linked_to_selected_origin",
        "unsupported_arrival_phase",
        "channel_mismatch",
        "outside_requested_window",
        "ambiguous_origin_link",
    }
    assert len(planned["event"]["picks"]) == 6

    downloaded = run_acquisition(tmp_path, path, download=True, make_figures=False)
    (case,) = json.loads((tmp_path / downloaded["manifest_path"]).read_text(encoding="utf-8"))
    (reference,) = case["arrival_references"]
    assert reference["pick_id"] == candidate["pick_id"]
    assert reference["sampling_rate_hz"] == 100
    assert case["arrival_reference"] == "reported_references_covered"

    expired = copy.deepcopy(planned)
    for channel in expired["channels"]:
        if channel["seed_id"] == candidate["seed_id"]:
            channel["station_epoch_end_utc"] = str(start + 0.3)
    screened = screen_arrival_references(expired)
    assert not screened["arrival_candidates"]
    assert any(
        entry["reason"] == "outside_metadata_epoch" and entry["pick_id"] == candidate["pick_id"]
        for entry in screened["arrival_reference_exclusions"]
    )
    rejected = confirm_arrival_coverage(planned, {"status": "rejected_technical"})
    assert rejected["arrival_reference"] == "no_covered_references"
    assert rejected["arrival_reference_exclusions"][-1]["reason"] == "waveform_rejected_technical"

    # Una ventana efectiva que empieza después del pick no autoriza usarlo.
    quality = {
        "status": "review_candidate",
        "quality_scope": "requested_window_copy",
        "traces": [
            {
                "seed_id": "XX.TEST.00.HHZ",
                "starttime_utc": str(start),
                "endtime_utc": str(start + 1),
            }
        ],
        "requested_window": {
            "status": "review_candidate",
            "traces": [
                {
                    "seed_id": "XX.TEST.00.HHZ",
                    "starttime_utc": str(start + 0.5),
                    "endtime_utc": str(start + 1),
                }
            ],
        },
    }
    result = confirm_arrival_coverage(planned, quality)
    assert result["arrival_references"] == []
    assert result["arrival_reference_exclusions"][-1]["reason"] == "outside_effective_trace"


def test_request_limit_and_pending_cases_are_explicit(tmp_path, monkeypatch):
    config, path = config_fixture(tmp_path)
    config["selection"]["stations"] = ["TEST", "OTHER"]
    config["limits"]["max_cases"] = 1
    path.write_text(json.dumps(config))
    session = FakeSession(response_handler)
    monkeypatch.setattr(requests, "Session", lambda: session)
    summary = run_acquisition(tmp_path, path, download=True, make_figures=False)
    assert summary["states"]["deferred_case_limit"] == 1
    assert summary["status"] == "partial_or_blocked"
    budget = new_budget({**DEFAULT_LIMITS, "max_requests": 1})
    session = FakeSession(lambda url, params: FakeResponse(b"1.0", url=url))
    fetch_resource(tmp_path, "https://example.test/one", {}, "diagnostic", budget, session=session)
    result = fetch_resource(
        tmp_path, "https://example.test/two", {}, "diagnostic", budget, session=session
    )
    assert result["status"] == "request_limit"
    assert budget["requests"] == 1


def test_orientation_components_12_are_not_rotated_or_relabelled():
    inventory = read_inventory(BytesIO(inventory_payload()))
    for channel in inventory[0][0].channels:
        if channel.code.endswith("N"):
            channel.code = "HH1"
            channel.azimuth = 35
        elif channel.code.endswith("E"):
            channel.code = "HH2"
            channel.azimuth = 125
    metadata = inventory_channels(
        inventory,
        "2026-08-10T12:33:27Z",
        "2026-08-10T12:33:28Z",
        "2026-08-10T12:33:27Z",
    )
    assert {item["channel"] for item in metadata} == {"HH1", "HH2", "HHZ"}
    assert next(item["azimuth_deg"] for item in metadata if item["channel"] == "HH1") == 35


def test_unknown_or_nonfinite_metadata_cannot_be_used_as_known_event():
    with pytest.raises(ValueError, match="no finito"):
        normalize_event(
            {
                "event_id": "TEST_ONLY",
                "source": "generated_test_fixture",
                "origin_time_utc": "2026-08-10T12:33:27Z",
                "magnitude": float("nan"),
            }
        )
