"""Planificación y adquisición por casos; el procesamiento derivado queda separado."""

import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read, read_inventory
from obspy.geodetics import gps2dist_azimuth

from seismic_wavelets.catalog import load_catalog, utc_time
from seismic_wavelets.data import (
    fetch_resource,
    new_budget,
    project_path,
    sha256_file,
    utc_now,
    validate_url,
    write_json,
)
from seismic_wavelets.preprocessing import trim_requested_copy
from seismic_wavelets.quality import inspect_stream
from seismic_wavelets.references import confirm_arrival_coverage, screen_arrival_references

DEFAULT_LIMITS = {
    "max_events": 10,
    "max_windows": 4,
    "max_stations": 2,
    "max_cases": 8,
    "max_requests": 32,
    "max_request_bytes": 16 * 1024 * 1024,
    "max_total_bytes": 64 * 1024 * 1024,
    "max_samples_per_case": 1_000_000,
    "max_duration_seconds": 600,
    "timeout_seconds": 20,
    "retries": 1,
    "batch_size": 2,
    "pause_between_batches_seconds": 0.2,
    "max_figures": 3,
}
DEFAULT_QUALITY = {
    "require_response": False,
    "coverage_tolerance_samples": 1.1,
    "synchronization_tolerance_samples": 0.1,
    "extreme_fraction_warning": 0.01,
    "step_warning_factor": 100.0,
}


def load_config(root, path):
    """Carga y valida el contrato de una corrida de adquisición.

    Valida proveedores, límites y selección antes de acceder a la red. La configuración
    distingue ventanas técnicas de consultas de catálogo y exige justificar el transporte HTTP
    si se habilita.
    """
    config = json.loads(project_path(root, path).read_text(encoding="utf-8"))
    provider = config["provider"]
    provider.setdefault("allow_http", False)
    if provider["allow_http"] and not provider.get("transport_justification"):
        raise ValueError("HTTP requiere justificación documental explícita.")
    for name in ("station_url", "waveform_url", "event_url"):
        if name in provider:
            validate_url(provider[name], provider["allow_http"])
    config["limits"] = {**DEFAULT_LIMITS, **config.get("limits", {})}
    limits = config["limits"]
    for name, value in limits.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Límite no válido: {name}")
        if name not in ("pause_between_batches_seconds", "retries", "max_figures") and value == 0:
            raise ValueError(f"Límite debe ser positivo: {name}")
        if name not in ("timeout_seconds", "pause_between_batches_seconds", "max_duration_seconds"):
            if not isinstance(value, int):
                raise ValueError(f"Límite debe ser entero: {name}")
    stations = config["selection"]["stations"]
    if not stations or len(stations) != len(set(stations)):
        raise ValueError("Lista explícita de estaciones única y no vacía.")
    if any(any(character in station for character in "*?,") for station in stations):
        raise ValueError("No usar comodines de estación para el piloto ni los lotes.")
    if any(character in config["selection"]["network"] for character in "*?,"):
        raise ValueError("La red debe ser explícita.")
    config.setdefault("technical_windows", [])
    for window in config["technical_windows"]:
        if not window.get("window_id") or not window.get("source"):
            raise ValueError("Ventana técnica requiere identificador local y fuente.")
        if utc_time(window["endtime_utc"]) <= utc_time(window["starttime_utc"]):
            raise ValueError("Fin de ventana debe ser posterior al inicio.")
    for name in ("starttime", "endtime"):
        if name in config["catalog"].get("query", {}):
            utc_time(config["catalog"]["query"][name])
    config["quality"] = {**DEFAULT_QUALITY, **config.get("quality", {})}
    for name, value in config["quality"].items():
        if name != "require_response" and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"Regla provisional de calidad no válida: {name}")
    if config["catalog"]["mode"] != "none":
        if config["event_window"]["seconds_before_origin"] < 0:
            raise ValueError("Ventana anterior negativa.")
        if config["event_window"]["seconds_after_origin"] <= 0:
            raise ValueError("Ventana posterior debe ser positiva.")
    if config["selection"].get("sort_by_distance") and config["catalog"]["mode"] == "none":
        raise ValueError("Distancia a evento requiere catálogo con coordenadas.")
    return config


def inventory_channels(inventory, start, end, anchor):
    """Describir épocas y respuesta completa, sin transformar unidades del miniSEED."""
    start, end, anchor = UTCDateTime(start), UTCDateTime(end), UTCDateTime(anchor)
    records = []
    for network in inventory:
        for station in network:
            for channel in station:
                left = channel.start_date
                right = channel.end_date
                if (left and left > end) or (right and right < start):
                    continue
                response = channel.response
                sensitivity = None if response is None else response.instrument_sensitivity
                records.append(
                    {
                        "seed_id": ".".join(
                            [
                                network.code,
                                station.code,
                                channel.location_code,
                                channel.code,
                            ]
                        ),
                        "network": network.code,
                        "station": station.code,
                        "location": channel.location_code,
                        "channel": channel.code,
                        "instrument_family": channel.code[:2],
                        "latitude": channel.latitude,
                        "longitude": channel.longitude,
                        "elevation_m": channel.elevation,
                        "depth_m": channel.depth,
                        "azimuth_deg": channel.azimuth,
                        "dip_deg": channel.dip,
                        "sample_rate_hz": channel.sample_rate,
                        "sensor_description": None
                        if channel.sensor is None
                        else channel.sensor.description,
                        "epoch_start_utc": None if left is None else str(left),
                        "epoch_end_utc": None if right is None else str(right),
                        "station_epoch_start_utc": str(station.start_date)
                        if station.start_date
                        else None,
                        "station_epoch_end_utc": str(station.end_date)
                        if station.end_date
                        else None,
                        "active_at_reference_time": (left is None or left <= anchor)
                        and (right is None or right >= anchor),
                        "covers_requested_window": (left is None or left <= start)
                        and (right is None or right >= end)
                        and (station.start_date is None or station.start_date <= start)
                        and (station.end_date is None or station.end_date >= end),
                        "response_available": bool(
                            response and response.response_stages and sensitivity
                        ),
                        "response_stage_count": 0
                        if response is None
                        else len(response.response_stages),
                        "response_input_units": None
                        if sensitivity is None
                        else sensitivity.input_units,
                        "response_output_units": None
                        if sensitivity is None
                        else sensitivity.output_units,
                        "sensitivity": None if sensitivity is None else sensitivity.value,
                        "sensitivity_frequency_hz": None
                        if sensitivity is None
                        else sensitivity.frequency,
                        "waveform_availability": "unverified_until_dataselect",
                    }
                )
    return records


def _windows(config, events):
    """Resuelve las ventanas técnicas explícitas o las ventanas basadas en eventos.

    Las ventanas de evento se construyen alrededor del origen del catálogo. Una ventana
    técnica carece de asociación con eventos y de picks.
    """
    windows = []
    for event in events:
        origin = utc_time(event["origin_time_utc"])
        settings = config["event_window"]
        windows.append(
            {
                "window_id": event["event_id"],
                "event_id": event["event_id"],
                "event": event,
                "reference_time_utc": str(origin),
                "starttime_utc": str(origin - settings["seconds_before_origin"]),
                "endtime_utc": str(origin + settings["seconds_after_origin"]),
                "source": event["source"],
                "kind": "event_origin_window",
                "partition": config.get("event_roles", {}).get(
                    event["event_id"], "candidate_unassigned"
                ),
            }
        )
    for item in config["technical_windows"][: config["limits"]["max_windows"]]:
        windows.append(
            {
                **item,
                "event_id": None,
                "event": None,
                "reference_time_utc": item["starttime_utc"],
                "kind": "technical_window",
                "partition": "technical_only",
            }
        )
    return windows


def plan_cases(root, config, events, budget):
    """Construye casos por ventana y estación a partir de inventarios FDSN.

    Separa familias de canal y localizaciones físicas, registra inventario y cobertura y
    conserva las razones de descarte. El caso es la unidad de adquisición, no un identificador
    de sismo.
    """
    cases, station_records = [], []
    network = config["selection"]["network"]
    stations = config["selection"]["stations"][: config["limits"]["max_stations"]]
    for window in _windows(config, events):
        duration = UTCDateTime(window["endtime_utc"]) - UTCDateTime(window["starttime_utc"])
        for station in stations:
            base = {
                "window_id": window["window_id"],
                "event_id": window["event_id"],
                "event": window["event"],
                "kind": window["kind"],
                "partition": window["partition"],
                "source": window["source"],
                "network": network,
                "station": station,
                "starttime_utc": window["starttime_utc"],
                "endtime_utc": window["endtime_utc"],
                "reference_time_utc": window["reference_time_utc"],
                "arrival_reference": "absent",
                "distance_km": None,
                "selection_basis": "explicit_station_list_not_nearest_claim",
            }
            if duration > config["limits"]["max_duration_seconds"]:
                cases.append({**base, "status": "window_duration_limit", "channels": []})
                continue
            params = {
                "network": network,
                "station": station,
                "location": config["selection"].get("location", "*"),
                "channel": config["selection"].get("channel", "HH*"),
                "starttime": str(UTCDateTime(window["starttime_utc"])),
                "endtime": str(UTCDateTime(window["endtime_utc"])),
                "level": "response",
                "format": "xml",
            }
            response = fetch_resource(
                root,
                config["provider"]["station_url"].rstrip("/") + "/fdsnws/station/1/query",
                params,
                "station",
                budget,
                allow_http=config["provider"]["allow_http"],
            )
            base["inventory_requests"] = [response]
            if response["status"] not in ("downloaded", "cached"):
                if config.get("allow_channel_metadata_fallback", True):
                    fallback = fetch_resource(
                        root,
                        response["query"]["url"],
                        {**params, "level": "channel"},
                        "station",
                        budget,
                        allow_http=config["provider"]["allow_http"],
                    )
                    base["inventory_requests"].append(fallback)
                    response = fallback
                if response["status"] not in ("downloaded", "cached"):
                    cases.append({**base, "status": "inventory_failed", "channels": []})
                    continue
            base["inventory_path"] = response["path"]
            base["inventory_sha256"] = response["sha256"]
            try:
                inventory = read_inventory(
                    project_path(root, response["path"]), format="STATIONXML"
                )
                channels = inventory_channels(
                    inventory,
                    window["starttime_utc"],
                    window["endtime_utc"],
                    window["reference_time_utc"],
                )
            except Exception as error:
                cases.append(
                    {**base, "status": "invalid_inventory", "channels": [], "error": str(error)}
                )
                continue
            # Separar red/estación/localización/familia, sin confundir sensores ni 1/2 con N/E.
            groups = defaultdict(list)
            for item in channels:
                if item["network"] != network or item["station"] != station:
                    continue
                item = {
                    **item,
                    "window_id": window["window_id"],
                    "inventory_path": response["path"],
                }
                station_records.append(item)
                if item["active_at_reference_time"]:
                    groups[(item["location"], item["instrument_family"])].append(item)
            if not groups:
                cases.append({**base, "status": "no_active_channels", "channels": []})
            for (location, family), selected in sorted(groups.items()):
                selected.sort(key=lambda item: item["seed_id"])
                case = {
                    **base,
                    "location": location,
                    "instrument_family": family,
                    "channels": selected,
                }
                case["status"] = "planned"
                components = {item["channel"][-1] for item in selected}
                case["component_layout"] = (
                    "ZNE"
                    if {"Z", "N", "E"} <= components
                    else "Z12_orientation_required"
                    if {"Z", "1", "2"} <= components
                    else "incomplete_or_other"
                )
                case["rotation_applied"] = False
                if len({item["seed_id"] for item in selected}) != len(selected):
                    case["status"] = "ambiguous_channel_epochs"
                event = case["event"]
                if event:
                    case["picks"] = [
                        pick
                        for pick in event["picks"]
                        if any(pick.get("seed_id") == item["seed_id"] for item in selected)
                    ]
                    case.update(screen_arrival_references(case))
                    if case["arrival_candidates"]:
                        case["arrival_reference"] = "linked_candidates_pending_waveform"
                    elif case["picks"]:
                        case["arrival_reference"] = "reported_references_preserved_not_validated"
                    if event["latitude"] is not None and event["longitude"] is not None:
                        point = selected[0]
                        if point["latitude"] is not None and point["longitude"] is not None:
                            distance, azimuth, backazimuth = gps2dist_azimuth(
                                event["latitude"],
                                event["longitude"],
                                point["latitude"],
                                point["longitude"],
                            )
                            case.update(
                                distance_km=distance / 1000,
                                azimuth_deg=azimuth,
                                backazimuth_deg=backazimuth,
                            )
                maximum = config["selection"].get("max_distance_km")
                if maximum is not None and (
                    case["distance_km"] is None or case["distance_km"] > maximum
                ):
                    case["status"] = "distance_filter_missing_or_outside"
                cases.append(case)
    if config["selection"].get("sort_by_distance"):
        cases.sort(
            key=lambda item: (
                item["window_id"],
                item["distance_km"] if item["distance_km"] is not None else float("inf"),
            )
        )
        for case in cases:
            case["selection_basis"] = "distance_order_within_explicit_queried_station_list"
    for case in cases:
        identity = {
            name: case.get(name)
            for name in (
                "window_id",
                "event_id",
                "network",
                "station",
                "location",
                "instrument_family",
                "starttime_utc",
                "endtime_utc",
            )
        }
        identity["provider"] = config["provider"]["waveform_url"]
        case["case_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
            :24
        ]
    return cases, station_records


def inspect_case(root, case, rules, max_samples, prepare_requested_window=False):
    """Audita un miniSEED local y prepara, si se solicita, una copia recortada.

    Verifica el límite de muestras con las cabeceras y contrasta dos lecturas. Conserva los
    diagnósticos de original y copia, los tiempos efectivos y los hashes. El recorte conserva
    las muestras de la rejilla original.
    """
    path = project_path(root, case["waveform_request"]["path"])
    before = sha256_file(path)
    # Leer primero solo las cabeceras para aplicar el límite antes de materializar las muestras.
    head = read(path, format="MSEED", headonly=True)
    if sum(trace.stats.npts for trace in head) > max_samples:
        return {"status": "rejected_technical", "reject_reasons": ["sample_limit"], "warnings": []}
    stream = read(path, format="MSEED")
    repeated = read(path, format="MSEED")

    def signature(st):
        """Extrae identificador, extremos UTC, frecuencia y longitud de cada traza."""
        return [
            (
                tr.id,
                str(tr.stats.starttime),
                str(tr.stats.endtime),
                tr.stats.sampling_rate,
                tr.stats.npts,
            )
            for tr in st
        ]

    if signature(stream) != signature(repeated):
        raise RuntimeError("Lectura repetida cambió metadatos.")
    for original, reread in zip(stream, repeated, strict=True):
        np.testing.assert_array_equal(original.data, reread.data)
    raw_quality = inspect_stream(
        stream.copy(),
        case["channels"],
        case["starttime_utc"],
        case["endtime_utc"],
        rules,
    )
    quality = {
        **raw_quality,
        "raw": raw_quality,
        "quality_scope": "original_records",
        "requested_window": None,
    }
    if prepare_requested_window:
        try:
            trimmed = trim_requested_copy(
                stream,
                case["starttime_utc"],
                case["endtime_utc"],
                rules["synchronization_tolerance_samples"],
            )
            # Evaluar la copia por separado: los márgenes del original y la ventana efectiva
            # tienen coberturas diferentes.
            requested_quality = inspect_stream(
                trimmed,
                case["channels"],
                case["starttime_utc"],
                case["endtime_utc"],
                rules,
            )
            output = BytesIO()
            trimmed.write(output, format="MSEED")
            payload = output.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            destination = project_path(
                root,
                f"data/processed/requested/{case['case_id']}.{digest[:12]}.mseed",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != digest:
                    raise RuntimeError("Derivado existente alterado.")
            else:
                with destination.open("xb") as out:
                    out.write(payload)
            derivation = {
                "path": destination.relative_to(root).as_posix(),
                "sha256": digest,
                "original_path": case["waveform_request"]["path"],
                "original_sha256": before,
                "operation": "copy_then_trim",
                "pad": False,
                "nearest_sample": False,
                "interpolation": False,
                "resampling": False,
                "filtering": False,
                "instrument_response_correction": False,
                "requested_starttime_utc": case["starttime_utc"],
                "requested_endtime_utc": case["endtime_utc"],
                "actual_trace_times": signature(trimmed),
            }
            write_json(
                project_path(root, f"data/metadata/derivations/{case['case_id']}.json"), derivation
            )
            quality.update(
                status=requested_quality["status"],
                reject_reasons=requested_quality["reject_reasons"],
                warnings=sorted(set(raw_quality["warnings"] + requested_quality["warnings"])),
                quality_scope="requested_window_copy",
                requested_window=requested_quality,
                derived_artifact=derivation,
            )
        except ValueError as error:
            quality["preparation_blocked"] = str(error)
    if sha256_file(path) != before or before != case["waveform_request"]["sha256"]:
        raise RuntimeError("Original alterado durante inspección.")
    quality["original_sha256_before_and_after"] = before
    quality["repeated_read_metadata_equal"] = True
    return quality


def _csv(path, rows):
    """Exporta filas de manifiesto, serializando estructuras anidadas como JSON."""
    encoded = [
        {
            name: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else value
            for name, value in row.items()
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(encoded).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def probe_services(root, config):
    """Comprueba rutas públicas y registra su respuesta con el presupuesto configurado."""
    budget = new_budget(config["limits"])
    results = []
    urls = config.get("probe_urls", [config["provider"]["station_url"]])
    for base in urls:
        for service in ("station", "dataselect", "event"):
            result = fetch_resource(
                root,
                base.rstrip("/") + f"/fdsnws/{service}/1/version",
                {},
                "diagnostic",
                budget,
                allow_http=config["provider"]["allow_http"],
            )
            result["service"] = service
            results.append(result)
    # Confirmar que el fallo de catálogo es de recurso y no confundirlo con cero eventos.
    query = config.get("diagnostic_event_query")
    if query and config["provider"].get("event_url"):
        results.append(
            fetch_resource(
                root,
                config["provider"]["event_url"].rstrip("/") + "/fdsnws/event/1/query",
                {**query, "format": "xml", "limit": 5},
                "catalog",
                budget,
                allow_http=config["provider"]["allow_http"],
            )
        )
    write_json(
        project_path(root, "data/metadata/services_linux.json"),
        {
            "checked_at_utc": utc_now(),
            "results": results,
            "budget": budget,
            "note": "Versiones no garantizan ondas; 404 no significa ausencia de sismos.",
        },
    )
    return results


def run_acquisition(root, config_path, *, download=False, make_figures=True):
    """Ejecuta catálogo, planificación y adquisición, con evidencia por corrida.

    La opción download habilita solicitudes de forma de onda; la planificación puede consultar
    catálogo e inventario. Cada caso conserva el estado de la solicitud y el control técnico.
    Se registran configuración, versiones y hashes de implementación.
    """
    root = Path(root).resolve()
    config = load_config(root, config_path)
    budget = new_budget(config["limits"])
    started = time.monotonic()
    run_id = utc_now().replace(":", "").replace("+", "_")
    output = project_path(root, f"data/metadata/runs/{run_id}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "configuration.json", config)
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "executable": sys.executable,
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "matplotlib",
                    "pandas",
                    "PyWavelets",
                    "obspy",
                    "requests",
                )
            },
        },
    )
    write_json(
        output / "implementation_hashes.json",
        {
            name: sha256_file(Path(__file__).with_name(name + ".py"))
            for name in (
                "acquisition",
                "catalog",
                "data",
                "quality",
                "preprocessing",
                "figures",
                "references",
            )
        },
    )
    events, rejections, catalog_status = load_catalog(root, config, budget)
    write_json(output / "catalog_status.json", catalog_status)
    write_json(output / "events.json", events)
    write_json(output / "catalog_rejections.json", rejections)
    cases, station_records = plan_cases(root, config, events, budget)
    batch_size = config["limits"]["batch_size"]
    processed = 0
    figures = 0
    for case in cases:
        if case["status"] != "planned":
            continue
        if processed >= config["limits"]["max_cases"]:
            case["status"] = "deferred_case_limit"
            continue
        if not download:
            continue
        processed += 1
        params = {
            "network": case["network"],
            "station": case["station"],
            "location": case["location"] or "--",
            "channel": ",".join(sorted({item["channel"] for item in case["channels"]})),
            "starttime": str(UTCDateTime(case["starttime_utc"])),
            "endtime": str(UTCDateTime(case["endtime_utc"])),
        }
        request = fetch_resource(
            root,
            config["provider"]["waveform_url"].rstrip("/") + "/fdsnws/dataselect/1/query",
            params,
            "waveform",
            budget,
            allow_http=config["provider"]["allow_http"],
        )
        case["waveform_request"] = request
        case["status"] = request["status"]
        if request["status"] in ("downloaded", "cached"):
            try:
                quality = inspect_case(
                    root,
                    case,
                    config["quality"],
                    config["limits"]["max_samples_per_case"],
                    config.get("prepare_requested_window", False),
                )
                quality_path = project_path(
                    root, f"results/quality/{run_id}/{case['case_id']}.json"
                )
                write_json(quality_path, quality)
                case["quality_path"] = quality_path.relative_to(root).as_posix()
                case["quality_status"] = quality["status"]
                case["reject_reasons"] = quality["reject_reasons"]
                case["warnings"] = quality["warnings"]
                case["quality_scope"] = quality.get("quality_scope", "original_records")
                case["derived_artifact"] = quality.get("derived_artifact")
                if case["component_layout"] == "incomplete_or_other":
                    case["warnings"].append("three_component_layout_not_confirmed")
                case["status"] = "acquired"
                if case.get("arrival_candidates"):
                    case.update(confirm_arrival_coverage(case, quality))
                if (
                    make_figures
                    and "traces" in quality
                    and figures < config["limits"]["max_figures"]
                ):
                    from seismic_wavelets.figures import plot_inspection

                    case["figure_path"] = plot_inspection(
                        root,
                        read(project_path(root, request["path"]), format="MSEED"),
                        case,
                        quality,
                    )
                    figures += 1
            except Exception as error:
                case.update(status="invalid_waveform_or_inspection", error=str(error))
        # Registro incremental: un fallo de la siguiente solicitud no borra lo obtenido.
        write_json(output / "manifest.json", cases)
        if processed % batch_size == 0 and processed < len(cases):
            time.sleep(config["limits"]["pause_between_batches_seconds"])

    acquired = [case for case in cases if case["status"] == "acquired"]
    failed = [case for case in cases if case["status"] not in ("acquired", "planned")]
    failures = (
        bool(failed)
        or bool(rejections)
        or catalog_status["status"] not in ("loaded", "not_provided")
    )
    qc_rejected = sum(case.get("quality_status") == "rejected_technical" for case in acquired)
    counts = Counter(case["status"] for case in cases)
    summary = {
        "run_id": run_id,
        "completed_at_utc": utc_now(),
        "status": "partial_or_blocked"
        if failures or qc_rejected
        else "technical_pilot_complete"
        if not events and acquired
        else "acquisition_complete"
        if acquired
        else "planned_only"
        if cases
        else "no_candidates",
        "download_enabled": download,
        "configuration_path": (output.relative_to(root) / "configuration.json").as_posix(),
        "catalog_status": catalog_status,
        "candidate_events": len(events),
        "acquired_event_ids": sorted({case["event_id"] for case in acquired if case["event_id"]}),
        "distinct_catalog_event_count": len(
            {case["event_id"] for case in acquired if case["event_id"]}
        ),
        "technical_windows": len(
            {case["window_id"] for case in acquired if case["event_id"] is None}
        ),
        "acquired_station_cases": len(acquired),
        "acquired_trace_count": sum(
            len(
                read(
                    project_path(root, case["waveform_request"]["path"]),
                    format="MSEED",
                    headonly=True,
                )
            )
            for case in acquired
        ),
        "candidate_case_count": len(cases),
        "states": dict(counts),
        "qc_rejected_cases": qc_rejected,
        "failed_or_deferred_case_count": len(failed),
        "stations_deferred_by_limit": config["selection"]["stations"][
            config["limits"]["max_stations"] :
        ],
        "technical_windows_deferred_by_limit": config["technical_windows"][
            config["limits"]["max_windows"] :
        ],
        "requests_and_volume": budget,
        "waveform_cache_hits": sum(
            case.get("waveform_request", {}).get("status") == "cached" for case in acquired
        ),
        "waveform_downloads": sum(
            case.get("waveform_request", {}).get("status") == "downloaded" for case in acquired
        ),
        "raw_waveform_bytes": sum(case["waveform_request"]["bytes"] for case in acquired),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "all_requested_cases_successful": bool(cases)
        and len(acquired) == len(cases)
        and not qc_rejected
        and not failures,
        "missing_files_or_requests": [
            {
                "case_id": case["case_id"],
                "station": case["station"],
                "status": case["status"],
                "error": case.get("error"),
            }
            for case in failed
        ],
        "manifest_path": (output.relative_to(root) / "manifest.json").as_posix(),
    }
    write_json(output / "manifest.json", cases)
    write_json(output / "station_channels.json", station_records)
    write_json(output / "summary.json", summary)
    _csv(output / "events.csv", events)
    _csv(output / "manifest.csv", cases)
    _csv(output / "station_channels.csv", station_records)
    write_json(
        project_path(root, "data/metadata/latest_run.json"),
        {
            "run_directory": output.relative_to(root).as_posix(),
            "summary": summary,
        },
    )
    build_collection(root)
    return summary


def build_collection(root):
    """Consolidar la última revisión de cada caso sin contar reejecuciones como datos nuevos."""
    root = Path(root).resolve()
    cases = {}
    events = {}
    runs = []
    for path in sorted(project_path(root, "data/metadata/runs").glob("*/manifest.json")):
        if not (path.parent / "summary.json").is_file():
            continue
        runs.append(path.parent.relative_to(root).as_posix())
        for case in json.loads(path.read_text(encoding="utf-8")):
            previous = cases.get(case["case_id"], {})
            if not case.get("figure_path") and previous.get("figure_path"):
                if project_path(root, previous["figure_path"]).is_file() and case.get(
                    "waveform_request", {}
                ).get("sha256") == previous.get("waveform_request", {}).get("sha256"):
                    case["figure_path"] = previous["figure_path"]
            cases[case["case_id"]] = {
                **case,
                "last_run_directory": path.parent.relative_to(root).as_posix(),
            }
            if case.get("event"):
                events[case["event_id"]] = case["event"]
    manifest = list(cases.values())
    acquired = [case for case in manifest if case["status"] == "acquired"]
    station_counts = Counter(case["station"] for case in acquired)
    trace_count = 0
    sample_count = 0
    raw_files = {}
    inventory_files = {}
    for case in acquired:
        path = project_path(root, case["waveform_request"]["path"])
        header = read(path, format="MSEED", headonly=True)
        trace_count += len(header)
        sample_count += sum(trace.stats.npts for trace in header)
        raw_files[case["waveform_request"]["path"]] = path.stat().st_size
    for case in manifest:
        if case.get("inventory_path"):
            path = project_path(root, case["inventory_path"])
            inventory_files[case["inventory_path"]] = path.stat().st_size
    summary = {
        "updated_at_utc": utc_now(),
        "runs": runs,
        "candidate_events": len(events),
        "distinct_acquired_catalog_events": len(
            {case["event_id"] for case in acquired if case["event_id"]}
        ),
        "acquired_station_window_cases": len(acquired),
        "acquired_trace_count": trace_count,
        "raw_sample_count": sample_count,
        "stations_with_waveforms": dict(station_counts),
        "technical_windows": len(
            {case["window_id"] for case in acquired if case["event_id"] is None}
        ),
        "unique_raw_waveform_files": raw_files,
        "unique_raw_waveform_bytes": sum(raw_files.values()),
        "unique_stationxml_files": inventory_files,
        "unique_stationxml_bytes": sum(inventory_files.values()),
        "case_states": dict(Counter(case["status"] for case in manifest)),
        "qc_review_candidates": sum(
            case.get("quality_status") == "review_candidate" for case in acquired
        ),
        "qc_rejected_cases": sum(
            case.get("quality_status") == "rejected_technical" for case in acquired
        ),
        "missing_cases": [
            {
                "case_id": case["case_id"],
                "station": case["station"],
                "window_id": case["window_id"],
                "status": case["status"],
            }
            for case in manifest
            if case["status"] != "acquired"
        ],
    }
    write_json(project_path(root, "data/metadata/collection_manifest.json"), manifest)
    write_json(project_path(root, "data/metadata/collection_summary.json"), summary)
    write_json(project_path(root, "data/metadata/candidate_events.json"), list(events.values()))
    _csv(project_path(root, "data/metadata/collection_manifest.csv"), manifest)
    _csv(project_path(root, "data/metadata/candidate_events.csv"), list(events.values()))
    return summary
