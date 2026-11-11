"""Catálogos configurables; no atribuir eventos a ventanas sin fuente verificable."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from io import BytesIO

from obspy import UTCDateTime, read_events

from seismic_wavelets.data import fetch_resource, project_path, sha256_file


def utc_time(value):
    """Entrada ISO-8601 inequívoca en UTC; nunca suponer zona de una hora local."""
    if not isinstance(value, str) or not value.endswith(("Z", "+00:00")):
        raise ValueError(f"Fecha requiere Z o +00:00 explícito: {value!r}")
    return UTCDateTime(value)


def normalize_event(row):
    """Normaliza un evento reportado y conserva su procedencia y sus picks.

    Exige identificador de origen y fecha inequívoca en UTC. Latitud y longitud están en
    grados, profundidad en kilómetros; los valores ausentes quedan como None. Un evento
    catalogado y cada pick conservan referencias separadas.
    """
    if not row.get("event_id") or not row.get("source"):
        raise ValueError("Evento requiere event_id y source; no se generan IDs oficiales.")
    event = dict(row)
    event["event_id"] = str(row["event_id"])
    event["origin_time_utc"] = str(utc_time(row["origin_time_utc"]))
    for key in ("latitude", "longitude", "depth_km", "magnitude"):
        value = row.get(key)
        event[key] = None if value in (None, "") else float(value)
        if event[key] is not None and not math.isfinite(event[key]):
            raise ValueError(f"Valor no finito: {key}")
    if event["latitude"] is not None and not -90 <= event["latitude"] <= 90:
        raise ValueError("Latitud fuera de rango.")
    if event["longitude"] is not None and not -180 <= event["longitude"] <= 180:
        raise ValueError("Longitud fuera de rango.")
    event.setdefault("magnitude_type", None)
    event.setdefault("picks", [])
    for pick in event["picks"]:
        utc_time(pick["time_utc"])
        if not pick.get("source") or not pick.get("reference_type"):
            raise ValueError("Pick requiere fuente y tipo: reportado o predicción declarada.")
    event["provenance_status"] = "source_record_preserved_not_independently_relocated"
    return event


def events_from_quakeml(payload, source):
    """Lee QuakeML y devuelve eventos normalizados y rechazos documentados.

    Selecciona el origen y la magnitud preferidos cuando existen; convierte metros a
    kilómetros. Conserva los picks del evento con identificador SEED, fase sugerida,
    incertidumbre y estado de evaluación disponibles. Vincula cada pick con los Arrival
    del origen seleccionado por identificador, sin usar hora, cercanía ni phase_hint
    para deducir un vínculo.
    """
    events = []
    rejected = []
    for item in read_events(BytesIO(payload), format="QUAKEML"):
        origin = item.preferred_origin() or (item.origins[0] if item.origins else None)
        magnitude = item.preferred_magnitude() or (item.magnitudes[0] if item.magnitudes else None)
        if origin is None or origin.time is None or item.resource_id is None:
            rejected.append({"event_id": str(item.resource_id), "reason": "missing_id_or_origin"})
            continue
        arrivals = defaultdict(list)
        origin_id = str(origin.resource_id) if origin.resource_id is not None else None
        for arrival in origin.arrivals:
            if arrival.pick_id is not None:
                arrivals[str(arrival.pick_id)].append(
                    {
                        "arrival_id": str(arrival.resource_id) if arrival.resource_id else None,
                        "phase": arrival.phase,
                    }
                )
        pick_ids = Counter(str(pick.resource_id) for pick in item.picks if pick.resource_id)
        picks = []
        for pick in item.picks:
            if pick.time is None or pick.waveform_id is None:
                continue
            picks.append(
                {
                    "pick_id": str(pick.resource_id) if pick.resource_id else None,
                    "origin_arrivals": arrivals.get(str(pick.resource_id), []),
                    "pick_id_ambiguous": pick_ids[str(pick.resource_id)] > 1,
                    "time_utc": str(pick.time),
                    "seed_id": pick.waveform_id.get_seed_string(),
                    "phase": pick.phase_hint,
                    "evaluation_mode": pick.evaluation_mode,
                    "evaluation_status": pick.evaluation_status,
                    "uncertainty_seconds": pick.time_errors.uncertainty,
                    "reference_type": "catalog_reported_pick",
                    "source": source,
                }
            )
        events.append(
            normalize_event(
                {
                    "event_id": str(item.resource_id),
                    "source": source,
                    "origin_time_utc": str(origin.time),
                    "latitude": origin.latitude,
                    "longitude": origin.longitude,
                    "depth_km": None if origin.depth is None else origin.depth / 1000,
                    "magnitude": None if magnitude is None else magnitude.mag,
                    "magnitude_type": None if magnitude is None else magnitude.magnitude_type,
                    "origin_id": origin_id,
                    "origin_selection": "preferred_or_first_reported",
                    "origin_evaluation_status": origin.evaluation_status,
                    "catalog_format": "quakeml",
                    "unresolved_arrival_pick_ids": sorted(
                        arrivals.keys() - {pick["pick_id"] for pick in picks}
                    ),
                    "picks": picks,
                }
            )
        )
    return events, rejected


def _preserve_catalog(root, path):
    """Conserva una copia del archivo de catálogo identificada por su SHA-256."""
    digest = sha256_file(path)
    destination = project_path(root, f"data/raw/catalogs/{digest}{path.suffix.lower()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError("Copia de catálogo alterada.")
    else:
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
    return {"path": destination.relative_to(root).as_posix(), "sha256": digest}


def load_catalog(root, config, budget):
    """Carga eventos, rechazos y procedencia desde el modo de catálogo configurado.

    Admite CSV, QuakeML local, eventos explícitos y FDSN. Reaplica filtros localmente, rechaza
    IDs duplicados y registra los datos ausentes cuando impiden evaluar un filtro. El estado
    del proveedor se conserva si la consulta falla.
    """
    settings = config["catalog"]
    mode = settings["mode"]
    rejected = []
    artifact = None
    if mode == "none":
        return [], [], {"status": "not_provided", "events": 0}
    if mode == "fdsn":
        params = dict(settings["query"])
        params.update(format="xml", limit=config["limits"]["max_events"])
        result = fetch_resource(
            root,
            config["provider"]["event_url"].rstrip("/") + "/fdsnws/event/1/query",
            params,
            "catalog",
            budget,
            allow_http=config["provider"]["allow_http"],
        )
        if result["status"] not in ("downloaded", "cached"):
            return [], [], result
        artifact = {"path": result["path"], "sha256": result["sha256"]}
        events, rejected = events_from_quakeml(
            project_path(root, result["path"]).read_bytes(), result["query"]["url"]
        )
    elif mode in ("csv", "quakeml"):
        path = project_path(root, settings["path"])
        artifact = _preserve_catalog(root, path)
        source = settings["source"]
        if mode == "quakeml":
            events, rejected = events_from_quakeml(path.read_bytes(), source)
        else:
            events = []
            mapping = settings.get("column_map", {})
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for index, row in enumerate(csv.DictReader(stream), 2):
                    values = {mapping.get(key, key): value for key, value in row.items()}
                    values.setdefault("source", source)
                    values["catalog_format"] = "csv"
                    try:
                        events.append(normalize_event(values))
                    except (KeyError, TypeError, ValueError) as error:
                        rejected.append({"row": index, "reason": str(error)})
    elif mode == "explicit":
        events = []
        for row in settings["events"]:
            try:
                events.append(normalize_event({**row, "catalog_format": "explicit"}))
            except (KeyError, TypeError, ValueError) as error:
                rejected.append({"event_id": row.get("event_id"), "reason": str(error)})
        artifact = {
            "configuration_events_sha256": hashlib.sha256(
                json.dumps(settings["events"], sort_keys=True).encode()
            ).hexdigest()
        }
    else:
        raise ValueError(f"Modo de catálogo desconocido: {mode}")
    accepted = []
    seen = set()
    filters = settings.get("query", {})
    for event in events:
        reason = None
        if event["event_id"] in seen:
            reason = "duplicate_event_id"
        for key, field, minimum in [
            ("minlatitude", "latitude", True),
            ("maxlatitude", "latitude", False),
            ("minlongitude", "longitude", True),
            ("maxlongitude", "longitude", False),
            ("minmagnitude", "magnitude", True),
            ("maxmagnitude", "magnitude", False),
            ("mindepth", "depth_km", True),
            ("maxdepth", "depth_km", False),
        ]:
            if key in filters:
                value = event[field]
                if value is None or (value < filters[key] if minimum else value > filters[key]):
                    reason = f"filter_{key}_missing_or_outside"
        origin = utc_time(event["origin_time_utc"])
        if "starttime" in filters and origin < utc_time(filters["starttime"]):
            reason = "before_query_interval"
        if "endtime" in filters and origin > utc_time(filters["endtime"]):
            reason = "after_query_interval"
        if len(accepted) >= config["limits"]["max_events"]:
            reason = "event_limit"
        seen.add(event["event_id"])
        if reason:
            rejected.append({"event_id": event["event_id"], "reason": reason})
        else:
            accepted.append(event)
    return accepted, rejected, {"status": "loaded", "mode": mode, "artifact": artifact}
