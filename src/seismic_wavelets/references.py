"""Asociación de llegadas de catálogo y comprobación de su cobertura por caso."""

from collections import defaultdict

from seismic_wavelets.catalog import utc_time


def _exclusion(event, pick, reason):
    """Registra la causa de exclusión junto a la identidad y la fecha de la marca."""
    return {
        "event_id": event["event_id"],
        "origin_id": event.get("origin_id"),
        "pick_id": pick.get("pick_id"),
        "seed_id": pick.get("seed_id"),
        "time_utc": pick["time_utc"],
        "reason": reason,
    }


def _catalog_exclusion(event, pick):
    """Comprueba que una marca del catálogo se vincule a una fase del origen elegido."""
    links = pick.get("origin_arrivals", [])
    if event.get("catalog_format") != "quakeml" or not event.get("origin_id"):
        return "origin_link_unverified"
    if pick.get("reference_type") != "catalog_reported_pick":
        return "not_a_reported_pick"
    if not pick.get("pick_id") or not links:
        return "not_linked_to_selected_origin"
    if pick.get("pick_id_ambiguous") or len(links) != 1:
        return "ambiguous_origin_link"
    if (
        event.get("origin_evaluation_status") == "rejected"
        or pick.get("evaluation_status") == "rejected"
    ):
        return "catalog_rejected"
    if links[0]["phase"] not in ("P", "S"):
        return "unsupported_arrival_phase"
    return None


def _case_exclusion(case, channel_epochs, pick_time):
    """Comprueba canal, ventana solicitada y vigencia del inventario en la llegada."""
    if not channel_epochs:
        return "channel_mismatch"
    if len(channel_epochs) != 1:
        return "ambiguous_channel_epoch"
    channel = channel_epochs[0]
    if not utc_time(case["starttime_utc"]) <= pick_time <= utc_time(case["endtime_utc"]):
        return "outside_requested_window"
    for prefix in ("epoch", "station_epoch"):
        left = channel.get(f"{prefix}_start_utc")
        right = channel.get(f"{prefix}_end_utc")
        if (left and pick_time < utc_time(left)) or (right and pick_time > utc_time(right)):
            return "outside_metadata_epoch"
    return None


def screen_arrival_references(case):
    """Separa candidatos P/S y exclusiones antes de recibir la forma de onda.

    Examina solo picks de la misma red y estación. Un candidato requiere un vínculo
    QuakeML único al origen seleccionado, fase del Arrival, canal exacto y un tiempo
    dentro de la ventana solicitada y las épocas conocidas del inventario.
    """
    event = case["event"]
    channels = defaultdict(list)
    for channel in case["channels"]:
        channels[channel["seed_id"]].append(channel)

    candidates = []
    exclusions = []
    for pick in event["picks"]:
        seed_id = pick.get("seed_id") or ""
        if seed_id.split(".")[:2] != [case["network"], case["station"]]:
            continue
        pick_time = utc_time(pick["time_utc"])
        reason = _catalog_exclusion(event, pick) or _case_exclusion(
            case, channels.get(seed_id, []), pick_time
        )
        if reason:
            exclusions.append(_exclusion(event, pick, reason))
            continue
        link = pick["origin_arrivals"][0]
        candidates.append(
            {
                "event_id": event["event_id"],
                "origin_id": event["origin_id"],
                "origin_time_utc": event["origin_time_utc"],
                "arrival_id": link["arrival_id"],
                "pick_id": pick["pick_id"],
                "pick_time_utc": pick["time_utc"],
                "seed_id": seed_id,
                "phase": link["phase"],
                "phase_hint": pick.get("phase"),
                "source": pick["source"],
                "uncertainty_seconds": pick.get("uncertainty_seconds"),
                "evaluation_mode": pick.get("evaluation_mode"),
                "evaluation_status": pick.get("evaluation_status"),
            }
        )
    return {
        "arrival_candidates": candidates,
        "arrival_references": [],
        "arrival_reference_exclusions": exclusions,
    }


def confirm_arrival_coverage(case, quality):
    """Confirma cobertura en la traza efectiva usada por el diagnóstico de calidad."""
    references = []
    exclusions = list(case["arrival_reference_exclusions"])
    inspected = quality.get("requested_window") or quality
    traces = {trace["seed_id"]: trace for trace in inspected.get("traces", [])}
    for candidate in case["arrival_candidates"]:
        trace = traces.get(candidate["seed_id"])
        pick_time = utc_time(candidate["pick_time_utc"])
        if quality.get("preparation_blocked"):
            reason = "requested_window_preparation_blocked"
        elif inspected["status"] != "review_candidate":
            reason = "waveform_rejected_technical"
        elif not trace or not (
            utc_time(trace["starttime_utc"]) <= pick_time <= utc_time(trace["endtime_utc"])
        ):
            reason = "outside_effective_trace"
        else:
            references.append(
                {
                    **candidate,
                    "trace_start_utc": trace["starttime_utc"],
                    "trace_end_utc": trace["endtime_utc"],
                    "sampling_rate_hz": trace["sampling_rate_hz"],
                    "quality_scope": quality.get("quality_scope", "original_records"),
                }
            )
            continue
        exclusions.append(
            _exclusion(case["event"], {**candidate, "time_utc": candidate["pick_time_utc"]}, reason)
        )
    return {
        "arrival_reference": "reported_references_covered"
        if references
        else "no_covered_references",
        "arrival_references": references,
        "arrival_reference_exclusions": exclusions,
    }
