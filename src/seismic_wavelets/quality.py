"""Control de calidad sin rellenar, alinear, remuestrear ni calibrar originales."""

import numpy as np
from obspy import UTCDateTime


def inspect_stream(stream, channels, start, end, rules):
    """Evalúa aptitud técnica de un Stream respecto a canales y ventana esperados.

    Comprueba época, respuesta, orientación, continuidad, muestreo y sincronía. Los extremos
    repetidos y pasos abruptos son indicadores heurísticos para revisión. La función devuelve
    diagnósticos y no corrige trazas ni convierte sus unidades.
    """
    start, end = UTCDateTime(start), UTCDateTime(end)
    expected = {item["seed_id"]: item for item in channels}
    report = {"traces": [], "gaps_or_overlaps": [], "reject_reasons": [], "warnings": []}
    observed = set()
    signatures = {}
    for trace in stream:
        metadata = expected.get(trace.id)
        data = np.ma.asarray(trace.data)
        masked = int(np.ma.count_masked(data))
        values = np.asarray(data.compressed(), dtype=float)
        finite = values[np.isfinite(values)]
        fs = float(trace.stats.sampling_rate)
        item = {
            "seed_id": trace.id,
            "sampling_rate_hz": fs,
            "npts": int(trace.stats.npts),
            "starttime_utc": str(trace.stats.starttime),
            "endtime_utc": str(trace.stats.endtime),
            "dtype": str(trace.data.dtype),
            "masked_samples": masked,
            "nonfinite_samples": int(np.count_nonzero(~np.isfinite(values))),
            "raw_units": "unknown",
            "response_available": False,
            "physical_units_after_correction": "not_applied",
            "metadata": metadata,
        }
        if not np.isfinite(fs) or fs <= 0 or values.size == 0:
            report["reject_reasons"].append(f"empty_or_invalid_sampling:{trace.id}")
            report["traces"].append(item)
            continue
        if masked or item["nonfinite_samples"]:
            report["reject_reasons"].append(f"masked_or_nonfinite:{trace.id}")
        if metadata is None:
            report["reject_reasons"].append(f"unexpected_channel:{trace.id}")
        else:
            observed.add(trace.id)
            # Entrada/salida de respuesta se describen; no confundirlas con muestras calibradas.
            item["response_available"] = metadata["response_available"]
            if (metadata.get("response_output_units") or "").upper() in ("COUNTS", "COUNT"):
                item["raw_units"] = "counts"
            else:
                report["warnings"].append(f"raw_units_unknown:{trace.id}")
            if not metadata["response_available"]:
                report["warnings"].append(f"response_missing:{trace.id}")
                if rules["require_response"]:
                    report["reject_reasons"].append(f"response_required:{trace.id}")
            if not metadata.get("covers_requested_window", False):
                report["reject_reasons"].append(f"metadata_epoch_not_covering_window:{trace.id}")
            if metadata.get("sample_rate_hz") is None or not np.isclose(
                fs, metadata["sample_rate_hz"], rtol=1e-8, atol=0
            ):
                report["reject_reasons"].append(f"sampling_metadata_mismatch:{trace.id}")
            if metadata.get("azimuth_deg") is None or metadata.get("dip_deg") is None:
                report["warnings"].append(f"orientation_unknown:{trace.id}")
            component = trace.stats.channel[-1:]
            azimuth, dip = metadata.get("azimuth_deg"), metadata.get("dip_deg")
            if component in "NE12" and dip is not None and abs(dip) > 5:
                report["warnings"].append(f"nonhorizontal_orientation:{trace.id}")
            if component == "Z" and dip is not None and abs(abs(dip) - 90) > 5:
                report["warnings"].append(f"nonvertical_orientation:{trace.id}")
            if component == "N" and azimuth is not None and min(azimuth, 360 - azimuth) > 5:
                report["warnings"].append(f"N_label_orientation_mismatch:{trace.id}")
            if component == "E" and azimuth is not None and abs(azimuth - 90) > 5:
                report["warnings"].append(f"E_label_orientation_mismatch:{trace.id}")
        if finite.size:
            minimum, maximum = float(finite.min()), float(finite.max())
            # Una concentración en los extremos amerita inspección visual; por sí sola no
            # identifica saturación del instrumento.
            plateau = np.count_nonzero((finite == minimum) | (finite == maximum)) / finite.size
            item.update(minimum=minimum, maximum=maximum, extreme_value_fraction=float(plateau))
            if minimum == maximum:
                report["reject_reasons"].append(f"constant_trace:{trace.id}")
            elif plateau >= rules["extreme_fraction_warning"]:
                report["warnings"].append(f"possible_clipping_or_quantization:{trace.id}")
            if np.issubdtype(trace.data.dtype, np.integer):
                integer = np.iinfo(trace.data.dtype)
                if minimum <= integer.min or maximum >= integer.max:
                    report["warnings"].append(f"integer_limit_reached:{trace.id}")
            if finite.size > 2:
                # La mediana de los incrementos absolutos sirve como escala robusta para
                # comparar con el mayor salto observado.
                differences = np.abs(np.diff(finite))
                median = float(np.median(differences))
                jump = float(differences.max())
                item["maximum_step"] = jump
                item["median_absolute_step"] = median
                if jump > 0 and jump > rules["step_warning_factor"] * max(median, 1e-20):
                    report["warnings"].append(f"abrupt_step_candidate:{trace.id}")
        signatures.setdefault(trace.id, []).append(
            (
                trace.stats.starttime,
                trace.stats.endtime,
                fs,
                trace.stats.npts,
            )
        )
        report["traces"].append(item)
    for gap in stream.get_gaps():
        report["gaps_or_overlaps"].append(
            {
                "seed_id": ".".join(gap[:4]),
                "left_end_utc": str(gap[4]),
                "right_start_utc": str(gap[5]),
                "seconds": gap[6],
                "samples": gap[7],
                "kind": "gap" if gap[6] > 0 else "overlap",
            }
        )
    if report["gaps_or_overlaps"]:
        report["reject_reasons"].append("gaps_or_overlaps")
    for seed_id in sorted(set(expected) - observed):
        report["reject_reasons"].append(f"missing_channel:{seed_id}")
    common = []
    for seed_id, segments in signatures.items():
        if len(segments) != 1:
            report["reject_reasons"].append(f"multiple_segments_not_merged:{seed_id}")
        left = min(item[0] for item in segments)
        right = max(item[1] for item in segments)
        rates = {item[2] for item in segments}
        # Convertir la tolerancia expresada en muestras a segundos antes de comparar marcas UTC.
        tolerance = rules["coverage_tolerance_samples"] / min(rates) if min(rates) > 0 else 0
        if left > start + tolerance or right < end - tolerance:
            report["reject_reasons"].append(f"incomplete_requested_window:{seed_id}")
        if left < start - tolerance or right > end + tolerance:
            report["warnings"].append(f"extra_samples_outside_request:{seed_id}")
        if len(segments) == 1:
            common.append(segments[0])
    # Una rejilla común exige igual número de muestras y frecuencia, además de coincidencia
    # temporal de ambos extremos dentro de tolerancia.
    synchronized = len(common) == len(expected) and bool(common)
    if synchronized:
        first = common[0]
        synchronized = all(
            np.isclose(item[2], first[2], rtol=1e-8, atol=0)
            and item[3] == first[3]
            and abs(item[0] - first[0]) <= rules["synchronization_tolerance_samples"] / first[2]
            and abs(item[1] - first[1]) <= rules["synchronization_tolerance_samples"] / first[2]
            for item in common[1:]
        )
    report["synchronized_components"] = synchronized
    if not synchronized:
        report["reject_reasons"].append("components_not_synchronized")
    report["reject_reasons"] = sorted(set(report["reject_reasons"]))
    report["warnings"] = sorted(set(report["warnings"]))
    report["status"] = "rejected_technical" if report["reject_reasons"] else "review_candidate"
    report["eligibility_note"] = "Aptitud técnica provisional; no aprobación experimental."
    return report
