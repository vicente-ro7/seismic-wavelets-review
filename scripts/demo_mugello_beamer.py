"""Demostración reproducible de una traza, sus disparos y el tiempo por lote."""

import argparse
import json
import os
import webbrowser
from base64 import b64encode
from html import escape
from pathlib import Path
from string import Template

import matplotlib
import numpy as np
from obspy import UTCDateTime

from seismic_wavelets.data import write_json
from seismic_wavelets.pilot import (
    COLORS,
    FS,
    HOP,
    METHODS,
    OFFSET,
    PROTOCOL,
    TIMING,
    WIDTH,
    detect,
    match_p,
    measure_batch,
    prepare,
)
from seismic_wavelets.waveforms import load_trace

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LIVE_DEST = ROOT / "results/mugello"


def write_presentation(destination, figure_path, trace, waveform_hash, picks, samples, rows):
    """Render the measured run as one self-contained HTML file."""
    image = b64encode(figure_path.read_bytes()).decode("ascii")
    table_rows = []
    matched = []
    for row in rows:
        association = row["association"]
        if association:
            delay_s = association["relative_to_p_s"]
            delay = f"{delay_s:+.3f}"
            matched.append((delay_s, row["method"]))
        else:
            delay = "Sin asociación"
        timing = row["timing"]
        table_rows.append(
            f"<tr><th scope='row'>{escape(row['method'])}</th><td>{delay}</td>"
            f"<td>{timing['median_ms']:.3f}</td><td>{timing['p95_ms']:.3f}</td></tr>"
        )
    interpretation = f"En esta ejecución, {len(matched)} de {len(rows)} métodos se asociaron con P."
    if matched:
        earliest = min(delay for delay, _ in matched)
        methods = ", ".join(method for delay, method in matched if delay == earliest)
        interpretation += f" Primer disparo asociado: {methods} ({earliest:+.3f} s respecto a P)."

    def utc_label(phase):
        utc = trace.stats.starttime + picks[phase]
        return utc.strftime("%H:%M:%S.%f")[:-3]

    template = Template(Path(__file__).with_name("demo_mugello_report.html").read_text("utf-8"))
    report = destination / "index.html"
    report.write_text(
        template.substitute(
            seed=escape(trace.id),
            rate=f"{FS:g}",
            signal_seconds=f"{len(samples) / FS:g}",
            pick_p=utc_label("P"),
            pick_s=utc_label("S"),
            image=image,
            table_rows="".join(table_rows),
            repetitions=TIMING["repetitions"],
            warmups=TIMING["warmups"],
            width=f"{WIDTH / FS:g}",
            hop=f"{HOP / FS:g}",
            band_low=PROTOCOL["band_hz"][0],
            band_high=PROTOCOL["band_hz"][1],
            background=PROTOCOL["background_frames"],
            holdoff=PROTOCOL["holdoff_s"],
            early=PROTOCOL["early_s"],
            late=PROTOCOL["late_s"],
            thresholds=" · ".join(
                f"{escape(row['method'])}: {row['threshold']:.3f}" for row in rows
            ),
            interpretation=escape(interpretation),
            waveform_hash=escape(waveform_hash),
        ),
        encoding="utf-8",
    )
    return report


def open_presentation(report):
    """Print the fallback path even when the desktop launcher is unavailable."""
    report_url = report.resolve().as_uri()
    print(f"Informe para proyectar: {report_url}")
    if os.environ.get("SEISMIC_DEMO_NO_BROWSER") == "1":
        return
    try:
        if webbrowser.open(report_url):
            return
    except OSError, webbrowser.Error:
        pass
    print("Abre index.html en el navegador para ver el informe.")


def load_demo_case():
    """Carga la traza archivada, las llegadas de referencia y los umbrales del piloto."""
    case = json.loads((ROOT / "data/mugello/case.json").read_text(encoding="utf-8"))
    trace = load_trace(ROOT, case)
    seed, start = trace.id, trace.stats.starttime
    if seed != "IV.FNVD..HHZ" or trace.stats.sampling_rate != FS:
        raise ValueError("El canal de la demostración difiere de FNVD a 100 Hz.")
    picks = {
        ref["phase"]: UTCDateTime(ref["pick_time_utc"]) - start
        for ref in case["arrival_references"]
    }
    thresholds = case["thresholds"]
    return trace, case["derived_artifact"]["sha256"], picks, thresholds


def run_demo_detectors(samples, picks, thresholds):
    """Mide el cálculo de cada detector y contrasta sus disparos con la P manual."""
    rows = []
    curves = {}
    for method in METHODS:
        threshold = thresholds[method]
        (triggers, ends, ratios), timing = measure_batch(
            detect, samples, method, threshold, sample_count=len(samples)
        )
        # El final de cada bloque es el instante en que su indicador está disponible.
        curves[method] = (ends / FS - picks["P"], ratios / threshold)
        rows.append(
            {
                "method": method,
                "threshold": threshold,
                "association": match_p(triggers, picks["P"], picks["S"]),
                "timing": timing,
            }
        )
    return rows, curves


def style_demo_axis(axis):
    axis.set_facecolor("#f5f9fb")
    axis.grid(axis="y", color="#cbdde7", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_preparation(trace, samples):
    """Show the initial segment before and after the exact preparation step."""
    count = 4 * FS
    t = np.arange(count) / FS
    reference = float(np.mean(trace.data[:OFFSET]))
    fig, axes = plt.subplots(
        2, 1, sharex=True, sharey=True, figsize=(10, 4.3), layout="constrained"
    )
    raw, centered = axes
    raw.plot(t, trace.data[:count], color=COLORS[0], lw=1.2)
    raw.axhline(reference, color="#b24721", ls="--", label="Media del primer segundo")
    raw.set(ylabel="Amplitud (cuentas)", title="A · Registro original")
    raw.legend(loc="upper right", frameon=False)
    centered.text(
        0.5, 0.9, "Tramo descartado", ha="center", transform=centered.get_xaxis_transform()
    )
    centered.plot(t[OFFSET:], samples[: count - OFFSET], color=COLORS[2], lw=1.2)
    centered.axhline(0, color="#73899a", lw=0.8)
    centered.set(
        ylabel="Amplitud (cuentas)",
        title="B · Señal después de restar la media y descartar el primer segundo",
        xlabel="Tiempo desde el inicio del registro (s)",
    )
    for axis in axes:
        axis.axvspan(0, OFFSET / FS, color="#e7b55c", alpha=0.28)
        axis.set_xlim(0, count / FS)
        style_demo_axis(axis)
    return fig


def plot_crossings(rows, curves):
    """Show the threshold crossings near P without changing the detector."""
    fig, axis = plt.subplots(figsize=(10, 3.2), layout="constrained")
    for row, color, line in zip(rows, COLORS, ("-", "--", "-."), strict=True):
        time, ratio = curves[row["method"]]
        visible = (time >= -0.2) & (time <= 0.37)
        axis.step(
            time[visible],
            ratio[visible],
            where="post",
            color=color,
            ls=line,
            lw=1.7,
            label="DWT Haar" if row["method"] == "DWT" else row["method"],
        )
        if match := row["association"]:
            axis.scatter(match["relative_to_p_s"], 1, color=color, edgecolor="white", zorder=5)
    axis.axvline(0, color="#486581", ls=":", label="P manual")
    axis.axhline(1, color="#486581", lw=0.9)
    axis.annotate(
        "Umbral = 1",
        (0.99, 1),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color="#334e68",
    )
    axis.set(
        title="Detalle · cruces cerca de la llegada P",
        xlabel="Tiempo antes (−) o después (+) de la llegada P (s)",
        ylabel="Indicador / umbral",
        yticks=(0, 1, 2, 3),
        ylim=(0, 3.2),
        xlim=(-0.2, 0.37),
    )
    axis.legend(ncols=4, loc="upper left", frameon=False)
    style_demo_axis(axis)
    return fig


def plot_demo(trace, picks, rows, curves):
    """Dibuja la traza y los indicadores con un origen temporal común en P."""
    data, seed = trace.data, trace.id
    time = np.arange(len(data)) / FS - picks["P"]
    view = (time >= -2) & (time <= 4)
    fig, (wave, scores) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.7))
    wave.plot(time[view], data[view], color="#174f65", lw=0.8)
    wave.set(ylabel="Amplitud (cuentas)", title=f"A · Forma de onda original · {seed}")
    wave.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    for row, color, line in zip(rows, COLORS, ("-", "--", "-."), strict=True):
        method = row["method"]
        matched = row["association"]
        relative, normalized = curves[method]
        visible = (relative >= -2) & (relative <= 4)
        scores.step(
            relative[visible],
            normalized[visible],
            where="post",
            color=color,
            ls=line,
            lw=1.6,
            label="DWT Haar" if method == "DWT" else method,
        )
        delay = matched["relative_to_p_s"] if matched else None
        if delay is not None:
            scores.scatter(delay, 1, color=color, edgecolors="white", zorder=3)
    scores.axhline(1, color="#486581", lw=0.9)
    scores.annotate(
        "Umbral = 1",
        (0.99, 1),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color="#334e68",
        bbox={"facecolor": "#f5f9fb", "edgecolor": "none", "pad": 1.5},
    )
    scores.set(
        title="B · Respuesta de los detectores",
        yscale="symlog",
        ylim=(0, None),
        ylabel="Indicador / umbral",
        xlabel="Tiempo antes (−) o después (+) de la llegada P (s)",
        xlim=(-2, 4),
    )
    scores.set_yticks([0, 1, 10, 100, 1000, 10000], ["0", "1", "10", "100", "1 000", "10 000"])
    scores.legend(loc="upper left", ncols=3, frameon=False)
    for axis in (wave, scores):
        style_demo_axis(axis)
        axis.axvline(0, ls="--", color="0.35", lw=0.85)
        axis.axvline(picks["S"] - picks["P"], ls=":", color="0.35", lw=0.85)
    for label, at in (("P manual", 0), ("S manual", picks["S"] - picks["P"])):
        wave.annotate(
            label,
            (at, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            weight="bold",
        )
    fig.tight_layout()
    return fig


def main():
    """Recalcula los tres detectores de Mugello y exporta la figura y sus métricas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--present", action="store_true", help="Abre la demo recién calculada en el navegador."
    )
    parser.add_argument("--output", type=Path, default=LIVE_DEST, help="Directorio de salida.")
    args = parser.parse_args()
    destination = args.output.resolve()
    trace, waveform_hash, picks, thresholds = load_demo_case()
    seed, start = trace.id, trace.stats.starttime
    samples = prepare(trace.data)
    rows, curves = run_demo_detectors(samples, picks, thresholds)
    print(f"{seed} | {len(samples) / FS:.0f} s preparados | P manual {start + picks['P']} UTC")
    print("Metodo     Desfase vs P (s)   Mediana lote (ms)   P95 lote (ms)")
    for row in rows:
        method, matched, timing = row["method"], row["association"], row["timing"]
        display = f"{matched['relative_to_p_s']:+.3f}" if matched else "sin asociación"
        print(f"{method:9} {display:>14} {timing['median_ms']:>17.3f} {timing['p95_ms']:>14.3f}")
    report = export_demo_report(destination, trace, waveform_hash, picks, samples, rows, curves)
    if args.present:
        open_presentation(report)


def export_demo_report(destination, trace, waveform_hash, picks, samples, rows, curves):
    """Export the figure, data and summary from the same detector execution."""
    seed, start = trace.id, trace.stats.starttime
    fig = plot_demo(trace, picks, rows, curves)
    destination.mkdir(parents=True, exist_ok=True)
    figure_path = destination / "demo_mugello.png"
    fig.savefig(figure_path, dpi=155)
    plt.close(fig)
    write_json(
        destination / "demo.json",
        {
            "seed": seed,
            "waveform_sha256": waveform_hash,
            "start_utc": str(start),
            "picks_s_from_start": picks,
            "trace_samples": len(trace.data),
            "methods": rows,
        },
    )
    return write_presentation(destination, figure_path, trace, waveform_hash, picks, samples, rows)


if __name__ == "__main__":
    main()
