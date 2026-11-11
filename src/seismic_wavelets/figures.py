"""Inspección de muestras originales, sin normalización ni procesamiento oculto."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seismic_wavelets.data import project_path


def plot_inspection(root, stream, case, quality, maximum_points=6000):
    """Grafica una traza por eje en segundos relativos al comienzo solicitado.

    El paso de dibujo limita puntos visibles y no es un remuestreo con antialiasing; puede
    ocultar picos breves. Se conservan las cuentas digitales o la unidad desconocida según el
    diagnóstico. La figura es una inspección técnica.
    """
    path = project_path(root, f"results/figures/inspection_{case['case_id']}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(stream),
        1,
        figsize=(10, max(3, 2.2 * len(stream))),
        squeeze=False,
        layout="constrained",
    )
    units = {item["seed_id"]: item["raw_units"] for item in quality["traces"]}
    from obspy import UTCDateTime

    reference = UTCDateTime(case["starttime_utc"])
    for axis, trace in zip(axes[:, 0], stream, strict=True):
        stride = max(1, (trace.stats.npts + maximum_points - 1) // maximum_points)
        axis.plot(
            trace.times()[::stride] + (trace.stats.starttime - reference),
            trace.data[::stride],
            linewidth=0.6,
        )
        label = "cuentas digitales" if units.get(trace.id) == "counts" else "unidad desconocida"
        axis.set(title=f"{trace.id} | fs={trace.stats.sampling_rate:g} Hz", ylabel=label)
        axis.set_xlabel(f"Tiempo desde {case['starttime_utc']} (s)")
        axis.grid(alpha=0.2)
    kind = "ventana técnica sin evento verificado" if case["event_id"] is None else "caso candidato"
    figure.suptitle(
        f"Inspección SGC — {kind}\n"
        f"Original sin filtrar | fuente: {case['waveform_request']['query']['url']}\n"
        "Gráfica con submuestreo visual; revisar todos los datos en el informe de calidad",
        fontsize=10,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path.relative_to(root).as_posix()
