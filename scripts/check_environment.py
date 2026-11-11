"""Comprueba instalación, importaciones y operaciones DSP con una señal sintética.

Las comparaciones con referencias y las reconstrucciones prueban convenciones y tolerancias
numéricas. El MiniSEED se construye en memoria para verificar lectura y escritura. El
diagnóstico genera artefactos técnicos en results.
"""

import argparse
import importlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

from seismic_wavelets.environment import project_python, require_project_python

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Evitar escrituras de cachés/configuración fuera del proyecto.
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(PROJECT_ROOT / ".cache")
os.environ["JUPYTER_CONFIG_DIR"] = str(PROJECT_ROOT / ".runtime" / "jupyter-config")
os.environ["JUPYTER_DATA_DIR"] = str(PROJECT_ROOT / ".runtime" / "jupyter-data")
os.environ["JUPYTER_RUNTIME_DIR"] = str(PROJECT_ROOT / ".runtime" / "jupyter")
os.environ["IPYTHONDIR"] = str(PROJECT_ROOT / ".runtime" / "ipython")
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
os.environ["JUPYTER_PREFER_ENV_PATH"] = "1"

DEMO_CONFIG = {
    "purpose": "technical_environment_check_only",
    "seed": 20261012,
    "sample_rate_hz": 128.0,
    "sample_count": 512,
    "tone_hz": 8.0,
    "pulse_center_s": 2.0,
    "pulse_width_s": 0.08,
    "pulse_tone_hz": 24.0,
    "noise_std": 0.01,
    "amplitude_units": "arbitrary",
    "stft_window": "hann",
    "stft_nperseg": 64,
    "stft_noverlap": 32,
    "stft_boundary": "zeros",
    "stft_padded": True,
    "cwt_wavelet": "morl",
    "cwt_scales": list(range(4, 36)),
    "cwt_method": "fft",
    "dwt_wavelet": "db2",
    "dwt_level": 3,
    "dwt_mode": "symmetric",
}

DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "PyWavelets": "pywt",
    "obspy": "obspy",
    "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel",
    "pytest": "pytest",
    "ruff": "ruff",
}


def check_kernel():
    """Ejecutar una celda real con el kernel local y comprobar su intérprete."""
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    # Consultar y arrancar la misma especificación local, aunque exista un kernel de
    # usuario con el mismo nombre registrado por otro clon.
    kernel_dir = project_python(PROJECT_ROOT).parent.parent / "share/jupyter/kernels"
    manager = KernelSpecManager(kernel_dirs=[str(kernel_dir)])
    spec = manager.get_kernel_spec("seismic-wavelets-review")
    require_project_python(PROJECT_ROOT, spec.argv[0])
    kernel_manager = KernelManager(
        kernel_name="seismic-wavelets-review", kernel_spec_manager=manager
    )
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import json, sys\n"
                "import seismic_wavelets, numpy, scipy, matplotlib, pandas, pywt, obspy\n"
                "print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix}))"
            )
        ]
    )
    NotebookClient(notebook, timeout=60, km=kernel_manager).execute(cwd=str(PROJECT_ROOT))
    output = "".join(item.get("text", "") for item in notebook.cells[0].outputs)
    kernel = json.loads(output.strip())
    require_project_python(PROJECT_ROOT, kernel["executable"], kernel["prefix"])
    return {"name": "seismic-wavelets-review", **kernel, "executed": True}


def run_checks(artifacts_dir=PROJECT_ROOT / "results"):
    """Comprobar importaciones, transformadas, lectura sísmica y figura sin GUI."""
    require_project_python(PROJECT_ROOT, sys.executable, sys.prefix)

    packages = {}
    for distribution, module in DISTRIBUTIONS.items():
        importlib.import_module(module)
        packages[distribution] = version(distribution)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pywt
    from obspy import Stream, Trace, read
    from scipy import fft, signal

    import seismic_wavelets

    config = DEMO_CONFIG
    fs = config["sample_rate_hz"]
    count = config["sample_count"]
    time = np.arange(count) / fs
    rng = np.random.default_rng(config["seed"])
    samples = (
        np.sin(2 * np.pi * config["tone_hz"] * time)
        + np.exp(-0.5 * ((time - config["pulse_center_s"]) / config["pulse_width_s"]) ** 2)
        * np.sin(2 * np.pi * config["pulse_tone_hz"] * time)
        + config["noise_std"] * rng.standard_normal(count)
    )
    # Compartir la señal mantiene constante la entrada; cada representación se verifica con su
    # propia convención.
    # La misma señal entra a todas las comprobaciones.
    spectrum = fft.rfft(samples)
    reconstructed_fft = fft.irfft(spectrum, n=count)
    freq_stft, time_stft, stft = signal.stft(
        samples,
        fs=fs,
        window=config["stft_window"],
        nperseg=config["stft_nperseg"],
        noverlap=config["stft_noverlap"],
        boundary=config["stft_boundary"],
        padded=config["stft_padded"],
    )
    _, reconstructed_stft = signal.istft(
        stft,
        fs=fs,
        window=config["stft_window"],
        nperseg=config["stft_nperseg"],
        noverlap=config["stft_noverlap"],
        boundary=True,
    )
    cwt, freq_cwt = pywt.cwt(
        samples,
        config["cwt_scales"],
        config["cwt_wavelet"],
        sampling_period=1 / fs,
        method=config["cwt_method"],
    )
    coeffs = pywt.wavedec(
        samples, config["dwt_wavelet"], mode=config["dwt_mode"], level=config["dwt_level"]
    )
    reconstructed_dwt = pywt.waverec(coeffs, config["dwt_wavelet"], mode=config["dwt_mode"])[:count]
    errors = {}
    for name, reconstructed in [
        ("fft", reconstructed_fft),
        ("stft", reconstructed_stft[:count]),
        ("dwt", reconstructed_dwt),
    ]:
        np.testing.assert_allclose(reconstructed, samples, rtol=1e-11, atol=1e-11)
        errors[name] = float(np.max(np.abs(reconstructed - samples)))
    if cwt.shape != (len(config["cwt_scales"]), count) or not np.isfinite(cwt).all():
        raise RuntimeError("Salida CWT inválida.")
    if not np.all(freq_cwt > 0) or not np.all(np.diff(freq_cwt) < 0):
        raise RuntimeError("Conversión escala-frecuencia CWT inesperada.")
    if not np.all(freq_cwt < fs / 2):
        raise RuntimeError("Frecuencia central CWT fuera del intervalo de Nyquist.")

    # Datos sintéticos en memoria: sin registrar estación, evento o llegadas ficticios.
    trace = Trace(samples.astype(np.float64), header={"sampling_rate": fs})
    buffer = BytesIO()
    Stream([trace]).write(buffer, format="MSEED", encoding="FLOAT64")
    buffer.seek(0)
    recovered = read(buffer, format="MSEED")[0]
    np.testing.assert_array_equal(recovered.data, samples)
    if recovered.stats.sampling_rate != fs:
        raise RuntimeError("Muestreo MiniSEED inesperado.")
    table = pd.DataFrame({"method": list(errors), "max_abs_error": list(errors.values())})
    figures = artifacts_dir / "figures"
    tables = artifacts_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    table_path = tables / "environment_check.csv"
    table.to_csv(table_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.plot(time, samples, linewidth=0.9)
    ax.set(xlabel="Tiempo (s)", ylabel="Amplitud (u. arbitrarias)")
    ax.set_title("Demostración sintética — comprobación técnica del entorno")
    ax.text(
        0.01,
        0.02,
        f"Fuente: síntesis local | fs={fs:g} Hz | N={count} | semilla={config['seed']}",
        transform=ax.transAxes,
        fontsize=8,
    )
    figure_path = figures / "environment_check.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    if not figure_path.is_file() or figure_path.stat().st_size == 0:
        raise RuntimeError("No se guardó la figura.")

    return {
        "status": "passed",
        "purpose": config["purpose"],
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "project_version": seismic_wavelets.__version__,
        "project_module": seismic_wavelets.__file__,
        "package_versions": packages,
        "demo_config": config,
        "checks": {
            "max_abs_reconstruction_error": errors,
            "stft_shape": list(stft.shape),
            "stft_frequency_bins": len(freq_stft),
            "stft_time_bins": len(time_stft),
            "cwt_shape": list(cwt.shape),
            "cwt_frequency_hz_range": [float(freq_cwt.min()), float(freq_cwt.max())],
            "cwt_central_frequencies_below_nyquist": True,
            "dwt_coefficient_lengths": [len(item) for item in coeffs],
            "obspy_mseed_roundtrip": "passed",
            "matplotlib_backend": matplotlib.get_backend(),
            "figure": figure_path.relative_to(PROJECT_ROOT).as_posix(),
            "table": table_path.relative_to(PROJECT_ROOT).as_posix(),
            "kernel": check_kernel(),
        },
        "limitations": [
            "Solo prueba técnica; no representa resultados sísmicos reales.",
            "No evalúa localización, ruido, concentración, compacidad ni costo.",
            "No caracteriza bordes CWT/DWT ni valida escalas para uso experimental.",
            "No verifica energía equivalente entre métodos ni respuesta instrumental.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "environment" / "environment_check.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directorio para la figura y la tabla; permite preservar las de referencia.",
    )
    args = parser.parse_args()
    report_path = args.report.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    if not all(path.is_relative_to(PROJECT_ROOT) for path in (report_path, artifacts_dir)):
        parser.error("El informe y los artefactos deben guardarse dentro del proyecto.")
    report = run_checks(artifacts_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
