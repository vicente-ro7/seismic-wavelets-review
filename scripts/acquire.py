"""Run optional FDSN acquisition separately from the bundled demonstration."""

import argparse
import json
from pathlib import Path

from seismic_wavelets.acquisition import load_config, probe_services, run_acquisition
from seismic_wavelets.data import project_path, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["probe", "run"])
    parser.add_argument("--config", required=True, help="Configuración relativa al proyecto.")
    parser.add_argument("--download", action="store_true", help="Descargar formas de onda.")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.download and args.command != "run":
        parser.error("--download solo se admite con run.")
    project = Path(__file__).resolve().parents[1]
    config = project_path(project, args.config)
    load_config(project, config)
    work = project / "results/acquisition"
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work / f"config-{sha256_file(config)}.json"
    if not snapshot.exists():
        snapshot.write_bytes(config.read_bytes())
    if args.command == "probe":
        results = probe_services(work, load_config(work, snapshot.name))
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0 if all(item["status"] in {"cached", "downloaded"} for item in results) else 2
    summary = run_acquisition(
        work, snapshot.name, download=args.download, make_figures=not args.no_figures
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2 if summary["status"] in {"partial_or_blocked", "no_candidates"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
