"""Registra seismic-wavelets-review con el ejecutable absoluto de la .venv.

El registro local acompaña al entorno. La opción --user registra también en el directorio
que Jupyter elige para el usuario. Al recrear el entorno se debe repetir el registro.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from ipykernel.kernelspec import install

from seismic_wavelets.environment import require_project_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user", action="store_true", help="Registrar además en Jupyter de usuario."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        require_project_python(root, sys.executable, sys.prefix)
    except RuntimeError as error:
        parser.error(str(error))
    destinations = [
        install(
            kernel_name="seismic-wavelets-review",
            display_name="Python (seismic-wavelets-review)",
            prefix=sys.prefix,
        )
    ]
    if args.user:
        # La activación de Fedora fija una carpeta de datos privada del proyecto.
        # Ignorar solo ese valor al pedir el registro de usuario; conservar otros perfiles.
        if os.environ.get("JUPYTER_DATA_DIR") == str(root / ".runtime/jupyter-data"):
            del os.environ["JUPYTER_DATA_DIR"]
        destinations.append(
            install(
                kernel_name="seismic-wavelets-review",
                display_name="Python (seismic-wavelets-review)",
                user=True,
            )
        )
    for destination in destinations:
        spec_path = Path(destination) / "kernel.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["env"] = {
            "MPLCONFIGDIR": str(root / ".cache/matplotlib"),
            "XDG_CACHE_HOME": str(root / ".cache"),
            "IPYTHONDIR": str(root / ".runtime/ipython"),
        }
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        print(spec_path)


if __name__ == "__main__":
    main()
