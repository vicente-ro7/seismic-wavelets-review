"""Identifica el entorno Python del clon sin seguir el ejecutable hasta el Python base."""

import os
from pathlib import Path


def project_python(root):
    """Devuelve la ruta nativa que crea venv en Windows o Fedora."""
    relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return Path(root).resolve() / ".venv" / relative


def require_project_python(root, executable, prefix=None):
    """Exige un ejecutable de este clon y, cuando está disponible, su prefijo real."""
    expected = project_python(root)
    actual = Path(executable)
    # En Fedora python puede ser un enlace al binario base. Comparar primero el directorio
    # del lanzador impide aceptar el Python global solo porque ambos apuntan al mismo binario.
    if (
        not actual.is_absolute()
        or actual.parent.resolve() != expected.parent.resolve()
        or not actual.is_file()
        or not expected.is_file()
        or not actual.samefile(expected)
        or (prefix is not None and Path(prefix).resolve() != expected.parent.parent)
    ):
        raise RuntimeError(f"Usa el Python de .venv del proyecto: {expected}. Recibido: {actual}")
