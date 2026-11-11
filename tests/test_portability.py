"""Regresiones de identidad del entorno y de rutas compartidas entre sistemas."""

import pytest

from seismic_wavelets.data import project_path
from seismic_wavelets.environment import project_python, require_project_python


def test_project_interpreter_rejects_other_launcher_and_prefix(tmp_path):
    executable = project_python(tmp_path)
    executable.parent.mkdir(parents=True)
    executable.touch()
    require_project_python(tmp_path, executable, tmp_path / ".venv")

    other = tmp_path / executable.name
    # Simula un alias al mismo archivo sin depender de permisos de symlink en Windows.
    other.hardlink_to(executable)
    assert executable.samefile(other)
    with pytest.raises(RuntimeError, match="Usa el Python"):
        require_project_python(tmp_path, other)
    with pytest.raises(RuntimeError, match="Usa el Python"):
        require_project_python(tmp_path, executable, tmp_path)


def test_project_paths_read_legacy_windows_metadata(tmp_path):
    expected = (tmp_path / "data/raw/trace.mseed").resolve()
    assert project_path(tmp_path, "data/raw/trace.mseed") == expected
    assert project_path(tmp_path, r"data\raw\trace.mseed") == expected
    with pytest.raises(ValueError, match="proyecto"):
        project_path(tmp_path, r"..\outside.mseed")
