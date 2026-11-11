# Ejecutar con: source scripts/activate.sh (Bash).
_seismic_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolver desde la ubicación del script permite activar el entorno aunque se invoque desde otro
# directorio.
source "$_seismic_project_root/.venv/bin/activate"
# Mantener cachés, configuración y conexiones del kernel bajo rutas locales del proyecto.
export MPLCONFIGDIR="$_seismic_project_root/.cache/matplotlib"
export XDG_CACHE_HOME="$_seismic_project_root/.cache"
export JUPYTER_CONFIG_DIR="$_seismic_project_root/.runtime/jupyter-config"
export JUPYTER_DATA_DIR="$_seismic_project_root/.runtime/jupyter-data"
export JUPYTER_RUNTIME_DIR="$_seismic_project_root/.runtime/jupyter"
export IPYTHONDIR="$_seismic_project_root/.runtime/ipython"
export TMPDIR="$_seismic_project_root/.runtime/tmp"
export JUPYTER_PLATFORM_DIRS=1
export JUPYTER_PREFER_ENV_PATH=1
mkdir -p "$JUPYTER_RUNTIME_DIR" "$TMPDIR"
chmod 700 "$JUPYTER_RUNTIME_DIR"
unset _seismic_project_root
