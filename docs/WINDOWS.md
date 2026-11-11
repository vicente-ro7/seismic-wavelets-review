# Instalación y comprobación en Windows

Usa PowerShell en la raíz del proyecto. Los ejemplos usan **CPython 3.14
x64 (AMD64)** y el [bloqueo Windows](../requirements.windows.lock).
La primera instalación necesita acceso a las ruedas de Python; después,
la demo y las pruebas funcionan sin consultar a INGV. No hace falta activar
PowerShell, modificar su política de ejecución ni instalar paquetes globales.

## 1. Crear el entorno

Si el lanzador ya encuentra Python 3.14 x64, ejecuta:

```powershell
py -3.14 -c "import platform, sys; print(sys.version); assert platform.machine().lower() in ('amd64', 'x86_64')"
py -3.14 -m venv .venv
```

Si no tienes Python 3.14, puedes instalarlo para tu usuario y volver a esos
comandos. También puedes usar este intérprete autónomo dentro del proyecto.
La receta recupera CPython 3.14.10 x64, compilación 20261026, que se utilizó
en la comprobación previa del proyecto en Windows. Ejecútala desde un clon
sin esos directorios; no reemplaza un intérprete ni un entorno existente.

```powershell
$ErrorActionPreference = 'Stop'
$archive = '.runtime/cpython-3.14.10-win-amd64.tar.gz'
$runtime = '.runtime/python-3.14.10'
$url = 'https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.14.10%2B20261026-x86_64-pc-windows-msvc-install_only.tar.gz'
$sha256 = '98bbe6a0d798bcbad142e96e5930a330e4dc407638e898b515c51eb73a27580d'
if ((Test-Path -LiteralPath $runtime) -or (Test-Path -LiteralPath '.venv')) {
    throw 'Ya existe un runtime o una .venv; revisa ese entorno antes de reemplazarlo.'
}
New-Item -ItemType Directory -Path '.runtime' -Force | Out-Null
curl.exe -fLsS $url -o $archive
if ($LASTEXITCODE -ne 0) { throw 'No se pudo descargar CPython.' }
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $sha256) {
    throw 'SHA-256 inesperado; no se extrae el archivo.'
}
New-Item -ItemType Directory -Path $runtime | Out-Null
tar.exe -xf $archive -C $runtime --strip-components=1
if ($LASTEXITCODE -ne 0) { throw 'No se pudo extraer CPython.' }
& "$runtime/python.exe" -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear la .venv.' }
```

## 2. Instalar el proyecto

Ejecuta cada paso y detente si da error. Las variables que siguen solo
afectan a esta sesión de PowerShell.

```powershell
$python = '.\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path $PWD '.cache/pip'
$env:MPLCONFIGDIR = Join-Path $PWD '.cache/matplotlib'
& $python -m pip install --only-binary=:all: -r requirements.windows.lock
& $python -m pip install --no-build-isolation --no-deps -e .
& $python scripts/register_kernel.py
& $python -m pip check
```

El bloqueo Windows sustituye las dependencias de terminal de Unix
`pexpect` y `ptyprocess` por `colorama` y `pywinpty`. Las otras 109
versiones fijadas coinciden con las de Fedora.

## 3. Probar sin conexión

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pytest -q
& $python scripts/demo_mugello_beamer.py
```

Abre `results/mugello/index.html` en el navegador. Para comprobar también
el notebook con el kernel registrado, sin abrir el navegador al final:

```powershell
$env:SEISMIC_DEMO_NO_BROWSER = '1'
try {
    & $python -m nbconvert --to notebook --execute notebooks/03_demo_mugello.ipynb --ExecutePreprocessor.kernel_name=seismic-wavelets-review --ExecutePreprocessor.timeout=90 --output 03_demo_mugello.executed.ipynb --output-dir .runtime/windows
} finally {
    Remove-Item Env:SEISMIC_DEMO_NO_BROWSER -ErrorAction SilentlyContinue
}
```

En VS Code, abre `seismic-wavelets-review.code-workspace`. Si conserva otro
intérprete, usa **Python: Select Interpreter** y selecciona
`.venv\Scripts\python.exe`. En el notebook, elige
**Python (seismic-wavelets-review)** o el entorno con esa misma ruta.
La primera celda imprime el ejecutable que realmente usa. Si quieres
repetir la demo desde la interfaz, usa **Ejecutar todo** o la tarea
**DSP: Demostración sísmica (Mugello)**. En la ejecución normal la última
celda intenta abrir el HTML.

La adquisición es una [ruta opcional](ADQUISICION.md); no es requisito
para validar esta instalación. Consulta [Entorno](ENTORNO.md) para el
alcance de las comprobaciones por plataforma.
