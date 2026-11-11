# Entorno y dependencias

El proyecto usa Python 3.14. `pyproject.toml` declara las dependencias
directas; `requirements.lock` (Linux) y `requirements.windows.lock` (Windows)
fijan las versiones directas y transitivas. Instala el bloqueo de tu sistema
antes del paquete local. Las instrucciones del README no necesitan activar
PowerShell ni modificar paquetes del sistema.
La [guía Windows](WINDOWS.md) recupera la receta de CPython 3.14.10 autónomo
y reúne la instalación, el registro del kernel y el ensayo sin red.

| Dependencia | Uso |
| --- | --- |
| NumPy | Rejillas, vectores, ventanas, potencia y estadísticas. |
| SciPy | FFT, ventanas Hann y funciones numéricas de apoyo. |
| PyWavelets | CWT, conversiones escala–frecuencia y descomposición DWT. |
| ObsPy | Lectura de MiniSEED, StationXML y QuakeML, tiempo UTC y metadatos. |
| Requests | Acceso FDSN en la ruta opcional de adquisición. |
| pandas | Tablas de diagnóstico y del notebook, y salidas de adquisición. |
| Matplotlib | Figuras de la demo y controles numéricos. |
| JupyterLab / ipykernel / nbconvert | Interfaz, kernel y ejecución por terminal del notebook. |
| pytest / Ruff | Pruebas, análisis estático y formato. |
| pip / setuptools | Instalación del bloqueo y del paquete local. |

`python -m pip check` comprueba la compatibilidad declarada. Para verificar
el intérprete, las operaciones numéricas y el kernel real:

```bash
python scripts/register_kernel.py
python scripts/check_environment.py --report .runtime/environment/environment.json --artifacts-dir .runtime/environment
```

En PowerShell sustituye `python` por `& $python`. Al cambiar de máquina, vuelve
a crear la .venv y registra el kernel; el entorno virtual no es portable.
El bloqueo Windows viene de la rama de origen; esta edición se verificó en
Fedora con CPython 3.14.10. Desde Fedora también se comprobó que hay ruedas para las 111 distribuciones
fijadas en Windows x64 con CPython 3.14, sin instalarlas. La ejecución de
esta copia y su uso desde VS Code en Windows requieren comprobación allí.

PyWavelets puede exponer una cadena de módulo distinta de la versión de
distribución. El diagnóstico y las pruebas numéricas son la comprobación
operativa; no sustituyas versiones solo para igualar ambas cadenas.
