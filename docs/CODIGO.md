# Guía del código

| Ruta | Responsabilidad |
| --- | --- |
| `scripts/demo_mugello_beamer.py` | Carga el caso, compara los métodos y exporta JSON, PNG y HTML. |
| `notebooks/03_demo_mugello.ipynb` | Explica el registro, el centrado, los indicadores, los cruces y la tabla. |
| `seismic_wavelets.waveforms` | Verifica hashes, canal, muestreo y conservación de muestras en el recorte. |
| `seismic_wavelets.pilot` | Configuración de la demo, preparación, indicadores, disparos y medición de tiempo. |
| `seismic_wavelets.transforms` | FFT unilateral, potencia espectral por ventana, razones de potencia y mapas CWT. |
| `seismic_wavelets.detection` y `metrics` | Detección causal de cruces y asociación con una llegada manual. |
| `seismic_wavelets.illustrations` y `synthetic` | Señales controladas, comprobación de kernels y reconstrucciones por banda. |
| `acquisition`, `catalog`, `references`, `data`, `quality`, `preprocessing` | Consultas, caché, trazabilidad y control técnico de registros. |

## Del registro al disparo

1. Se comprueban los hashes del original y el recorte, y se comparan sus
   muestras. Los valores son cuentas digitales, no velocidad ni aceleración.
2. Se resta la media del primer segundo y se descarta ese tramo. Los
   índices de salida recuperan el desplazamiento respecto al registro original.
3. Cada método estima potencia en ventanas de 64 muestras a 100 Hz con
   un salto de 5 muestras. STA/LTA usa la media cuadrática, STFT la potencia
   en [3,125, 25) Hz con Hann, y DWT Haar la energía de D4–D2 por muestra.
   La banda wavelet es nominal.
4. La potencia reciente se divide por el promedio de 100 ventanas previas
   que no comparten muestras con la actual. El indicador se fecha al cierre
   de la ventana. Un cruce ascendente del umbral produce un candidato; se
   espera 1 s entre disparos aceptados. Los umbrales ya están fijados.
5. Se busca el primer disparo entre P − 0,10 s y P + 1,50 s, anterior a S.
   P y S se usan para evaluar la asociación, no para calcular el indicador.

La gráfica divide cada indicador por el umbral de su método: **1** marca
el umbral y las alturas no comparan potencia física. Los puntos muestran
cuándo ocurrió el disparo. El eje temporal de esta gráfica toma P como cero.
La vista de preparación toma como cero el inicio del registro.

El tiempo por lote se mide sobre la señal ya preparada, con 20 repeticiones
tras dos calentamientos; excluye lectura, preparación y dibujo. El desfase
respecto a P es una medida diferente. Los mapas CWT centrados son
descriptivos y pueden incluir muestras posteriores al centro; no forman
parte del detector causal de esta demo.

Para leer el caso desde la biblioteca, con la .venv activa y desde la raíz:

```python
import json
from pathlib import Path

from obspy import UTCDateTime
from seismic_wavelets.pilot import detect, match_p, prepare
from seismic_wavelets.waveforms import load_trace

root = Path.cwd()
case = json.loads((root / "data/mugello/case.json").read_text(encoding="utf-8"))
trace = load_trace(root, case)
picks = {
    item["phase"]: UTCDateTime(item["pick_time_utc"]) - trace.stats.starttime
    for item in case["arrival_references"]
}
samples = prepare(trace.data)
triggers, ends, ratios = detect(samples, "DWT", case["thresholds"]["DWT"])
print(match_p(triggers, picks["P"], picks["S"]))
```
