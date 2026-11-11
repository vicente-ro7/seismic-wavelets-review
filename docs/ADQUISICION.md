# Adquisición FDSN opcional

La demo y las pruebas no consultan servicios. La adquisición solo se ejecuta
al invocar `scripts/acquire.py`. El ejemplo `configs/acquisition_mugello.json`
usa HTTPS con verificación de certificado y limita la selección a FNVD,
canal vertical y al evento de Mugello. Requiere conexión y disponibilidad
del proveedor.

Desde la raíz y con el Python del proyecto:

```bash
# Comprueba los servicios definidos en la configuración (con red).
python scripts/acquire.py probe --config configs/acquisition_mugello.json

# Planifica con catálogo e inventario. No descarga la forma de onda.
python scripts/acquire.py run --config configs/acquisition_mugello.json --no-figures

# Autoriza además la descarga de formas de onda para los casos seleccionados.
python scripts/acquire.py run --config configs/acquisition_mugello.json --download --no-figures
```

En Windows, usa `& $python` como en el README. Las formas de onda originales,
los recortes, los metadatos de solicitudes y el manifiesto se guardan en
`results/acquisition/`. No sustituyen el caso incluido en `data/mugello/`.
Una consulta nueva puede devolver bytes distintos por cómo el proveedor
empaqueta los registros; la demo sigue usando el original y el recorte incluidos.

El flujo separa el acceso al catálogo, el inventario por época, la consulta
de muestras y el control de calidad. La caché conserva la consulta y la
respuesta original; la copia recortada registra su SHA-256 y las operaciones
aplicadas. Se comprueba cobertura de referencias antes de usarlas como
picks de una traza. Para otro proveedor, prepara una configuración explícita
con sus endpoints, estaciones, intervalo y límites. La prueba automática
de adquisición simula las respuestas; una ejecución de la demo no valida
la disponibilidad de los servicios reales.
