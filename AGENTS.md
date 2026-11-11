# Trabajo en esta edición de revisión

- Leer `README.md` y `docs/CODIGO.md`. Trabajar desde la raíz y con la `.venv`
  propia, recreada con el bloqueo de su plataforma. No versionar entornos,
  cachés, salidas ejecutadas de notebooks ni informes regenerados en `results/`.
- Conservar los MiniSEED originales y su atribución. Los recortes deben
  indicar operación, tiempos efectivos, unidades y SHA-256; nunca modificar
  los originales para producir una figura.
- Mantener la demo y las pruebas ejecutables sin red. La adquisición real
  requiere invocación explícita; consultar `docs/ADQUISICION.md`.
- Distinguir ilustración, comparación exploratoria y validación. Mantener
  parámetros explícitos y probar cambios con `python -m pytest -q` y Ruff.
- Documentación y textos de interfaz en español; identificadores de código
  en inglés. Revisar autoría, procedencia y alcance antes de publicar.
