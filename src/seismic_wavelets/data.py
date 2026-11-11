"""Descarga pública acotada, conservación de bytes y caché verificable."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

import requests


def utc_now():
    """Devuelve la marca actual en ISO 8601 con zona UTC explícita."""
    return datetime.now(UTC).isoformat()


def sha256_file(path):
    """Calcula el SHA-256 de los bytes del archivo mediante lectura por bloques."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(root, relative):
    """Resuelve rutas del proyecto, incluidas referencias Windows de manifiestos previos."""
    root = Path(root).resolve()
    path = Path(relative)
    if not path.is_absolute():
        windows_path = PureWindowsPath(path)
        if windows_path.anchor:
            raise ValueError(f"Ruta no relativa al proyecto: {relative}")
        path = Path(*windows_path.parts)
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Ruta fuera del proyecto: {relative}")
    return path


def write_json(path, value):
    """Actualizar solo productos derivados; nunca usar para reemplazar originales."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(path)


def validate_url(url, allow_http=False):
    """Valida un endpoint público y exige autorización explícita para HTTP."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ValueError("Se requiere una URL pública HTTP(S).")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("URL de proveedor sin credenciales, query ni fragmento.")
    if parsed.scheme == "http" and not allow_http:
        raise ValueError("HTTP exige allow_http=true y justificación en configuración.")


def new_budget(limits):
    """Crea el contador de peticiones y bytes que comparten las etapas de una corrida."""
    return {"requests": 0, "downloaded_bytes": 0, "limits": limits}


def _record(root, result):
    """Añade el resultado de una consulta a la bitácora JSONL de procedencia."""
    path = project_path(root, "data/metadata/request_log.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def fetch_resource(root, url, params, kind, budget, *, allow_http=False, session=None):
    """GET con TLS verificado; bytes originales, sin merge, trim ni filtrado.

    Los fallos no crean una entrada de caché válida. Una entrada alterada se rechaza,
    conservando el archivo para auditoría. Reanudación al repetir la misma consulta.
    """
    root = Path(root).resolve()
    validate_url(url, allow_http)
    limits = budget["limits"]
    query = {"url": url, "params": params, "kind": kind}
    # La clave identifica la consulta, mientras que el hash del sidecar comprueba el contenido
    # recibido. Son identidades distintas.
    key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
    suffix = {"waveform": ".mseed", "station": ".xml", "catalog": ".xml"}.get(kind, ".txt")
    relative = f"data/raw/fdsn/{kind}/{key}{suffix}"
    path = project_path(root, relative)
    sidecar = project_path(root, f"data/metadata/requests/{key}.json")
    result = {
        "request_key": key,
        "checked_at_utc": utc_now(),
        "query": query,
        "transport": urlsplit(url).scheme,
        "tls_verification": True,
        "path": relative,
        "attempts": [],
    }
    if path.exists() or sidecar.exists():
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            if cached["query"] != query or cached["sha256"] != sha256_file(path):
                raise ValueError("Hash o consulta de caché no coincide.")
            result.update(
                status="cached",
                sha256=cached["sha256"],
                bytes=path.stat().st_size,
                original_fetch=cached,
            )
        except (OSError, ValueError, KeyError) as error:
            result.update(status="cache_invalid", error=str(error))
        return _record(root, result)

    http = session or requests.Session()
    try:
        # Los intentos fallidos consumen presupuesto de red. La caché válida se resuelve antes
        # del bucle, sin consumir otro intento.
        for index in range(limits["retries"] + 1):
            if budget["requests"] >= limits["max_requests"]:
                result.update(status="request_limit")
                break
            remaining = limits["max_total_bytes"] - budget["downloaded_bytes"]
            if remaining <= 0:
                result.update(status="volume_limit")
                break
            budget["requests"] += 1
            attempt = {"number": index + 1, "started_at_utc": utc_now()}
            started = time.monotonic()
            retry = False
            received = 0
            try:
                with http.get(
                    url,
                    params=params,
                    timeout=limits["timeout_seconds"],
                    verify=True,
                    stream=True,
                    headers={"User-Agent": "seismic-wavelets/0.1"},
                ) as response:
                    attempt.update(http_status=response.status_code, final_url=response.url)
                    if urlsplit(url).scheme == "https" and urlsplit(response.url).scheme != "https":
                        result.update(status="transport_downgrade")
                    elif response.status_code == 200:
                        maximum = min(limits["max_request_bytes"], remaining)
                        payload = bytearray()
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            received += len(chunk)
                            budget["downloaded_bytes"] += len(chunk)
                            if received > maximum:
                                result.update(status="volume_limit")
                                break
                            payload.extend(chunk)
                        else:
                            if not payload:
                                result.update(status="no_data")
                            else:
                                path.parent.mkdir(parents=True, exist_ok=True)
                                # Exclusivo: jamás reemplazar un original.
                                with path.open("xb") as stream:
                                    stream.write(payload)
                                result.update(
                                    status="downloaded",
                                    sha256=sha256_file(path),
                                    bytes=len(payload),
                                    final_url=response.url,
                                    content_type=response.headers.get("Content-Type"),
                                    fetched_at_utc=utc_now(),
                                )
                    else:
                        states = {
                            204: "no_data",
                            400: "bad_request",
                            401: "restricted",
                            403: "restricted",
                            404: "http_not_found",
                            413: "request_too_large",
                            429: "rate_limited",
                        }
                        result.update(status=states.get(response.status_code, "http_error"))
                        # No inferir ausencia de evento a partir de un 404.
                        body = next(response.iter_content(chunk_size=2048), b"")[:2048]
                        result["error_body"] = body.decode("utf-8", errors="replace")
                        retry = response.status_code >= 500 or response.status_code == 429
            except requests.exceptions.SSLError as error:
                result.update(status="tls_error", error=str(error))
            except requests.exceptions.Timeout as error:
                result.update(status="timeout", error=str(error))
                retry = True
            except requests.exceptions.RequestException as error:
                result.update(status="network_error", error=str(error))
                retry = True
            attempt.update(elapsed_seconds=round(time.monotonic() - started, 6), bytes=received)
            result["attempts"].append(attempt)
            if result["status"] == "downloaded":
                write_json(sidecar, result)
                break
            if not retry or index == limits["retries"]:
                break
            time.sleep(min(2**index, 5))
    finally:
        if session is None:
            http.close()
    return _record(root, result)
