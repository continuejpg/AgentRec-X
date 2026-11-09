#!/usr/bin/env python
"""P14-A limited Shopping Companion catalogue builder (stdlib only).

Downloads a **contiguous byte prefix** of the official Shopping Companion
``data/products.jsonl`` (HF mirror) using ordered parallel HTTP Range
requests, drops the trailing partial line, then streams and parses every
complete JSON line and projects it to the AgentRec-X P14-A document schema.

This builds a **LIMITED catalogue sample** (a contiguous byte prefix of an
~18.7 GiB / 1,298,797-product file).  It is never the full catalogue.

Outputs
-------
``<out-dir>/products_prefix.jsonl``
    Raw downloaded byte prefix, kept as the raw limited-sample artifact.
``<out-dir>/docs.jsonl``
    One projected JSON object per line (``id``, ``contents``, ``product``).
``<out-dir>/manifest.json``
    Full provenance / stats manifest.
``<out-dir>/../manifest.json`` (default ``data/p14a/manifest.json``)
    Merge-updated ``"catalogue"`` key with the same essentials.
``--fixture-out`` (default ``tests/data/p14a/catalogue_fixture.jsonl``)
    Tiny committed test fixture: the first N real projected products with
    only ``product.description`` truncated to 200 characters.

Documented CLI
--------------
    --url URL               products.jsonl URL (default: official HF mirror)
    --out-dir DIR           output directory (default: data/p14a/catalogue)
    --max-bytes N           raw prefix budget in bytes (default 1.2 GiB)
    --workers N             parallel range requests (default 10)
    --chunk-size N          HTTP Range chunk size in bytes (default 8 MiB)
    --fixture-out PATH      committed fixture output path

Extra (optional) knobs
----------------------
    --fixture-count N       fixture products (default 6)
    --timeout-seconds F     per-request HTTP timeout (default 90)
    --min-rate-mibps F      early-stop threshold (default 2.0 MiB/s aggregate)
    --rate-window-seconds F early-stop rolling window (default 45 s)
    --fresh                 ignore an existing products_prefix.jsonl
    --quiet                 suppress per-batch progress lines

Download design: the prefix target is split into fixed-size byte ranges;
``workers`` ranges run in parallel and results are written strictly in order,
so the file on disk is always a valid byte prefix (also usable for resume:
an existing ``products_prefix.jsonl`` is extended, not re-downloaded).
Peak memory is bounded by ``workers * chunk_size`` bytes.

Only the Python standard library is used: ``urllib.request``, ``json``,
``hashlib``, ``argparse``, ``concurrent.futures`` (+ os/pathlib/sys/time/
platform convenience modules).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://hf-mirror.com/datasets/yuzhan2205/Shopping-companion/"
    "resolve/main/data/products.jsonl"
)
DEFAULT_OUT_DIR = "data/p14a/catalogue"
DEFAULT_FIXTURE_OUT = "tests/data/p14a/catalogue_fixture.jsonl"
DEFAULT_MAX_BYTES = int(1.2 * (1024 ** 3))  # 1.2 GiB
DEFAULT_WORKERS = 10
DEFAULT_CHUNK_SIZE = 8 * (1024 ** 2)  # 8 MiB
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_MIN_RATE_MIBPS = 2.0
DEFAULT_RATE_WINDOW_SECONDS = 45.0
DEFAULT_FIXTURE_COUNT = 6
MAX_RETRIES = 3
DESCRIPTION_LIMIT = 1200
FIXTURE_DESCRIPTION_LIMIT = 200

SCHEMA_VERSION = "agentrecx.p14a.catalogue.v1"
SELECTION_METHOD = (
    "contiguous byte prefix; LIMITED catalogue, NOT the full "
    "1,298,797-product catalogue"
)
USER_AGENT = "agentrecx-p14a-catalogue-builder/1.0 (stdlib-only; limited prefix sample)"
CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    """Print one ASCII progress line (flush so log tails stay live)."""
    print(f"[p14a] {message}", flush=True)


def _sha256_file(path: Path) -> tuple:
    """Stream-hash a file.  Returns (hexdigest, byte_size)."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _replace_file(tmp: Path, final: Path) -> None:
    os.replace(str(tmp), str(final))


def _json_dump_bytes(obj) -> bytes:
    """Serialise JSON with a trailing newline; UTF-8, non-ASCII preserved."""
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_json_atomic(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    payload = _json_dump_bytes(obj)
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _replace_file(tmp, path)
    return len(payload)


def _encode_json_line(obj):
    """Compact single-line JSON as bytes.

    Returns ``(encoded_bytes, used_ascii_fallback)``.  Non-ASCII text is kept
    verbatim (UTF-8); the ASCII-escape fallback only triggers for lone
    surrogates (malformed-but-parsable raw JSON), keeping the file valid UTF-8.
    """
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    try:
        return line.encode("utf-8"), False
    except UnicodeEncodeError:
        escaped = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
        return escaped.encode("utf-8"), True


# ---------------------------------------------------------------------------
# HTTP: probe + ordered parallel range download
# ---------------------------------------------------------------------------


def _parse_content_range(value):
    if not value:
        return None
    match = CONTENT_RANGE_RE.search(value)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    total = None if match.group(3) == "*" else int(match.group(3))
    return start, end, total


def _http_headers(extra=None):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
    }
    if extra:
        headers.update(extra)
    return headers


def probe_source(url: str, timeout: float) -> dict:
    """HEAD probe for total size; falls back to a 1-byte range GET."""
    info = {
        "total_bytes": None,
        "etag": None,
        "accept_ranges": None,
        "method": "HEAD",
        "http_status": None,
    }
    try:
        request = urllib.request.Request(url, method="HEAD", headers=_http_headers())
        with urllib.request.urlopen(request, timeout=timeout) as response:
            info["http_status"] = getattr(response, "status", None) or response.getcode()
            info["etag"] = response.headers.get("ETag")
            info["accept_ranges"] = response.headers.get("Accept-Ranges")
            length = response.headers.get("Content-Length")
            if length is not None:
                info["total_bytes"] = int(length)
    except Exception as exc:  # noqa: BLE001 - probe must fall back, not die
        log(f"HEAD probe failed ({exc!r}); falling back to range GET")
        info["method"] = "GET-range-0-0"
        request = urllib.request.Request(
            url, headers=_http_headers({"Range": "bytes=0-0"})
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            info["http_status"] = getattr(response, "status", None) or response.getcode()
            info["etag"] = response.headers.get("ETag")
            info["accept_ranges"] = response.headers.get("Accept-Ranges")
            range_info = _parse_content_range(response.headers.get("Content-Range"))
            response.read(1)
            if range_info and range_info[2] is not None:
                info["total_bytes"] = range_info[2]
    if info["total_bytes"] is None:
        log("WARNING: source total byte size is unknown; prefix cannot be clamped to EOF")
    else:
        log(
            f"source: total_bytes={info['total_bytes']} "
            f"etag={info['etag']} accept_ranges={info['accept_ranges']}"
        )
    return info


def fetch_range(
    url: str,
    start: int,
    end: int,
    total_bytes,
    timeout: float,
    max_retries: int = MAX_RETRIES,
):
    """Fetch one inclusive byte range (206 only), retrying with backoff.

    The returned body length is always verified against the requested range
    length.  The final chunk of a prefix may legitimately be shorter than
    ``chunk_size`` only because its range end is clamped to the prefix/source
    end; an exact length is still required.
    """
    expected = end - start + 1
    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            request = urllib.request.Request(
                url, headers=_http_headers({"Range": f"bytes={start}-{end}"})
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status != 206:
                    raise RuntimeError(
                        f"range {start}-{end}: HTTP {status}, expected 206 "
                        "(server did not honour Range)"
                    )
                range_info = _parse_content_range(
                    response.headers.get("Content-Range")
                )
                if range_info is None:
                    raise RuntimeError(
                        f"range {start}-{end}: missing/invalid Content-Range"
                    )
                got_start, got_end, got_total = range_info
                if got_start != start or got_end != end:
                    raise RuntimeError(
                        f"range {start}-{end}: server answered "
                        f"{got_start}-{got_end}"
                    )
                if (
                    total_bytes is not None
                    and got_total is not None
                    and got_total != total_bytes
                ):
                    raise RuntimeError(
                        f"source changed size ({got_total} != {total_bytes})"
                    )
                body = response.read()
            if len(body) != expected:
                raise RuntimeError(
                    f"range {start}-{end}: short read {len(body)}/{expected} bytes"
                )
            return body
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last_error = exc
            if attempt <= max_retries:
                delay = float(2 ** (attempt - 1))  # 1s, 2s, 4s
                log(
                    f"retry {attempt}/{max_retries} range {start}-{end} "
                    f"after error: {exc!r} (sleep {delay:.0f}s)"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"range {start}-{end} failed after {max_retries} retries: {last_error!r}"
    )


def download_prefix(
    url: str,
    total_bytes,
    target_bytes: int,
    raw_path: Path,
    *,
    chunk_size: int,
    workers: int,
    timeout: float,
    min_rate_mibps: float,
    rate_window_seconds: float,
    fresh: bool,
    quiet: bool,
) -> dict:
    """Download ``target_bytes`` as an ordered contiguous prefix.

    Parallel range fetches are written strictly in offset order, so the raw
    file is always a valid byte prefix (resume-friendly).  Stops early with a
    recorded reason if aggregate throughput stays below ``min_rate_mibps``.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and raw_path.exists():
        raw_path.unlink()
    existing = raw_path.stat().st_size if raw_path.exists() else 0

    if existing >= target_bytes:
        final_bytes = existing
        log(
            f"reusing existing raw prefix ({final_bytes} bytes >= requested "
            f"{target_bytes}); no download needed"
        )
        return {
            "resumed_from_byte": existing,
            "downloaded_bytes": 0,
            "final_bytes": final_bytes,
            "requested_target_bytes": target_bytes,
            "chunks_requested": 0,
            "chunks_completed": 0,
            "chunk_size_bytes": chunk_size,
            "workers": workers,
            "download_seconds": 0.0,
            "aggregate_mibps": 0.0,
            "early_stop_reason": None,
            "early_stopped": False,
            "throughput_mibps_samples": [],
            "reused_existing_prefix": True,
        }

    download_target = target_bytes
    offsets = list(range(existing, download_target, chunk_size))
    ranges = [(off, min(off + chunk_size, download_target) - 1) for off in offsets]
    total_to_download = download_target - existing
    log(
        f"downloading prefix bytes {existing}..{download_target - 1} "
        f"({total_to_download} bytes) as {len(ranges)} range chunks "
        f"of {chunk_size} bytes with {workers} workers"
    )

    started = time.perf_counter()
    downloaded = 0
    completed = 0
    early_stop_reason = None
    samples = []  # (elapsed_seconds, cumulative_downloaded_bytes)
    batch_rates = []
    mode = "ab" if existing > 0 else "wb"
    batches_total = (len(ranges) + workers - 1) // workers

    with open(raw_path, mode) as handle:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, workers)
        ) as pool:
            for batch_index, batch_start in enumerate(
                range(0, len(ranges), workers), start=1
            ):
                batch = ranges[batch_start:batch_start + workers]
                futures = [
                    pool.submit(
                        fetch_range, url, start, end, total_bytes, timeout
                    )
                    for start, end in batch
                ]
                failure = None
                for future in futures:
                    try:
                        body = future.result()
                    except Exception as exc:  # noqa: BLE001 - graceful early stop
                        failure = exc
                        break
                    handle.write(body)
                    downloaded += len(body)
                    completed += 1
                if failure is not None:
                    early_stop_reason = (
                        f"range chunk download failed after retries: {failure}"
                    )
                    log(f"early stop: {early_stop_reason}")
                    break
                handle.flush()
                elapsed = time.perf_counter() - started
                samples.append((elapsed, downloaded))
                overall_mibps = downloaded / max(elapsed, 1e-9) / (1024 ** 2)
                batch_rates.append(round(overall_mibps, 3))

                window_mibps = None
                if elapsed >= rate_window_seconds and len(samples) > 1:
                    base = None
                    for sample in samples[:-1]:
                        if elapsed - sample[0] >= rate_window_seconds:
                            base = sample
                        else:
                            break
                    if base is not None and elapsed > base[0]:
                        window_mibps = (
                            (downloaded - base[1])
                            / (elapsed - base[0])
                            / (1024 ** 2)
                        )

                if not quiet:
                    log(
                        f"batch {batch_index}/{batches_total}: "
                        f"{downloaded / (1024 ** 2):.1f}/"
                        f"{total_to_download / (1024 ** 2):.1f} MiB "
                        f"({100.0 * downloaded / max(total_to_download, 1):.1f}%), "
                        f"aggregate {overall_mibps:.2f} MiB/s"
                        + (
                            f", window {window_mibps:.2f} MiB/s"
                            if window_mibps is not None
                            else ""
                        )
                    )

                if window_mibps is not None and window_mibps < min_rate_mibps:
                    early_stop_reason = (
                        f"aggregate throughput {window_mibps:.2f} MiB/s below "
                        f"{min_rate_mibps} MiB/s over the last "
                        f"{rate_window_seconds:.0f}s; stopped with the bytes "
                        "already obtained"
                    )
                    log(f"early stop: {early_stop_reason}")
                    break

    download_seconds = time.perf_counter() - started
    final_bytes = raw_path.stat().st_size
    expected_final = existing + downloaded
    if final_bytes != expected_final:
        raise RuntimeError(
            f"raw prefix size mismatch: on disk {final_bytes}, "
            f"expected {expected_final}"
        )
    aggregate_mibps = (
        downloaded / download_seconds / (1024 ** 2) if download_seconds > 0 else 0.0
    )
    log(
        f"download finished: {downloaded} bytes in {download_seconds:.2f}s "
        f"({aggregate_mibps:.2f} MiB/s aggregate); prefix bytes={final_bytes}"
    )
    return {
        "resumed_from_byte": existing,
        "downloaded_bytes": downloaded,
        "final_bytes": final_bytes,
        "requested_target_bytes": target_bytes,
        "chunks_requested": len(ranges),
        "chunks_completed": completed,
        "chunk_size_bytes": chunk_size,
        "workers": workers,
        "download_seconds": round(download_seconds, 3),
        "aggregate_mibps": round(aggregate_mibps, 3),
        "early_stop_reason": early_stop_reason,
        "early_stopped": early_stop_reason is not None,
        "throughput_mibps_samples": batch_rates,
        "reused_existing_prefix": False,
    }


# ---------------------------------------------------------------------------
# Streaming parse + projection
# ---------------------------------------------------------------------------


def _as_text(value):
    """Best-effort scalar -> str (JSON strings kept verbatim)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return None


def _id_text(value):
    """Raw product id as a non-empty string (or None)."""
    text = _as_text(value)
    if text is None or not text.strip():
        return None
    return text


def _price_value(value):
    """JSON number or None (bools are not prices)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _description_text(value) -> str:
    if isinstance(value, str):
        return value
    return ""


def _norm_str_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            text = _as_text(item)
            if text is not None:
                out.append(text)
        return out
    text = _as_text(value)
    return [] if text is None else [text]


def _norm_attr_dict(value):
    """dict[str, list[str]] with non-empty string value lists only."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, raw_values in value.items():
        text_key = _as_text(key)
        if not text_key:
            continue
        values = _norm_str_list(raw_values)
        if values:
            out[text_key] = values
    return out


def _norm_options(value):
    """list[dict[str, list[str]]] (non-empty option dicts only)."""
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalised = _norm_attr_dict(item)
        if normalised:
            out.append(normalised)
    return out


def build_contents(product_name: str, attributes, options) -> str:
    """Searchable text: product_name then ``k: v1, v2`` lines.

    Attributes first (insertion order), then every option dict in list order.
    Repeated identical ``k: v1, v2`` lines are de-duplicated (first occurrence
    wins), matching the official Shopping Companion product-doc format.
    Reviews are never included.
    """
    lines = [product_name]
    seen = set()

    def emit(key, values):
        non_empty = [value for value in values if value != ""]
        if not key or not non_empty:
            return
        line = f"{key}: {', '.join(non_empty)}"
        if line in seen:
            return
        seen.add(line)
        lines.append(line)

    for key, values in attributes.items():
        emit(key, values)
    for option in options:
        for key, values in option.items():
            emit(key, values)
    return "\n".join(lines)


def normalize_product(raw_product: dict) -> dict:
    """Project one raw ``record["product"]`` dict into the P14-A product shape.

    Returns ``{"product": <output dict>, "contents": <searchable text>,
    "flags": {field: present}}``.  Unknown / non-JSON-safe fields are never
    copied: only the documented P14-A product keys are emitted.
    """
    attributes = _norm_attr_dict(raw_product.get("attributes"))
    options = _norm_options(raw_product.get("options"))
    product_name = _as_text(raw_product.get("product_name"))
    description = _description_text(raw_product.get("description"))
    price = _price_value(raw_product.get("price"))
    category = _as_text(raw_product.get("category"))
    seller_id = _as_text(raw_product.get("seller_id"))
    seller_name = _as_text(raw_product.get("seller_name"))
    product_id = _as_text(raw_product.get("product_id"))
    contents = (
        build_contents(product_name, attributes, options) if product_name else ""
    )
    product = {
        "product_id": product_id,
        "product_name": product_name,
        "seller_id": seller_id,
        "seller_name": seller_name,
        "category": category,
        "price": price,
        "attributes": attributes,
        "options": options,
        "description": description[:DESCRIPTION_LIMIT],
    }
    flags = {
        "name": bool(product_name and product_name.strip()),
        "category": bool(category),
        "price": price is not None,
        "seller": seller_id is not None or seller_name is not None,
        "attributes": bool(attributes),
        "options": bool(options),
        "description": bool(description),
    }
    return {"product": product, "contents": contents, "flags": flags}


def compute_complete_bytes(raw_path: Path, raw_size: int) -> int:
    """Byte offset just past the last complete (newline-terminated) line."""
    if raw_size <= 0:
        return 0
    with open(raw_path, "rb") as handle:
        handle.seek(raw_size - 1)
        if handle.read(1) == b"\n":
            return raw_size
        block = 1024 * 1024
        position = raw_size
        while position > 0:
            start = max(0, position - block)
            handle.seek(start)
            chunk = handle.read(position - start)
            index = chunk.rfind(b"\n")
            if index != -1:
                return start + index + 1
            position = start
    return 0


def _iter_complete_lines(raw_path: Path, complete_bytes: int):
    """Yield newline-terminated lines from the complete prefix only."""
    consumed = 0
    with open(raw_path, "rb") as handle:
        for raw_line in handle:
            consumed += len(raw_line)
            if consumed > complete_bytes:
                return
            yield raw_line


_EMPTY_FIELD_FLAGS = {
    "name": 0,
    "category": 0,
    "price": 0,
    "seller": 0,
    "attributes": 0,
    "options": 0,
    "description": 0,
}
_EMPTY_COVERAGE = {
    "with_category": 0,
    "with_price": 0,
    "with_seller_name": 0,
    "with_attributes": 0,
    "with_options": 0,
    "with_description": 0,
}


def parse_and_project_prefix(
    raw_path: Path,
    complete_bytes: int,
    docs_path: Path,
    fixture_count: int,
    description_limit: int = DESCRIPTION_LIMIT,
) -> dict:
    """Stream-parse complete lines, project products, write docs.jsonl.

    Never loads the whole prefix: one line at a time, bounded ``seen`` id set.
    Returns the stats dict plus ``fixture_docs`` (first kept projections) and
    ``docs_sha256`` / ``docs_bytes``.
    """
    stats = {
        "raw_lines_read": 0,
        "blank_lines": 0,
        "utf8_decode_errors": 0,
        "malformed_json_lines": 0,
        "malformed_json_samples": [],
        "non_object_records": 0,
        "parsed_records": 0,
        "kept_products": 0,
        "duplicate_ids": 0,
        "skipped_missing_id": 0,
        "skipped_missing_name": 0,
        "skipped_missing_contents": 0,
        "docs_unicode_escape_fallbacks": 0,
    }
    missing_fields = dict(_EMPTY_FIELD_FLAGS)
    field_coverage = dict(_EMPTY_COVERAGE)
    contents_chars_total = 0
    price_null_count = 0
    seen_ids = set()
    fixture_docs = []
    docs_bytes = 0
    docs_digest = hashlib.sha256()
    parse_started = time.perf_counter()
    tmp_docs = _tmp_path(docs_path)
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp_docs, "wb") as docs_handle:
        for raw_line in _iter_complete_lines(raw_path, complete_bytes):
            stats["raw_lines_read"] += 1
            line = raw_line
            if line.endswith(b"\n"):
                line = line[:-1]
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                stats["utf8_decode_errors"] += 1
                text = line.decode("utf-8", "replace")
            try:
                record = json.loads(text)
            except Exception as exc:  # noqa: BLE001 - malformed lines are skipped
                stats["malformed_json_lines"] += 1
                if len(stats["malformed_json_samples"]) < 3:
                    stats["malformed_json_samples"].append(
                        {
                            "line_number": stats["raw_lines_read"],
                            "error": f"{type(exc).__name__}: {exc}"[:200],
                            "sample": text[:300],
                        }
                    )
                continue
            if not isinstance(record, dict):
                stats["non_object_records"] += 1
                continue
            stats["parsed_records"] += 1

            raw_product = record.get("product")
            if not isinstance(raw_product, dict):
                raw_product = {}
            normalised = normalize_product(raw_product)
            flags = normalised["flags"]
            for field in missing_fields:
                if not flags[field]:
                    missing_fields[field] += 1

            product_id = _id_text(record.get("id"))
            if product_id is None:
                stats["skipped_missing_id"] += 1
                continue
            if not flags["name"]:
                stats["skipped_missing_name"] += 1
                continue
            contents = normalised["contents"]
            if not contents.strip():
                stats["skipped_missing_contents"] += 1
                continue
            if product_id in seen_ids:
                stats["duplicate_ids"] += 1
                continue
            seen_ids.add(product_id)

            product = normalised["product"]
            if product["product_id"] is None:
                product["product_id"] = product_id
            doc = {"id": product_id, "contents": contents, "product": product}
            encoded, used_ascii_fallback = _encode_json_line(doc)
            if used_ascii_fallback:
                stats["docs_unicode_escape_fallbacks"] += 1
            docs_handle.write(encoded)
            docs_handle.write(b"\n")
            docs_bytes += len(encoded) + 1
            docs_digest.update(encoded)
            docs_digest.update(b"\n")

            stats["kept_products"] += 1
            contents_chars_total += len(contents)
            if product["price"] is None:
                price_null_count += 1
            for coverage_key in field_coverage:
                field = coverage_key[len("with_"):]
                if product.get(field):
                    field_coverage[coverage_key] += 1
            if len(fixture_docs) < fixture_count:
                fixture_docs.append(doc)

        docs_handle.flush()
        os.fsync(docs_handle.fileno())
    _replace_file(tmp_docs, docs_path)

    parse_seconds = time.perf_counter() - parse_started
    docs_sha256, docs_size = _sha256_file(docs_path)
    if docs_size != docs_bytes:
        raise RuntimeError(
            f"docs.jsonl size mismatch: on disk {docs_size}, wrote {docs_bytes}"
        )
    if docs_sha256 != docs_digest.hexdigest():
        raise RuntimeError("docs.jsonl hash mismatch between write and re-read")

    stats["missing_field_counts"] = missing_fields
    stats["missing_field_definitions"] = {
        "name": "product.product_name absent/empty/non-text",
        "category": "product.category absent/empty/non-text",
        "price": "product.price absent or not a JSON number",
        "seller": "both product.seller_id and product.seller_name absent",
        "attributes": "product.attributes absent/empty after normalisation",
        "options": "product.options absent/empty after normalisation",
        "description": "product.description absent/empty",
    }
    stats["missing_field_counts_scope"] = "all successfully parsed records"
    stats["field_coverage"] = field_coverage
    stats["field_coverage_fractions"] = {
        key: round(value / stats["kept_products"], 4) if stats["kept_products"] else None
        for key, value in field_coverage.items()
    }
    stats["field_coverage_scope"] = "kept products"
    stats["price_null_count"] = price_null_count
    stats["contents_chars_total"] = contents_chars_total
    stats["contents_chars_mean"] = (
        round(contents_chars_total / stats["kept_products"], 2)
        if stats["kept_products"]
        else None
    )
    stats["docs_bytes"] = docs_size
    stats["docs_sha256"] = docs_sha256
    stats["parse_project_seconds"] = round(parse_seconds, 3)
    return {
        "stats": stats,
        "fixture_docs": fixture_docs,
        "docs_path": docs_path,
    }


# ---------------------------------------------------------------------------
# Fixture + manifests
# ---------------------------------------------------------------------------


def write_fixture(fixture_docs, fixture_path: Path, description_limit: int) -> dict:
    """Write the committed test fixture: real projections, description clipped.

    Records are byte-for-byte the projected docs except
    ``product.description`` truncated to ``description_limit`` characters.
    """
    if not fixture_docs:
        raise RuntimeError("no kept products available for the fixture")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(fixture_path)
    digest = hashlib.sha256()
    size = 0
    ids = []
    with open(tmp, "wb") as handle:
        for doc in fixture_docs:
            fixture_doc = {
                "id": doc["id"],
                "contents": doc["contents"],
                "product": dict(doc["product"]),
            }
            fixture_doc["product"]["description"] = (
                fixture_doc["product"].get("description") or ""
            )[:description_limit]
            encoded, _used_ascii_fallback = _encode_json_line(fixture_doc)
            handle.write(encoded)
            handle.write(b"\n")
            digest.update(encoded)
            digest.update(b"\n")
            size += len(encoded) + 1
            ids.append(doc["id"])
        handle.flush()
        os.fsync(handle.fileno())
    _replace_file(tmp, fixture_path)
    on_disk_size = fixture_path.stat().st_size
    if on_disk_size != size:
        raise RuntimeError("fixture size mismatch after write")
    return {
        "path": str(fixture_path),
        "count": len(ids),
        "product_ids": ids,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "description_chars": description_limit,
        "note": (
            "REAL projected products from the beginning of docs.jsonl; "
            "unmodified except product.description truncated to "
            f"{description_limit} characters"
        ),
    }


def build_catalogue_manifest(
    *,
    args,
    probe: dict,
    effective_target_bytes: int,
    raw_path: Path,
    raw_sha256: str,
    raw_bytes: int,
    complete_bytes: int,
    download: dict,
    parse: dict,
    fixture: dict,
    timings: dict,
    docs_path: Path,
) -> dict:
    stats = parse["stats"]
    total = probe["total_bytes"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_method": SELECTION_METHOD,
        "limitation": (
            "LIMITED catalogue: this artifact holds only a contiguous byte "
            "prefix of the official dataset file. It is NOT the full "
            "1,298,797-product catalogue and must not be described as such."
        ),
        "source": {
            "dataset_id": "yuzhan2205/Shopping-companion",
            "revision": "main",
            "hf_mirror": "https://hf-mirror.com",
            "url": args.url,
            "file_path": "data/products.jsonl",
            "total_size_bytes": total,
            "etag": probe.get("etag"),
            "accept_ranges": probe.get("accept_ranges"),
            "probe_method": probe.get("method"),
            "http_status": probe.get("http_status"),
        },
        "prefix": {
            "requested_max_bytes": args.max_bytes,
            "effective_target_bytes": effective_target_bytes,
            "final_bytes": raw_bytes,
            "fraction_of_source_file": (
                round(raw_bytes / total, 6) if total else None
            ),
            "complete_lines_bytes": complete_bytes,
            "trailing_partial_bytes_dropped": raw_bytes - complete_bytes,
            "sha256": raw_sha256,
            "path": str(raw_path),
            "raw_artifact_kept": True,
        },
        "download": {
            "chunk_size_bytes": download["chunk_size_bytes"],
            "workers": download["workers"],
            "chunks_requested": download["chunks_requested"],
            "chunks_completed": download["chunks_completed"],
            "resumed_from_byte": download["resumed_from_byte"],
            "downloaded_bytes": download["downloaded_bytes"],
            "download_seconds": download["download_seconds"],
            "aggregate_mibps": download["aggregate_mibps"],
            "early_stopped": download["early_stopped"],
            "early_stop_reason": download["early_stop_reason"],
            "throughput_mibps_samples": download["throughput_mibps_samples"],
            "reused_existing_prefix": download["reused_existing_prefix"],
        },
        "artifacts": {
            "raw_prefix": {
                "path": str(raw_path),
                "bytes": raw_bytes,
                "sha256": raw_sha256,
            },
            "docs": {
                "path": str(docs_path),
                "bytes": stats["docs_bytes"],
                "sha256": stats["docs_sha256"],
                "lines": stats["kept_products"],
                "format": "one JSON object per line; id+contents+product",
            },
        },
        "stats": stats,
        "searchable_text": {
            "contents_fields_used": [
                "product.product_name",
                "product.attributes",
                "product.options",
            ],
            "contents_format": (
                "product_name then newline-joined 'k: v1, v2' lines for present "
                "attributes and options; repeated identical lines de-duplicated"
            ),
            "payload_only_fields": [
                "product.description (truncated to 1200 chars; not part of contents)"
            ],
        },
        "excluded_fields": [
            "product.good_reviews (never indexed, never copied into docs)",
            "product.bad_reviews (never indexed, never copied into docs)",
            "product.description beyond 1200 characters",
            "raw record['contents'] (rebuilt deterministically from product fields)",
        ],
        "fixture": fixture,
        "timings": timings,
        "environment": {
            "os": platform.platform(),
            "windows_version": platform.version(),
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
        },
        "tool": {
            "path": "experiments/p14a_build_catalogue.py",
            "included_in_commit": True,
        },
    }


def catalogue_essentials(manifest: dict) -> dict:
    """Subset merged into the repo-root ``data/p14a/manifest.json``."""
    stats = manifest["stats"]
    return {
        "schema_version": manifest["schema_version"],
        "generated_at_utc": manifest["generated_at_utc"],
        "selection_method": manifest["selection_method"],
        "limitation": manifest["limitation"],
        "source": {
            "dataset_id": manifest["source"]["dataset_id"],
            "revision": manifest["source"]["revision"],
            "url": manifest["source"]["url"],
            "total_size_bytes": manifest["source"]["total_size_bytes"],
            "etag": manifest["source"]["etag"],
        },
        "product_count": stats["kept_products"],
        "prefix": {
            "requested_max_bytes": manifest["prefix"]["requested_max_bytes"],
            "final_bytes": manifest["prefix"]["final_bytes"],
            "fraction_of_source_file": manifest["prefix"]["fraction_of_source_file"],
            "sha256": manifest["prefix"]["sha256"],
            "path": manifest["prefix"]["path"],
        },
        "docs": manifest["artifacts"]["docs"],
        "raw_prefix": manifest["artifacts"]["raw_prefix"],
        "stats": {
            "raw_lines_read": stats["raw_lines_read"],
            "kept_products": stats["kept_products"],
            "duplicate_ids": stats["duplicate_ids"],
            "malformed_json_lines": stats["malformed_json_lines"],
            "utf8_decode_errors": stats["utf8_decode_errors"],
            "missing_field_counts": stats["missing_field_counts"],
            "field_coverage": stats["field_coverage"],
            "docs_bytes": stats["docs_bytes"],
            "docs_sha256": stats["docs_sha256"],
        },
        "download": {
            "chunk_size_bytes": manifest["download"]["chunk_size_bytes"],
            "workers": manifest["download"]["workers"],
            "download_seconds": manifest["download"]["download_seconds"],
            "aggregate_mibps": manifest["download"]["aggregate_mibps"],
            "early_stopped": manifest["download"]["early_stopped"],
            "early_stop_reason": manifest["download"]["early_stop_reason"],
        },
        "timings": manifest["timings"],
        "fixture": manifest["fixture"],
    }


def merge_root_manifest(path: Path, essentials: dict) -> int:
    """Merge the ``catalogue`` key into the repo-root P14-A manifest."""
    data = {}
    if path.exists():
        try:
            existing_text = path.read_text(encoding="utf-8")
            data = json.loads(existing_text) if existing_text.strip() else {}
        except Exception as exc:  # noqa: BLE001 - never clobber unknown content
            raise RuntimeError(
                f"existing {path} is not valid JSON; refusing to overwrite: {exc!r}"
            )
        if not isinstance(data, dict):
            raise RuntimeError(f"existing {path} is not a JSON object; refusing")
    data["catalogue"] = essentials
    return _write_json_atomic(path, data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LIMITED P14-A Shopping Companion catalogue from a "
            "contiguous byte prefix of products.jsonl."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="products.jsonl URL")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="output directory (raw prefix, docs.jsonl, manifest.json)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="raw prefix byte budget (never the full ~18.7 GiB file by default)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="parallel range requests"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="HTTP Range chunk size in bytes",
    )
    parser.add_argument(
        "--fixture-out",
        default=DEFAULT_FIXTURE_OUT,
        help="committed test fixture output path",
    )
    parser.add_argument(
        "--fixture-count",
        type=int,
        default=DEFAULT_FIXTURE_COUNT,
        help="number of real products in the fixture",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-request HTTP timeout seconds",
    )
    parser.add_argument(
        "--min-rate-mibps",
        type=float,
        default=DEFAULT_MIN_RATE_MIBPS,
        help="early-stop if aggregate throughput drops below this for a while",
    )
    parser.add_argument(
        "--rate-window-seconds",
        type=float,
        default=DEFAULT_RATE_WINDOW_SECONDS,
        help="rolling window for the early-stop throughput check",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete any existing products_prefix.jsonl and re-download",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-batch progress lines"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_bytes <= 0:
        raise SystemExit("--max-bytes must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    total_started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "products_prefix.jsonl"
    docs_path = out_dir / "docs.jsonl"
    manifest_path = out_dir / "manifest.json"
    root_manifest_path = out_dir.parent / "manifest.json"
    fixture_path = Path(args.fixture_out)

    log(f"url={args.url}")
    log(f"out_dir={out_dir} raw={raw_path} docs={docs_path}")
    log(
        f"budget max_bytes={args.max_bytes} workers={args.workers} "
        f"chunk_size={args.chunk_size}"
    )

    probe = probe_source(args.url, args.timeout_seconds)
    total_bytes = probe["total_bytes"]
    effective_target = args.max_bytes
    if total_bytes is not None:
        effective_target = min(args.max_bytes, total_bytes)

    download = download_prefix(
        args.url,
        total_bytes,
        effective_target,
        raw_path,
        chunk_size=args.chunk_size,
        workers=args.workers,
        timeout=args.timeout_seconds,
        min_rate_mibps=args.min_rate_mibps,
        rate_window_seconds=args.rate_window_seconds,
        fresh=args.fresh,
        quiet=args.quiet,
    )

    hash_started = time.perf_counter()
    raw_sha256, raw_bytes = _sha256_file(raw_path)
    if raw_bytes != download["final_bytes"]:
        raise RuntimeError("raw prefix changed between download and hashing")
    complete_bytes = compute_complete_bytes(raw_path, raw_bytes)
    hashing_seconds = time.perf_counter() - hash_started
    log(
        f"raw prefix: {raw_bytes} bytes sha256={raw_sha256} "
        f"complete_lines={complete_bytes} "
        f"trailing_partial={raw_bytes - complete_bytes}"
    )

    parse = parse_and_project_prefix(
        raw_path,
        complete_bytes,
        docs_path,
        fixture_count=max(0, args.fixture_count),
    )
    stats = parse["stats"]
    log(
        f"parsed {stats['parsed_records']} records, kept {stats['kept_products']} "
        f"products, duplicates={stats['duplicate_ids']}, "
        f"malformed={stats['malformed_json_lines']}"
    )
    if stats["kept_products"] == 0:
        raise SystemExit(
            "no products projected from the downloaded prefix; "
            "increase --max-bytes or inspect the raw artifact"
        )

    fixture = write_fixture(
        parse["fixture_docs"], fixture_path, FIXTURE_DESCRIPTION_LIMIT
    )
    log(
        f"fixture: {fixture['count']} products -> {fixture['path']} "
        f"({fixture['bytes']} bytes)"
    )

    total_seconds = time.perf_counter() - total_started
    timings = {
        "download_seconds": download["download_seconds"],
        "hashing_seconds": round(hashing_seconds, 3),
        "parse_project_seconds": stats["parse_project_seconds"],
        "total_wall_seconds": round(total_seconds, 3),
    }
    manifest = build_catalogue_manifest(
        args=args,
        probe=probe,
        effective_target_bytes=effective_target,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        raw_bytes=raw_bytes,
        complete_bytes=complete_bytes,
        download=download,
        parse=parse,
        fixture=fixture,
        timings=timings,
        docs_path=docs_path,
    )
    manifest_bytes = _write_json_atomic(manifest_path, manifest)
    root_manifest_bytes = merge_root_manifest(
        root_manifest_path, catalogue_essentials(manifest)
    )
    log(f"manifest: {manifest_path} ({manifest_bytes} bytes)")
    log(f"root manifest merged: {root_manifest_path} ({root_manifest_bytes} bytes)")

    log("=" * 72)
    log(f"docs_path            : {docs_path}")
    log(f"product_count        : {stats['kept_products']}")
    log(f"docs_bytes           : {stats['docs_bytes']}")
    log(f"docs_sha256          : {stats['docs_sha256']}")
    log(f"raw_prefix_path      : {raw_path}")
    log(f"raw_prefix_bytes     : {raw_bytes}")
    log(f"raw_prefix_sha256    : {raw_sha256}")
    log(f"prefix_of_source     : {manifest['prefix']['fraction_of_source_file']}")
    log(f"download_seconds     : {download['download_seconds']}")
    log(f"download_aggregate   : {download['aggregate_mibps']} MiB/s")
    log(f"parse_project_seconds: {stats['parse_project_seconds']}")
    log(f"early_stop_reason    : {download['early_stop_reason']}")
    log(f"fixture_path         : {fixture_path}")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
