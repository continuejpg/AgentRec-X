"""Build the real BGE-base dense catalogue index (NumPy exact retrieval).

This script is intentionally a one-shot artifact builder, not a serving path.  It reads
the normalized Sports & Outdoors catalogue (JSONL envelope + records), constructs the
trusted dense text with the production module, encodes with the pinned BGE model, and
writes an ``agentrecx.dense_catalog.v3`` NPZ plus a JSON sidecar for audit.

The serving module is loaded by file path when ``--module-file`` is supplied, so the same
script can run in a remote dedicated environment without installing the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("dense_catalog_build", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> dict[str, SimpleNamespace]:
    records: dict[str, SimpleNamespace] = {}
    empty_text = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "parent_asin" not in row:
                continue
            identity = str(row["parent_asin"])
            if identity in records:
                raise SystemExit(f"duplicate catalogue identity: {identity}")
            record = SimpleNamespace(
                parent_asin=identity,
                title=row.get("title"),
                store=row.get("store"),
                main_category=row.get("main_category"),
                categories=tuple(row.get("categories") or ()),
                features=tuple(row.get("features") or ()),
                description=tuple(row.get("description") or ()),
            )
            records[identity] = record
    for record in records.values():
        text = " ".join(
            str(item or "").strip()
            for item in (
                getattr(record, "title", None),
                *getattr(record, "categories", ()),
                *getattr(record, "features", ()),
                *getattr(record, "description", ()),
            )
            if str(item or "").strip()
        )
        if not text:
            empty_text += 1
    print(f"records={len(records)} empty_text={empty_text}", flush=True)
    if not records:
        raise SystemExit("catalogue contains no records")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--module-file", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.module_file is not None:
        module = _load_module(args.module_file)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from recommendation.inference import dense_catalog as module  # type: ignore

    records = _read_records(args.catalogue)
    mapping_sha = _sha256_file(args.mapping)
    print(
        f"building BGE index model={args.model} revision={args.model_revision} "
        f"device={args.device} batch={args.batch_size}",
        flush=True,
    )
    index = module.DenseCatalogIndex.build(
        records,
        model_path=args.model,
        model_revision=args.model_revision,
        mapping_sha=mapping_sha,
        device=args.device,
        batch_size=args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.save(args.out)
    sidecar = {
        "index": str(args.out),
        "index_sha256": _sha256_file(args.out),
        "item_count": index.size,
        "embedding_dim": index.dim,
        "model_id": index.model_id,
        "model_revision": index.model_revision,
        "model_path": index.model_path,
        "query_instruction": index.query_instruction,
        "catalogue_fingerprint": index.catalogue_fingerprint,
        "identity_fingerprint": index.identity_fingerprint,
        "mapping_fingerprint": index.mapping_fingerprint,
        "source_catalogue_sha256": _sha256_file(args.catalogue),
        "source_mapping_sha256": mapping_sha,
        "text_build_version": index.text_build_version,
        "built_at": index.built_at,
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(sidecar, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
