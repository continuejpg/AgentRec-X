"""Build the frozen TF-IDF similar-item index once and persist it for the query pass.

Why this is a separate step
---------------------------
The index build peaks at ~5.93 GiB on this host, and the query pass also needs the frozen
``parent_asin`` <-> ``item_id`` mapping. Building and querying in one process would leave no
headroom on a 6.9 GiB budget (preregistration §12.3 records the OOM that already occurred once).
Building once and streaming the query pass from the persisted vectors keeps each phase inside its
own envelope, and does not change what is retrieved: the vectors are the same objects the frozen
:class:`~recommendation.control.similar_item.SimilarItemIndex` builds.

The artefact records the source hashes it was built from, so the index contents are reproducible
from the frozen inputs and cannot drift silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import resource
import sys
import time
from typing import Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from recommendation.catalog.metadata import MetadataIndex  # noqa: E402
from recommendation.control.similar_item import build_similar_item_index  # noqa: E402

OUT = REPO / "runs/m4_evidence"
INDEX_PKL = OUT / "similar_item_index.pkl"
INDEX_META = OUT / "similar_item_index.json"

PRODUCTS = REPO / "data/processed/Sports_and_Outdoors_products.jsonl"


def peak_rss_gib() -> float:
    """Peak resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def sha256(path: pathlib.Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> None:
    """Build the index and persist the objects the query pass needs."""
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    catalogue = MetadataIndex.load(PRODUCTS)
    load_seconds = time.time() - started
    build_started = time.time()
    index = build_similar_item_index(catalogue)
    build_seconds = time.time() - build_started

    # The pickled payload is exactly the frozen index's own state: identities, weights, idf and
    # norms. `_vectors` values are plain dicts of float, so the file is portable across processes.
    payload = {
        "identities": tuple(index.identities),
        "position": {identity: i for i, identity in enumerate(index.identities)},
        "vectors": index._vectors,  # noqa: SLF001 - persisting the frozen object's own state
        "norms": index._norms,  # noqa: SLF001
        "idf": index._idf,  # noqa: SLF001
        "document_frequency": dict(index._document_frequency),  # noqa: SLF001
    }
    with open(INDEX_PKL, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "artefact": str(INDEX_PKL.relative_to(REPO)),
        "sha256": sha256(INDEX_PKL),
        "size_bytes": INDEX_PKL.stat().st_size,
        "vocabulary_size": index.vocabulary_size,
        "documents": len(index.identities),
        "catalogue_load_seconds": load_seconds,
        "index_build_seconds": build_seconds,
        "peak_rss_gib": peak_rss_gib(),
        "inputs": {
            "products": {
                "path": str(PRODUCTS.relative_to(REPO)),
                "sha256": sha256(PRODUCTS),
            }
        },
    }
    INDEX_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"index built: vocab={index.vocabulary_size} docs={len(index.identities)} "
        f"load={load_seconds:.1f}s build={build_seconds:.1f}s peakRSS={peak_rss_gib():.2f} GiB",
        flush=True,
    )
    print(f"  -> {INDEX_PKL}  sha256={meta['sha256']}", flush=True)
    print(f"  -> {INDEX_META}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the frozen similar-item index")
    parser.parse_args(argv)
    build()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
