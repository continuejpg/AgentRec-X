"""Build versioned enriched retrieval documents for Advanced Retrieval v2.

This is additive: the C4 catalogue and existing BGE index are read, never overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.p14e.catalogue import CATALOG_SOURCE, ElectronicsCatalogue
from recommendation.catalog.enrichment import DEFAULT_SIDECAR
from recommendation.retrieval import (
    DEFAULT_DOCS_PATH, DEFAULT_MANIFEST_PATH, build_manifest,
    write_retrieval_documents,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, default=REPO / CATALOG_SOURCE)
    parser.add_argument("--sidecar", type=Path, default=REPO / DEFAULT_SIDECAR)
    parser.add_argument("--out", type=Path, default=REPO / DEFAULT_DOCS_PATH)
    parser.add_argument("--manifest", type=Path, default=REPO / DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    started = time.perf_counter()
    catalogue = ElectronicsCatalogue.load(
        args.catalogue, metadata_mode="enriched", metadata_sidecar=args.sidecar,
    )
    load_s = round(time.perf_counter() - started, 3)
    build_started = time.perf_counter()
    stats = write_retrieval_documents(catalogue, args.out)
    build_s = round(time.perf_counter() - build_started, 3)
    manifest = build_manifest(
        catalogue_path=args.catalogue, sidecar_path=args.sidecar,
        docs_path=args.out, docs_stats=stats, catalogue=catalogue,
    )
    manifest.update({
        "catalogue_load_seconds": load_s,
        "documents_build_seconds": build_s,
        "catalogue_metadata_mode": catalogue.metadata_mode,
        "catalogue_metadata_stats": catalogue.metadata_stats,
    })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "manifest": str(args.manifest),
        "documents": stats["count"], "chars": manifest["char_stats"],
        "conflicts": stats["conflict_documents"],
        "load_s": load_s, "build_s": build_s,
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
