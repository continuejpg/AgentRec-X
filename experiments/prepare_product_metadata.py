"""Prepare the normalized catalogue-metadata artifact (Milestone 8-A).

Turns the official Amazon Reviews 2023 product-metadata file for one category into a
deterministic, ``parent_asin``-keyed JSONL artifact plus a processing manifest, and
prints the catalog-coverage audit.

The raw download is never modified.  Raw acquisition is a separate, explicit step so
this script performs **no network access**::

    # 1. acquire the raw metadata once (official source, matching the accepted
    #    dataset provenance used for the review interactions)
    URL=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz
    mkdir -p data/raw/meta_categories
    curl -sS -o data/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz "$URL"
    sha256sum data/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz

    # 2. normalize against the accepted SASRec catalog
    .venv/bin/python -m experiments.prepare_product_metadata

This is data preparation, not evaluation: no recommendation metric is computed and no
recommendation-quality claim is made.

Usage::

    .venv/bin/python -m experiments.prepare_product_metadata
    .venv/bin/python -m experiments.prepare_product_metadata --all-records
    .venv/bin/python -m experiments.prepare_product_metadata --json /tmp/m8a.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config as project_config  # noqa: E402
from recommendation.catalog import build_metadata_artifact  # noqa: E402

#: Official source of the product metadata, matching the review-interaction
#: provenance already documented for this project: Amazon Reviews 2023
#: (McAuley Lab), ``meta_categories/meta_<Category>.jsonl.gz``.
SOURCE_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_{slug}.jsonl.gz"
)

def main(argv: list[str] | None = None) -> int:
    """Build the artifact; returns 0 on success."""
    parser = argparse.ArgumentParser(description="Prepare normalized product metadata (M8-A)")
    parser.add_argument(
        "--raw",
        type=Path,
        default=project_config.default_metadata_path(),
        help="raw official metadata file (.jsonl.gz); must already be downloaded",
    )
    parser.add_argument(
        "--mappings",
        type=Path,
        default=project_config.PROCESSED_DIR
        / f"{project_config.DEFAULT_CATEGORY}_mappings.json",
        help="accepted SASRec mapping artifact defining the catalog scope",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=project_config.default_catalog_metadata_path(),
        help="normalized JSONL artifact destination",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_config.PROCESSED_DIR
        / f"{project_config.metadata_category_slug()}_products_manifest.json",
        help="processing-manifest destination",
    )
    parser.add_argument("--category", default=project_config.DEFAULT_CATEGORY)
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="write every source record, not only those inside the accepted catalog",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 8-A catalogue metadata preparation")
    print("Deterministic normalization. No recommendation metric is computed.")
    print("=" * 78)

    if not args.raw.exists():
        print(f"raw metadata not found: {args.raw}", file=sys.stderr)
        print(
            "\nDownload it once from the official source, then rerun:\n"
            f"  curl -sS -o {args.raw} \\\n"
            f"    {SOURCE_URL_TEMPLATE.format(slug=project_config.metadata_category_slug(args.category))}\n"
            "\nRaw metadata stays immutable and is never committed.",
            file=sys.stderr,
        )
        return 2

    if not args.mappings.exists():
        print(f"accepted mappings not found: {args.mappings}", file=sys.stderr)
        return 2

    print(f"\nraw source      : {args.raw}")
    print(f"catalog mapping : {args.mappings}")
    print(f"artifact        : {args.artifact}")
    print(f"manifest        : {args.manifest}")
    print(f"catalog scope   : {'all source records' if args.all_records else 'accepted catalog only'}")
    print("\nnormalizing (this reads the whole source once) ...", flush=True)

    outcome = build_metadata_artifact(
        args.raw,
        args.artifact,
        mappings_path=None if args.all_records else args.mappings,
        category=args.category,
        source_url=SOURCE_URL_TEMPLATE.format(
            slug=project_config.metadata_category_slug(args.category)
        ),
        manifest_path=args.manifest,
        catalog_only=not args.all_records,
    )

    report = outcome.report
    coverage = outcome.coverage
    envelope = outcome.envelope

    print("\nsource")
    print(f"  raw sha256                : {envelope['source']['raw_sha256']}")
    print(f"  raw bytes                 : {envelope['source']['raw_size_bytes']:,}")
    print(f"  source records seen       : {report.records_seen:,}")
    print(f"  parse errors              : {report.parse_errors:,}")
    print(f"  non-object records        : {report.non_object_records:,}")
    print(f"  missing parent_asin       : {report.missing_parent_asin:,}")
    print(f"  duplicate keys            : {report.duplicate_keys:,} (policy: first wins)")

    matching = coverage.num_catalog_items_with_metadata
    outside = report.records_outside_catalog
    print("\npopulation accounting (raw source vs processed artifact)")
    print(f"  source_records_total              : {report.records_seen:,}")
    print(f"  source_records_matching_catalog   : {matching:,}")
    print(f"  source_records_outside_catalog    : {outside:,}")
    print(f"  processed_catalog_only_records    : {report.normalized_records:,}")
    print(
        "  processed_records_outside_catalog : "
        f"{coverage.num_metadata_records_not_in_catalog:,}"
    )
    print(
        "  matching + outside == total       : "
        f"{matching + outside == report.records_seen} "
        f"({matching:,} + {outside:,} = {matching + outside:,})"
    )
    print(
        "  processed records == catalog match: "
        f"{report.normalized_records == matching}"
    )
    print(
        "\n  Note: 'processed metadata outside catalog = 0' describes the catalog-only\n"
        "  artifact. It is NOT 'raw source metadata outside catalog', which is the large\n"
        "  source count above; that source metadata was deliberately not written."
    )

    print("\ncatalog coverage")
    print(f"  catalog items             : {coverage.num_catalog_items:,}")
    print(f"  metadata records          : {coverage.num_metadata_records:,}")
    print(f"  catalog items with metadata: {coverage.num_catalog_items_with_metadata:,}")
    print(f"  catalog items missing     : {coverage.num_catalog_items_missing_metadata:,}")
    print(f"  coverage                  : {coverage.coverage_percentage:.4f}%")
    print(f"  metadata not in catalog    : {coverage.num_metadata_records_not_in_catalog:,}")

    print("\nartifact")
    print(f"  path                      : {outcome.artifact_path}")
    print(f"  bytes                     : {outcome.artifact_bytes:,}")
    print(f"  sha256                    : {outcome.artifact_sha256}")
    print(f"  normalization version     : {envelope['normalization_version']}")
    print(f"  elapsed                   : {outcome.elapsed_seconds:.1f}s")

    print(
        "\nNote: coverage is a descriptive data statistic about catalogue metadata "
        "availability.\nIt is not a recommendation-quality metric."
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(outcome.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
