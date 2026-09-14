"""Catalog metadata layer (Milestone 8-A).

Turns the official Amazon Reviews 2023 product-metadata file into a deterministic,
``parent_asin``-keyed artifact, and exposes a narrow read-only lookup over it::

    ProductMetadata / MissingMetadata
        recommendation/catalog/schemas.py
    normalization + artifact + index
        recommendation/catalog/metadata.py

Scope rules:

* the accepted SASRec mapping stays authoritative for ``parent_asin <-> item_id``;
  this package attaches descriptive facts to that identity and never redefines it;
* ``parent_asin`` is treated as an **opaque** string (no ``B`` prefix assumption);
* only fields that really exist in the source are represented; anything the source
  does not supply stays missing and is never inferred;
* a catalog item without metadata is a normal, explicit outcome
  (:class:`~recommendation.catalog.schemas.MissingMetadata`), never a reason to drop
  or substitute a recommendation candidate;
* there is no candidate generation, no product ranking, no reranking and no network
  access in this package.

Metadata never becomes a recommendation signal.  See ``README.md`` for the source
provenance, the normalization policy and the coverage audit.
"""

from __future__ import annotations

from .metadata import (
    ARTIFACT_FORMAT,
    DUPLICATE_POLICY,
    CatalogCoverage,
    MetadataIndex,
    NormalizationOutcome,
    NormalizationReport,
    build_metadata_artifact,
    catalog_item_ids,
    iter_raw_metadata_records,
    load_metadata_index,
    normalize_product_record,
)
from .schemas import (
    MISSING_METADATA_STATUS,
    NORMALIZATION_VERSION,
    MetadataLookup,
    MetadataRecord,
    MissingMetadata,
    ProductMetadata,
)

__all__ = [
    "ARTIFACT_FORMAT",
    "DUPLICATE_POLICY",
    "MISSING_METADATA_STATUS",
    "NORMALIZATION_VERSION",
    "CatalogCoverage",
    "MetadataIndex",
    "MetadataLookup",
    "MetadataRecord",
    "MissingMetadata",
    "NormalizationOutcome",
    "NormalizationReport",
    "ProductMetadata",
    "build_metadata_artifact",
    "catalog_item_ids",
    "iter_raw_metadata_records",
    "load_metadata_index",
    "normalize_product_record",
]

__version__ = "0.1.0"
