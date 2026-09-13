"""Configuration for the AgentRec-X recommendation data pipeline.

This module holds only *data* (paths, schema field names, defaults) so that every
other module can import it without creating cycles.  Behaviour lives in
``io_utils.py`` and ``preprocess.py``.

Milestone 1 scope: preprocessing only.  No model code lives here.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #

#: Repository root (``<repo>/recommendation/config.py`` -> ``<repo>``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Root of all data.  ``data/`` is git-ignored (see .gitignore): raw Amazon
#: archives and generated processed artifacts must never be committed.
DATA_DIR = Path(os.environ.get("AGENTRECX_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

#: Target Amazon Reviews 2023 category for Milestone 1.
DEFAULT_CATEGORY = "Sports_and_Outdoors"

# --------------------------------------------------------------------------- #
# Amazon Reviews 2023 interaction (review) schema
# --------------------------------------------------------------------------- #
#
# Source of truth: the official Amazon Reviews 2023 dataset card
# (McAuley-Lab/Amazon-Reviews-2023) and https://amazon-reviews-2023.github.io/.
# The "User Reviews" record schema is documented as:
#
#   rating            float  1.0-5.0
#   title             str    review title
#   text              str    review body
#   images            list   user-posted images
#   asin              str    ID of the product (variant-level)
#   parent_asin       str    parent ID of the product
#   user_id           str    reviewer ID
#   timestamp         int    unix time
#   verified_purchase bool
#   helpful_vote      int
#
# Field mapping for sequential recommendation:
#
#   user  <- user_id
#   item  <- parent_asin
#   time  <- timestamp
#   label <- rating
#
# Why ``parent_asin`` and not ``asin``: the dataset card states that products
# differing only by colour/style/size share one parent ID, and that the ``asin``
# column of *previous* Amazon datasets was in fact the parent ID, adding
# "Please use parent ID to find product meta".  Using ``parent_asin`` therefore
# (a) collapses variants of the same product into a single catalogue item, and
# (b) keeps this pipeline directly comparable with prior Amazon benchmarks that
# were built on parent-level item IDs.  ``parent_asin`` is thus the canonical
# item identifier here; ``asin`` is intentionally ignored.
# --------------------------------------------------------------------------- #

#: Canonical schema keys used by the pipeline (physical names in the raw JSON).
FIELD_USER = "user_id"
FIELD_ITEM = "parent_asin"
FIELD_TIMESTAMP = "timestamp"
FIELD_RATING = "rating"

#: Fields a raw record must contain to be usable.  Everything else is ignored.
REQUIRED_FIELDS = (FIELD_USER, FIELD_ITEM, FIELD_TIMESTAMP, FIELD_RATING)

#: Fields the pipeline keeps in memory; the rest of the raw record is dropped.
#: Review text/images/metadata are explicitly out of scope for Milestone 1.
KEPT_FIELDS = REQUIRED_FIELDS

#: Plausible lower bound for review timestamps (May 1996, from the dataset
#: card).  Used only to *warn* about suspicious values, never to reject rows:
#: the raw files store unix time, and some mirrors store seconds instead of
#: milliseconds, so both magnitudes are accepted.
MIN_PLAUSIBLE_TIMESTAMP_SECONDS = 833_000_000  # 1996-05-23

# --------------------------------------------------------------------------- #
# Preprocessing defaults
# --------------------------------------------------------------------------- #

#: Minimum interactions per user / per item (k-core thresholds).
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

#: Hard stop for the iterative k-core loop, so a pathological input cannot spin
#: forever.  Each round strictly shrinks the graph, so convergence is expected
#: in a handful of rounds; this is a safety net, not a tuning knob.
MAX_K_CORE_ROUNDS = 100

#: How to treat records that repeat the same (user, item, timestamp) triple.
#: ``"last"`` keeps the final occurrence (a re-imported/updated review row),
#: ``"first"`` keeps the first, ``"keep"`` disables deduplication entirely.
DEDUPLICATE_POLICY = "last"
VALID_DEDUPLICATE_POLICIES = ("last", "first", "keep")

# --------------------------------------------------------------------------- #
# ID conventions
# --------------------------------------------------------------------------- #

#: Integer ID reserved for padding.  No real item (or user) may receive it, so
#: embedding tables can safely use this index for PAD.
PAD_ID = 0

#: First integer ID handed out to a real entity (item or user).
FIRST_REAL_ID = 1

# --------------------------------------------------------------------------- #
# Artifact names
# --------------------------------------------------------------------------- #


def artifact_paths(
    category: str = DEFAULT_CATEGORY,
    processed_dir: Path | None = None,
) -> dict[str, Path]:
    """Return the canonical output paths for ``category``.

    Returns a dict with keys ``sequences``, ``mappings`` and ``metadata``.
    """
    out_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    return {
        "sequences": out_dir / f"{category}_sequences.json",
        "mappings": out_dir / f"{category}_mappings.json",
        "metadata": out_dir / f"{category}_metadata.json",
    }


#: Conventional raw input path for the target category, mirroring the official
#: download layout (``review_categories/<Category>.jsonl.gz``).
def default_raw_path(category: str = DEFAULT_CATEGORY, raw_dir: Path | None = None) -> Path:
    """Return the conventional raw review path for ``category``."""
    base = Path(raw_dir) if raw_dir is not None else RAW_DIR
    return base / "review_categories" / f"{category}.jsonl.gz"
