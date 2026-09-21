#!/usr/bin/env python
"""Generate the small, offline demo catalog used by the one-command Docker run.

Why this exists
---------------
The accepted runtime artifacts total ~832 MB and are deliberately **not** in Git (the raw
Amazon corpus is multi-GB; checkpoints and processed catalogs are regenerable and belong
outside version control).  A reviewer who clones this repository therefore cannot start the
real server without first downloading and preprocessing a public dataset - which is exactly
what a "quick start" must not require.

So the packaging phase ships a **small, synthetic, checked-in demo catalog** and this script
that materialises the artifacts the serving stack expects from it.  It writes:

* ``recommendation/demo/artifacts/demo_products.jsonl``  - hand-written product records
* ``runs/demo_catalog/checkpoint.pt``                   - a tiny, untrained SASRec checkpoint
* ``runs/demo_catalog/mappings.json``                   - the matching ``parent_asin`` map

``runs/`` is git-ignored, so the generated artifacts stay out of version control while the
catalog they are built from is reviewable in the repository.

What this deliberately is not
-----------------------------
This is **not** a benchmark artifact and it is **not** evidence about recommendation
quality.  The checkpoint is randomly initialised: its scores are meaningless and its purpose
is to make the *plumbing* - grounding, provenance, constraints, the bounded loop, the HTTP
contract - runnable and inspectable offline.  Every measured number in this repository comes
from the accepted checkpoints and the public dataset, and the README says so at the point of
use.

Usage::

    python -m experiments.build_demo_catalog            # write into runs/demo_catalog
    python -m experiments.build_demo_catalog --out DIR   # somewhere else
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Where the checked-in catalog lives.  The *catalog* is committed; the generated
#: checkpoint and mappings are not.
DEMO_CATALOG_PATH = REPO_ROOT / "recommendation" / "demo" / "artifacts" / "demo_products.jsonl"

#: Default generated-artifact directory (git-ignored, like every other experiment output).
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "demo_catalog"

#: The synthetic catalog.  Identities are deliberately opaque: ``parent_asin`` is treated as an
#: opaque string everywhere in this codebase, and a fixture that looked like a real Amazon id
#: would invite assumptions the code must not make.
DEMO_PRODUCTS: tuple[dict[str, Any], ...] = (
    {
        "parent_asin": "demo-trail-runner",
        "title": "Trail Runner Hiking Shoe",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Hiking"],
        "features": ["waterproof mesh upper", "grippy rubber outsole", "lightweight"],
        "description": ["A lightweight trail running shoe for wet hiking trails."],
        "price_text": "119.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Slate", "Material": "Mesh", "Weight": "280 g"},
    },
    {
        "parent_asin": "demo-alpine-boot",
        "title": "Alpine Hiking Boot",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Hiking"],
        "features": ["full-grain leather", "waterproof membrane", "ankle support"],
        "description": ["A sturdy leather boot for alpine hiking in cold weather."],
        "price_text": "189.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Brown", "Material": "Leather", "Weight": "640 g"},
    },
    {
        "parent_asin": "demo-canyon-boot",
        "title": "Canyon Leather Hiking Boot",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Hiking"],
        "features": ["waterproof leather", "cushioned midsole", "ankle support"],
        "description": ["A leather hiking boot for dry canyon trails."],
        "price_text": "149.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Brown", "Material": "Leather", "Weight": "560 g"},
    },
    {
        "parent_asin": "demo-rain-shell",
        "title": "Stormline Rain Shell Jacket",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Clothing"],
        "features": ["waterproof", "breathable", "packable"],
        "description": ["A packable waterproof shell for wet weather hiking."],
        "price_text": "139.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Slate", "Material": "Nylon", "Weight": "310 g"},
    },
    {
        "parent_asin": "demo-fleece",
        "title": "Ridgeline Fleece Pullover",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Clothing"],
        "features": ["midlayer", "breathable", "warm"],
        "description": ["A warm fleece midlayer for cold mornings on the trail."],
        "price_text": "79.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Green", "Material": "Polyester", "Weight": "420 g"},
    },
    {
        "parent_asin": "demo-daypack",
        "title": "Summit 24L Daypack",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Backpacks"],
        "features": ["24 litre", "hydration sleeve", "rain cover"],
        "description": ["A compact daypack for half-day hikes."],
        "price_text": "89.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Slate", "Material": "Nylon", "Weight": "740 g"},
    },
    {
        "parent_asin": "demo-trek-pack",
        "title": "Continental 45L Trekking Pack",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Backpacks"],
        "features": ["45 litre", "load lifter straps", "rain cover"],
        "description": ["A larger pack for multi-day trekking."],
        "price_text": "199.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Green", "Material": "Nylon", "Weight": "1480 g"},
    },
    {
        "parent_asin": "demo-trek-pole",
        "title": "Basalt Trekking Pole Pair",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Hiking"],
        "features": ["collapsible", "cork grip", "lightweight"],
        "description": ["A collapsible pair of trekking poles for steep descents."],
        "price_text": "69.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Black", "Material": "Aluminium", "Weight": "480 g"},
    },
    {
        "parent_asin": "demo-water-bottle",
        "title": "Cascade Insulated Bottle 1L",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Hydration"],
        "features": ["insulated", "1 litre", "leakproof"],
        "description": ["An insulated bottle that keeps water cold all day."],
        "price_text": "39.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Blue", "Material": "Steel", "Weight": "520 g"},
    },
    {
        "parent_asin": "demo-headlamp",
        "title": "Nightbeam 400 Headlamp",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Lighting"],
        "features": ["400 lumen", "rechargeable", "waterproof"],
        "description": ["A rechargeable headlamp for early starts and late finishes."],
        "price_text": "49.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Black", "Material": "Plastic", "Weight": "110 g"},
    },
    {
        "parent_asin": "demo-first-aid",
        "title": "Trailside First Aid Kit",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Safety"],
        "features": ["compact", "waterproof case", "trail essentials"],
        "description": ["A compact first aid kit for day hikes."],
        "price_text": "29.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Red", "Material": "Nylon", "Weight": "260 g"},
    },
    {
        "parent_asin": "demo-sleeping-bag",
        "title": "Bivouac 20F Sleeping Bag",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Camping"],
        "features": ["20F rating", "compressible", "water-resistant shell"],
        "description": ["A compressible sleeping bag rated to 20F for three-season camping."],
        "price_text": "219.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Green", "Material": "Down", "Weight": "1100 g"},
    },
    {
        "parent_asin": "demo-stove",
        "title": "Pocket Canister Stove",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Camping"],
        "features": ["ultralight", "piezo ignition", "carry case"],
        "description": ["An ultralight canister stove for backpacking meals."],
        "price_text": "59.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Silver", "Material": "Steel", "Weight": "95 g"},
    },
    {
        "parent_asin": "demo-gaiters",
        "title": "Mudline Trail Gaiters",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Hiking"],
        "features": ["water-resistant", "ankle height", "strap fit"],
        "description": ["Ankle gaiters that keep mud and scree out of boots."],
        "price_text": "34.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Black", "Material": "Nylon", "Weight": "150 g"},
    },
    {
        "parent_asin": "demo-socks",
        "title": "Drysock Merino Crew Socks",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Clothing"],
        "features": ["merino wool", "cushioned", "moisture wicking"],
        "description": ["Cushioned merino socks that stay warm when wet."],
        "price_text": "24.00",
        "source": "agentrecx.demo:handwritten",
        "details": {"Color": "Grey", "Material": "Merino", "Weight": "90 g"},
    },
)

#: The demo profiles the served runtime offers, expressed as history over the catalog above.
#: A profile is application-owned history, not a user record: two profiles share catalog
#: products so a cross-profile comparison is observable.
DEMO_PROFILES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "demo-hiker",
        "Weekend hiker",
        ("demo-socks", "demo-trail-runner", "demo-fleece", "demo-daypack"),
    ),
    (
        "demo-alpinist",
        "Alpine trekker",
        ("demo-socks", "demo-alpine-boot", "demo-rain-shell", "demo-trek-pack"),
    ),
    (
        "demo-camper",
        "Camping shopper",
        ("demo-stove", "demo-sleeping-bag", "demo-first-aid", "demo-headlamp"),
    ),
)


def catalog_records() -> list[dict[str, Any]]:
    """The demo catalog as fresh copies, in file order."""
    return [dict(record) for record in DEMO_PRODUCTS]


def write_catalog(path: Path) -> Path:
    """Write the reviewed catalog artifact.

    The envelope is the accepted catalogue artifact's envelope (same ``format`` tag), because
    the accepted :meth:`~recommendation.catalog.MetadataIndex.load` reads exactly that tag and
    a second loader would be a second read path.  The ``category`` and ``description`` fields
    say plainly what this file is, so nothing here can be mistaken for the Amazon catalogue.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = catalog_records()
    payload = {
        "format": "agentrecx.catalog_products.v1",
        "category": "synthetic demo catalogue (NOT Amazon Reviews 2023)",
        "description": (
            "Hand-written products for the offline demo. NOT the Amazon Reviews 2023 "
            "catalogue and not used by any benchmark."
        ),
        "counts": {"records": len(records)},
    }
    with path.open("wt", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def build_artifacts(
    *, out_dir: Path, catalog_path: Path = DEMO_CATALOG_PATH, seed: int = 2026
) -> dict[str, Any]:
    """Generate the checkpoint and mappings the serving stack needs.

    The checkpoint is a real :class:`~recommendation.models.sasrec.SASRec` in the accepted
    serving format, randomly initialised from a fixed seed.  It is untrained on purpose: a
    trained model would be a benchmark claim, and this artifact exists only so the plumbing
    runs offline.
    """
    import torch

    from recommendation.models.sasrec import SASRec, SASRecConfig
    from recommendation.training.checkpoint import CHECKPOINT_FORMAT

    records = catalog_records()
    num_items = len(records)
    max_seq_len = 20
    config = SASRecConfig(
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_size=32,
        num_blocks=1,
        num_heads=2,
        dropout=0.1,
    )

    torch.manual_seed(seed)
    model = SASRec(config)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "seed": seed,
            "num_items": num_items,
            "max_seq_len": max_seq_len,
            "model_config": {
                "num_items": num_items,
                "max_seq_len": max_seq_len,
                "hidden_size": config.hidden_size,
                "num_blocks": config.num_blocks,
                "num_heads": config.num_heads,
                "dropout": config.dropout,
            },
            "model_state_dict": model.state_dict(),
            "note": (
                "synthetic, untrained demo checkpoint generated by "
                "experiments/build_demo_catalog.py; carries no recommendation quality claim"
            ),
        },
        checkpoint_path,
    )

    mappings_path = out_dir / "mappings.json"
    mappings_path.write_text(
        json.dumps(
            {
                "padding": 0,
                "num_items": num_items,
                "item2id": {
                    record["parent_asin"]: index
                    for index, record in enumerate(records, start=1)
                },
                "id2item": [None] + [record["parent_asin"] for record in records],
                "note": "synthetic demo mappings; not a preprocessing-run artifact",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "catalog": str(catalog_path),
        "checkpoint": str(checkpoint_path),
        "mappings": str(mappings_path),
        "num_items": num_items,
        "hidden_size": config.hidden_size,
        "max_seq_len": max_seq_len,
        "trained": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the small offline demo catalog")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEMO_CATALOG_PATH,
        help="where to write the reviewed catalog artifact",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    catalog = write_catalog(args.catalog)
    built = build_artifacts(out_dir=args.out, catalog_path=catalog)
    if not args.quiet:
        print("demo catalog built (synthetic, untrained - no quality claim)")
        for key, value in built.items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
