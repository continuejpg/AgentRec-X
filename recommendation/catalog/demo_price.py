"""Demo-only historical USD -> CNY price adapter for the Electronics catalogue.

Why this exists
---------------
The production C4 catalogue has no price, while the Raw Meta sidecar carries a real,
historical Amazon USD price for a subset of the same ``parent_asin`` identities.  The Web demo
is allowed to convert those historical prices at a **fixed, explicit demo rate** so a budget
can be exercised end to end.

Hard boundaries
---------------
* This adapter is only active in ``AGENTRECX_PRICE_MODE=demo_historical_fx``; the default
  ``strict`` mode never sees a demo price and the existing UNKNOWN/USD behaviour is unchanged.
* Conversion uses :class:`decimal.Decimal` and quantizes to 2 decimal places with
  ``ROUND_HALF_UP``.  It is deterministic and never a live quote.
* The original USD amount, the fixed rate, the source and the converted CNY amount are all
  retained on the fact so a card and a durable replay can state exactly what was used.
* A missing, malformed or non-positive price stays absent (UNKNOWN); nothing is invented.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "DEMO_FX_RATE",
    "DEMO_FX_SOURCE",
    "DEMO_PRICE_MODE",
    "ENV_DEMO_PRICE_SIDECAR",
    "ENV_PRICE_MODE",
    "PRICE_MODES",
    "STRICT_PRICE_MODE",
    "DemoPrice",
    "DemoPriceIndex",
    "read_price_mode",
]

DEMO_PRICE_MODE = "demo_historical_fx"
STRICT_PRICE_MODE = "strict"
PRICE_MODES = (STRICT_PRICE_MODE, DEMO_PRICE_MODE)
ENV_PRICE_MODE = "AGENTRECX_PRICE_MODE"
ENV_DEMO_PRICE_SIDECAR = "AGENTRECX_DEMO_PRICE_SIDECAR"
DEMO_FX_RATE = Decimal("7.00")
DEMO_FX_SOURCE = "demo_fixed_rate"
DEMO_PRICE_NOTE = "历史价格按固定汇率换算，非实时售价（1 USD = 7.00 CNY）"
SIDECAR_SCHEMA = "agentrecx.electronics_metadata_sidecar.v1"
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_SIDECAR = _REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"
_CENT = Decimal("0.01")


def read_price_mode(value: str | None = None) -> str:
    """Return the validated price mode; only the exact demo value enables conversion."""
    raw = str(value if value is not None else os.environ.get(ENV_PRICE_MODE, STRICT_PRICE_MODE))
    mode = raw.strip().casefold()
    if not mode:
        mode = STRICT_PRICE_MODE
    if mode not in PRICE_MODES:
        raise ValueError(f"{ENV_PRICE_MODE} must be one of {PRICE_MODES}, got {raw!r}")
    return mode


def _usd_amount(value: Any) -> Decimal | None:
    """Parse the sidecar's historical USD amount without inventing anything."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip().replace(",", "").replace("$", "").strip()
    else:
        return None
    if not text or text.casefold() in {"none", "null", "nan", "inf", "-inf"}:
        return None
    try:
        amount = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount <= 0:
        return None
    return amount


def _cny(amount: Decimal, rate: Decimal = DEMO_FX_RATE) -> Decimal:
    return (amount * rate).quantize(_CENT, rounding=ROUND_HALF_UP)


def _money(value: Decimal) -> str:
    return f"{value.quantize(_CENT, rounding=ROUND_HALF_UP):.2f}"


@dataclass(frozen=True)
class DemoPrice:
    """A real historical USD price with its deterministic demo CNY conversion."""

    item_id: str
    usd_amount: Decimal
    cny_amount: Decimal
    source: str
    dataset_version: str = ""
    fx_rate: Decimal = DEMO_FX_RATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "usd_amount": _money(self.usd_amount),
            "cny_amount": _money(self.cny_amount),
            "fx_rate": _money(self.fx_rate),
            "fx_source": DEMO_FX_SOURCE,
            "source": self.source,
            "dataset_version": self.dataset_version,
            "price_kind": "historical_crawl_time",
            "historical_time": None,
            "historical_time_note": (
                "the Raw Meta sidecar has no per-item crawl timestamp; "
                f"dataset_version={self.dataset_version or 'unknown'}"
            ),
            "price_mode": DEMO_PRICE_MODE,
            "note": DEMO_PRICE_NOTE,
        }

    def as_fact(self) -> dict[str, Any]:
        """Return a catalogue ``inspect(price)`` fact that the budget Gate can verify."""
        return {
            "status": "historical_metadata",
            "value": float(self.cny_amount),
            "currency": "CNY",
            "price_kind": "historical_crawl_time",
            "source": self.source,
            "verification": "upstream_metadata_demo_fx",
            "demo_price_mode": DEMO_PRICE_MODE,
            "original_usd_price": float(self.usd_amount),
            "original_currency": "USD",
            "converted_cny_price": _money(self.cny_amount),
            "fx_rate": _money(self.fx_rate),
            "fx_source": DEMO_FX_SOURCE,
            "dataset_version": self.dataset_version,
            "historical_time": None,
            "historical_time_note": (
                "the Raw Meta sidecar has no per-item crawl timestamp; "
                f"dataset_version={self.dataset_version or 'unknown'}"
            ),
            "price_note": DEMO_PRICE_NOTE,
            "reason": (
                "historical Amazon USD price converted at the fixed demo rate "
                f"1 USD = {_money(self.fx_rate)} CNY; not a live price or availability"
            ),
        }

    def evidence_text(self) -> str:
        """The exact text used as a price evidence span (kept with the original USD)."""
        return (
            f"Historical price {_money(self.usd_amount)} USD -> "
            f"{_money(self.cny_amount)} CNY at fixed demo rate 1 USD = "
            f"{_money(self.fx_rate)} CNY; not a live price"
        )


@dataclass
class DemoPriceIndex:
    """Historical USD prices for catalogue identities, converted deterministically."""

    records: dict[str, DemoPrice] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_DEMO_SIDECAR,
        *,
        item_ids: Iterable[str],
        fx_rate: Decimal = DEMO_FX_RATE,
    ) -> "DemoPriceIndex":
        wanted = {str(item).strip() for item in item_ids if str(item).strip()}
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"demo price sidecar not found: {source}")
        rate = Decimal(fx_rate)
        if not rate.is_finite() or rate <= 0:
            raise ValueError("demo FX rate must be a positive finite Decimal")
        records: dict[str, DemoPrice] = {}
        rows = 0
        matched = 0
        missing_price = 0
        invalid_price = 0
        duplicates = 0
        versions: set[str] = set()
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_price += 1
                    continue
                if not isinstance(row, Mapping):
                    invalid_price += 1
                    continue
                item_id = str(row.get("item_id") or "").strip()
                if item_id not in wanted:
                    continue
                matched += 1
                if item_id in records:
                    duplicates += 1
                    continue
                usd = _usd_amount(row.get("price"))
                if usd is None:
                    if row.get("price") in (None, ""):
                        missing_price += 1
                    else:
                        invalid_price += 1
                    continue
                version = str(row.get("dataset_version") or "")
                if version:
                    versions.add(version)
                records[item_id] = DemoPrice(
                    item_id=item_id,
                    usd_amount=usd,
                    cny_amount=_cny(usd, rate),
                    source=str(row.get("source") or "amazon_reviews_2023:raw_meta_Electronics"),
                    dataset_version=version,
                    fx_rate=rate,
                )
        stats = {
            "mode": DEMO_PRICE_MODE,
            "path": str(source),
            "fx_rate": _money(rate),
            "fx_source": DEMO_FX_SOURCE,
            "rows_scanned": rows,
            "catalogue_ids": len(wanted),
            "matched_records": matched,
            "priced_records": len(records),
            "missing_price": missing_price,
            "invalid_price": invalid_price,
            "duplicate_ids": duplicates,
            "match_rate": round(matched / len(wanted), 6) if wanted else 0.0,
            "priced_rate": round(len(records) / len(wanted), 6) if wanted else 0.0,
            "dataset_versions": sorted(versions),
        }
        return cls(records=records, stats=stats)

    def get(self, item_id: str) -> DemoPrice | None:
        return self.records.get(str(item_id or "").strip())

    def __len__(self) -> int:
        return len(self.records)

    def coverage(self, ids: Iterable[str]) -> dict[str, Any]:
        wanted = {str(item) for item in ids}
        matched = wanted.intersection(self.records)
        return {
            "catalogue_ids": len(wanted),
            "priced_records": len(matched),
            "priced_rate": round(len(matched) / len(wanted), 6) if wanted else 0.0,
        }
