"""Materialise the **generator** token layout from the accepted Semantic-ID layout (Step 2.5).

Why this module exists
----------------------
The accepted Step-2.4F artifact is immutable.  Its ``sid/layout.json`` declares three special
tokens (``pad``, ``bos``, ``eos``) and ``vocab_size = 1027``; it was produced before the generator
needed an explicit item-boundary token.  The generator does need one, and deriving its id at model
use time would be unsafe: a guessed id above the code space could alias a real code, and nothing
would be hashed for a checkpoint to bind.

So the generator's token space is a **separate, derived artifact**:

```text
accepted sid/layout.json  --read-only-->  generator_layout.json  --hashed-->  checkpoint dep
```

The accepted file is opened for reading only.  Nothing in this module writes inside ``sid/``; the
caller names an output path, and the derived artifact records the SHA256 of both the accepted
layout and the accepted Semantic IDs, so the derivation is auditable rather than implicit.

Two rules the module enforces
-----------------------------
1. **No catalogue SID changes.**  The derivation copies ``levels``, ``dedup_levels``,
   ``codebook_size``, ``dedup_vocab_size`` and ``level_offsets`` verbatim and never touches the
   assignment.  A test asserts the accepted artifact is byte-identical after a run.
2. **SEP cannot alias a code.**  Every special token is required to sit at or above
   ``code_space``, to be pairwise distinct, and to be absent from every digit actually used by the
   accepted Semantic IDs.

Two accepted shapes are recognised, because they are the two that exist: the Step-2.4F production
artifact (``pad``/``bos``/``eos``, ``vocab_size = code_space + 3``) and a layout the backend's own
``fit-sid`` writes after SEP was registered (already carrying ``sep``, ``vocab_size = code_space +
4``).  In both cases the result is the same generator vocabulary; only the provenance field
``source_sid_layout_had_sep`` differs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "GENERATOR_LAYOUT_FORMAT",
    "GeneratorLayoutError",
    "assert_no_code_alias",
    "derive_generator_layout",
    "load_generator_layout",
    "materialise_generator_layout",
    "sha256_file",
]

#: The derived artifact's format tag.  Distinct from ``agentrecx.tiger.token_layout.v3`` because it
#: describes a *different* object: the generator's vocabulary, not the accepted SID layout.
GENERATOR_LAYOUT_FORMAT = "agentrecx.tiger.generator_layout.v1"

#: Special tokens the generator vocabulary must declare, in the order they are appended above the
#: catalogue code space.
SPECIAL_ORDER = ("pad", "bos", "eos", "sep")


class GeneratorLayoutError(ValueError):
    """Raised when the accepted layout cannot yield a safe generator vocabulary."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(payload: Mapping[str, Any], name: str) -> int:
    if name not in payload:
        raise GeneratorLayoutError(f"the accepted layout declares no {name!r}")
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeneratorLayoutError(f"{name} must be an int, got {value!r}")
    return int(value)


def assert_no_code_alias(
    layout: Mapping[str, Any], assignment: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Prove no special token can collide with a catalogue code or with another special.

    Three independent checks, because they fail in different ways:

    * ``special >= code_space`` - a special *below* the code space would silently shadow a real
      code on every level that reaches it;
    * pairwise distinctness - two specials sharing an id would make BOS and SEP indistinguishable;
    * absence from the used digits - a special equal to a digit that the accepted Semantic IDs
      actually map onto the same token id only if that digit is on the level whose offset puts it
      there, which is why the check is on *token ids*, not on digit values.

    ``assignment[0]`` is the reserved PAD sentinel row and is skipped: it is not a catalogue item.
    """
    levels = _require_int(layout, "levels")
    dedup_levels = _require_int(layout, "dedup_levels")
    codebook_size = _require_int(layout, "codebook_size")
    dedup_vocab_size = int(layout.get("dedup_vocab_size", codebook_size))
    offsets = list(layout["level_offsets"])
    code_space = _require_int(layout, "code_space")
    specials = layout["special"]

    width = levels + dedup_levels
    if len(offsets) != width:
        raise GeneratorLayoutError(
            f"the accepted layout declares {len(offsets)} offsets for {width} levels"
        )
    # The blocks are ``offset[l]`` wide enough for their own level, so the code space must cover
    # the end of every block.  The dedup block may legitimately be narrower or wider than the
    # semantic blocks (the backend's own fit-sid writes a 32-wide codebook beside a 256-wide dedup
    # block), so the last block end is the binding constraint rather than ``levels x codebook``.
    for index, offset in enumerate(offsets):
        if int(offset) != index * codebook_size:
            raise GeneratorLayoutError(
                f"level {index} offset {offset} is not {index} x {codebook_size}; the offsets "
                "must be the layout's own level blocks"
            )
    last_block_end = int(offsets[-1]) + (dedup_vocab_size if dedup_levels else codebook_size)
    if code_space < last_block_end:
        raise GeneratorLayoutError(
            f"code_space {code_space} does not cover the last level block "
            f"({offsets[-1]} + "
            f"{dedup_vocab_size if dedup_levels else codebook_size} = {last_block_end})"
        )

    values = [int(specials[name]) for name in SPECIAL_ORDER]
    if len(set(values)) != len(values):
        raise GeneratorLayoutError(f"special tokens are not pairwise distinct: {values}")

    low = min(offsets[0], 0)
    for name, token in zip(SPECIAL_ORDER, values, strict=True):
        if token < code_space:
            raise GeneratorLayoutError(
                f"{name} token {token} is inside the catalogue code space [0, {code_space}); it "
                "would shadow a real SID code"
            )
    if low < 0:
        raise GeneratorLayoutError(f"a level offset is negative ({low})")

    # Every token the accepted assignment can actually produce.
    used: set[int] = set()
    for codes in assignment[1:]:
        if len(codes) != width:
            raise GeneratorLayoutError(
                f"a Semantic ID has {len(codes)} digits, expected {width} for this layout"
            )
        for index, digit in enumerate(codes):
            value = int(digit)
            size = codebook_size if index < levels else dedup_vocab_size
            if not 0 <= value < size:
                raise GeneratorLayoutError(
                    f"digit {index} = {value} is outside [0, {size}) for its level"
                )
            used.add(int(offsets[index]) + value)
    aliased = sorted(used.intersection(values))
    if aliased:
        raise GeneratorLayoutError(
            f"special token(s) {aliased} alias a token the accepted Semantic IDs actually use"
        )

    return {
        "code_space": code_space,
        "special_tokens": dict(zip(SPECIAL_ORDER, values, strict=True)),
        "distinct_sid_tokens": len(used),
        "catalogue_items": len(assignment) - 1,
        "aliased_tokens": aliased,
    }


def derive_generator_layout(
    *,
    sid_dir: Path,
    generator_layout_path: Path,
    write: bool = True,
) -> dict[str, Any]:
    """Read the accepted layout read-only and (optionally) write the derived generator layout.

    ``sid_dir`` is opened for reading only.  The returned record is the artifact that gets written
    to ``generator_layout_path``; pass ``write=False`` to verify without touching the filesystem.
    """
    layout_path = Path(sid_dir) / "layout.json"
    semantic_ids_path = Path(sid_dir) / "semantic_ids.json"
    for required in (layout_path, semantic_ids_path):
        if not required.is_file():
            raise GeneratorLayoutError(f"missing accepted artifact {required}")

    accepted = json.loads(layout_path.read_text(encoding="utf-8"))
    sid_record = json.loads(semantic_ids_path.read_text(encoding="utf-8"))
    assignment = sid_record["assignment"]

    levels = _require_int(accepted, "levels")
    dedup_levels = _require_int(accepted, "dedup_levels")
    codebook_size = _require_int(accepted, "codebook_size")
    code_space = _require_int(accepted, "code_space")
    vocab_size = _require_int(accepted, "vocab_size")
    offsets = [int(value) for value in accepted["level_offsets"]]

    # The accepted layout must be internally consistent before anything is derived from it.  Two
    # shapes are recognised, because they are the two that exist:
    #   * the Step-2.4F production artifact: pad/bos/eos only, vocab_size = code_space + 3;
    #   * a layout the backend's own `fit-sid` wrote after SEP was registered: it already carries
    #     `sep`, so the derivation preserves it rather than appending a second one.
    accepted_specials = sorted(accepted.get("special") or {})
    accepted_has_sep = "sep" in (accepted.get("special") or {})
    expected_accepted_vocab = code_space + (4 if accepted_has_sep else 3)
    if vocab_size != expected_accepted_vocab:
        raise GeneratorLayoutError(
            f"the accepted layout declares vocab_size {vocab_size}, expected "
            f"{expected_accepted_vocab} (code_space + {'pad/bos/eos/sep' if accepted_has_sep else 'pad/bos/eos'})"
        )
    expected_specials = ["bos", "eos", "pad", "sep"] if accepted_has_sep else ["bos", "eos", "pad"]
    if accepted_specials != expected_specials:
        raise GeneratorLayoutError(
            f"the accepted layout's special tokens are {accepted_specials}, expected "
            f"{expected_specials}; refusing to derive from an unrecognised artifact"
        )
    if len(assignment) < 2:
        raise GeneratorLayoutError("the accepted assignment holds no real items")

    # The generator vocabulary: the accepted specials, with SEP appended above them only when the
    # accepted layout does not already declare one.
    generator_specials = {
        "pad": int(accepted["special"]["pad"]),
        "bos": int(accepted["special"]["bos"]),
        "eos": int(accepted["special"]["eos"]),
        "sep": int(accepted["special"]["sep"]) if accepted_has_sep else vocab_size,
    }
    generator_vocab_size = vocab_size + (0 if accepted_has_sep else 1)
    derived = {
        "format": GENERATOR_LAYOUT_FORMAT,
        "source_sid_layout_sha256": sha256_file(layout_path),
        "source_semantic_ids_sha256": sha256_file(semantic_ids_path),
        "source_sid_layout_format": accepted.get("format"),
        "source_sid_layout_vocab_size": vocab_size,
        "source_sid_layout_had_sep": accepted_has_sep,
        "levels": levels,
        "dedup_levels": dedup_levels,
        "codebook_size": codebook_size,
        "dedup_vocab_size": int(accepted.get("dedup_vocab_size", codebook_size)),
        "level_offsets": offsets,
        "per_item_tokens": levels + dedup_levels,
        "code_space": code_space,
        "special": generator_specials,
        "vocab_size": generator_vocab_size,
        "sid_to_token": accepted.get("sid_to_token", "token = level_offsets[l] + code_l"),
        "sentinel_tokenisable": False,
        "derivation": (
            "generator vocabulary = accepted code space + pad/bos/eos (copied) + sep "
            + ("preserved from the accepted layout" if accepted_has_sep else "appended directly above them")
            + "; no Semantic ID, digit, offset or codebook value is changed"
        ),
        # Kept so a reader can see the extension is at most one token and where it landed.
        "extension": {
            "added_special": None if accepted_has_sep else "sep",
            "sep_token": generator_specials["sep"],
            "accepted_vocab_size": vocab_size,
            "generator_vocab_size": generator_vocab_size,
        },
    }

    # The same checks the model relies on, run at materialisation time against the real assignment.
    derived["audit"] = assert_no_code_alias(derived, assignment)

    # `sep` must be tokenisable as a boundary but never as a digit: it is outside every level block.
    for index, offset in enumerate(offsets):
        width = codebook_size if index < levels else derived["dedup_vocab_size"]
        if offset <= generator_specials["sep"] < offset + width:
            raise GeneratorLayoutError(
                f"SEP {generator_specials['sep']} falls inside level {index}'s block "
                f"[{offset}, {offset + width})"
            )

    if write:
        generator_layout_path = Path(generator_layout_path)
        generator_layout_path.parent.mkdir(parents=True, exist_ok=True)
        generator_layout_path.write_text(
            json.dumps(derived, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        derived["generator_layout_sha256"] = sha256_file(generator_layout_path)
    return derived


def load_generator_layout(path: Path) -> dict[str, Any]:
    """Read a derived generator layout, refusing an artifact that is not one."""
    path = Path(path)
    if not path.is_file():
        raise GeneratorLayoutError(f"missing generator layout {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != GENERATOR_LAYOUT_FORMAT:
        raise GeneratorLayoutError(
            f"{path} declares format {payload.get('format')!r}, expected {GENERATOR_LAYOUT_FORMAT!r}"
        )
    specials = payload.get("special") or {}
    if sorted(specials) != sorted(SPECIAL_ORDER):
        raise GeneratorLayoutError(
            f"the generator layout declares special tokens {sorted(specials)}, expected "
            f"{sorted(SPECIAL_ORDER)}"
        )
    if int(payload["vocab_size"]) != int(payload["code_space"]) + len(SPECIAL_ORDER):
        raise GeneratorLayoutError(
            f"the generator layout's vocab_size {payload['vocab_size']} is not code_space "
            f"{payload['code_space']} + {len(SPECIAL_ORDER)} specials"
        )
    return payload


def materialise_generator_layout(*, sid_dir: Path, out: Path) -> dict[str, Any]:
    """Materialise the artifact under ``out`` as ``generator_layout.json`` (non-mutating)."""
    return derive_generator_layout(
        sid_dir=Path(sid_dir), generator_layout_path=Path(out) / "generator_layout.json"
    )
