"""The AgentRec-X <-> TigerBackend contract (Step 2.3 skeleton).

This module is the **single declaration** of the boundary described in
``docs/TIGER_BACKEND.md``.  It is deliberately dependency-light: the standard library plus
NumPy, nothing else.  Nothing here imports AgentRec-X, and AgentRec-X does not import this
module - the two sides are joined by files on disk and a subprocess, and the frozen
dataclasses below are **mirrored field-for-field** in
``recommendation/backends/tiger_backend.py``.  The contract is the *schema*, not the class
identity, so both sides re-declare it and every reader checks ``format`` plus
``CONTRACT_VERSION``.

Two absences are structural rather than conventional:

* no ``parent_asin`` appears in any type, field or artifact - canonical identity belongs to
  AgentRec-X and is re-attached outside this boundary;
* no target appears in any type, field or artifact - the evaluation handoff carries
  histories and an integer frontier request, and nothing else.

Step 2.3 implements **no ML**.  Item embeddings, the RQ-VAE, the dedup training and the TIGER
generator arrive in Steps 2.4-2.6; until then the scoring stage is a deterministic stub whose
only job is to prove the process and score-batch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

__all__ = [
    "CONTRACT_VERSION",
    "PAD_SENTINEL",
    "STUB_MODULUS",
    "CatalogueHandoff",
    "ContractError",
    "EvalCohortHandoff",
    "ItemFeatureArtifact",
    "PerCaseFrontier",
    "ScoreBatch",
    "SemanticIdArtifact",
    "TigerBackend",
    "TigerCheckpoint",
    "TokenLayout",
    "TrainExposureHandoff",
    "assert_real_item_ids",
    "build_token_layout",
    "collision_audit",
    "compute_scores",
    "required_frontier_for",
    "require_contract_version",
    "require_format",
    "require_sha256",
]

#: Bumped whenever any schema in this module changes shape.  Every artifact reader on both
#: sides checks it, so a silent mismatch is impossible.
CONTRACT_VERSION = "agentrecx.tiger_backend.v3"

#: The reserved PAD row holds this value in every position.  No real digit is ever negative,
#: so a PAD row can never be tokenised, resolved, or confused with the legal code ``0``.
PAD_SENTINEL = -1

#: Upper bound used by the Step-2.3 stub scorer.  Frozen so the stub is reproducible.
STUB_MODULUS = 1_000_000


class ContractError(ValueError):
    """Raised when an artifact or an argument violates the declared contract."""


# --------------------------------------------------------------------------- #
# Token layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenLayout:
    """How Semantic IDs are laid out as model tokens.

    The code space occupies ``[0, (levels + dedup_levels) * codebook_size)`` and the special
    tokens sit strictly **above** it, so the item vocabulary's ``PAD = 0`` can never be a
    model token and a model token can never be mistaken for an item id.  Token ``0`` is the
    legal code ``(level 0, code 0)``.
    """

    levels: int
    dedup_levels: int
    codebook_size: int
    dedup_vocab_size: int
    level_offsets: tuple[int, ...]
    pad_token: int
    bos_token: int
    eos_token: int
    vocab_size: int

    @property
    def per_item_tokens(self) -> int:
        return self.levels + self.dedup_levels

    @property
    def code_space(self) -> int:
        """How many token slots the code space occupies; special tokens start here."""
        return (self.levels + self.dedup_levels) * self.codebook_size

    def validate(self) -> None:
        expected = self.levels + self.dedup_levels
        if len(self.level_offsets) != expected:
            raise ContractError(
                f"level_offsets must hold {expected} entries, got {len(self.level_offsets)}"
            )
        for index, offset in enumerate(self.level_offsets):
            if offset != index * self.codebook_size:
                raise ContractError(
                    f"level_offsets[{index}] must be {index * self.codebook_size}, got {offset}"
                )
        if self.dedup_levels not in (0, 1):
            raise ContractError(f"dedup_levels must be 0 or 1, got {self.dedup_levels}")
        if self.pad_token != self.code_space:
            raise ContractError(
                f"pad_token must be {self.code_space} (strictly above the code space), "
                f"got {self.pad_token}"
            )
        if (self.bos_token, self.eos_token) != (self.code_space + 1, self.code_space + 2):
            raise ContractError("bos_token/eos_token must follow pad_token in that order")
        if self.vocab_size != self.code_space + 3:
            raise ContractError(
                f"vocab_size must be {self.code_space + 3}, got {self.vocab_size}"
            )

    def is_tokenisable(self, semantic_id: Sequence[int]) -> bool:
        """True when a Semantic ID is a real (non-sentinel) code the tokeniser accepts."""
        if len(semantic_id) != self.per_item_tokens:
            return False
        return all(int(digit) >= 0 for digit in semantic_id)

    def tokenise(self, semantic_id: Sequence[int]) -> tuple[int, ...]:
        """Map one item's Semantic ID to its model token block.

        The PAD sentinel is **not tokenisable** and raises: a negative digit would otherwise
        produce an out-of-range token that silently aliases a real code.
        """
        if len(semantic_id) != self.per_item_tokens:
            raise ContractError(
                f"Semantic ID must hold {self.per_item_tokens} digits, got {len(semantic_id)}"
            )
        tokens: list[int] = []
        for index, digit in enumerate(semantic_id):
            value = int(digit)
            if value < 0:
                raise ContractError(
                    f"Semantic ID digit {index} is negative ({value}); the PAD sentinel is "
                    "not tokenisable"
                )
            size = self.codebook_size if index < self.levels else self.dedup_vocab_size
            if value >= size:
                raise ContractError(
                    f"Semantic ID digit {index} = {value} is outside [0, {size})"
                )
            tokens.append(self.level_offsets[index] + value)
        return tuple(tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "agentrecx.tiger.token_layout.v3",
            "levels": self.levels,
            "dedup_levels": self.dedup_levels,
            "codebook_size": self.codebook_size,
            "dedup_vocab_size": self.dedup_vocab_size,
            "per_item_tokens": self.per_item_tokens,
            "level_offsets": list(self.level_offsets),
            "special": {
                "pad": self.pad_token,
                "bos": self.bos_token,
                "eos": self.eos_token,
            },
            "vocab_size": self.vocab_size,
            "code_space": self.code_space,
            "sid_to_token": "token = level_offsets[l] + code_l",
            "sentinel_tokenisable": False,
        }


def build_token_layout(
    *,
    levels: int,
    codebook_size: int,
    dedup_levels: int = 0,
    dedup_vocab_size: int | None = None,
) -> TokenLayout:
    """Derive the token layout; ``vocab_size`` is computed, never hardcoded."""
    if levels < 1:
        raise ContractError(f"levels must be >= 1, got {levels}")
    if codebook_size < 1:
        raise ContractError(f"codebook_size must be >= 1, got {codebook_size}")
    if dedup_levels not in (0, 1):
        raise ContractError(f"dedup_levels must be 0 or 1, got {dedup_levels}")
    dedup_size = codebook_size if dedup_vocab_size is None else int(dedup_vocab_size)
    if dedup_size < 1:
        raise ContractError(f"dedup_vocab_size must be >= 1, got {dedup_size}")
    offsets = tuple(index * codebook_size for index in range(levels + dedup_levels))
    code_space = (levels + dedup_levels) * codebook_size
    layout = TokenLayout(
        levels=levels,
        dedup_levels=dedup_levels,
        codebook_size=codebook_size,
        dedup_vocab_size=dedup_size,
        level_offsets=offsets,
        pad_token=code_space,
        bos_token=code_space + 1,
        eos_token=code_space + 2,
        vocab_size=code_space + 3,
    )
    layout.validate()
    return layout


# --------------------------------------------------------------------------- #
# AgentRec-X -> backend
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CatalogueHandoff:
    """The frozen catalogue as the backend may see it: opaque item ids only.

    ``item_ids`` is the explicit ``backend_row -> item_id`` table.  A position is never
    inferred and no ``parent_asin`` is carried; the same table is also written to
    ``catalogue_items.jsonl`` so the artifact stays readable without this module.

    The declared ``*_sha256`` values must be real 64-character hex digests, so a hand-built
    handoff with a blank digest is refused rather than silently accepted.
    """

    format: str
    contract_version: str
    num_items: int
    num_users: int
    pad_id: int
    first_real_id: int
    item_id_span: tuple[int, int]
    mappings_sha256: str
    sequences_sha256: str
    products_sha256: str
    item_ids: tuple[int, ...]

    def validate(self) -> None:
        require_format(self.format, "agentrecx.tiger.catalogue.v3", "catalogue")
        require_contract_version(self.contract_version)
        if self.num_items < 1:
            raise ContractError(f"num_items must be >= 1, got {self.num_items}")
        if self.num_users < 0:
            raise ContractError(f"num_users must be >= 0, got {self.num_users}")
        if self.pad_id != 0:
            raise ContractError(f"pad_id must be 0, got {self.pad_id}")
        if self.first_real_id != 1:
            raise ContractError(f"first_real_id must be 1, got {self.first_real_id}")
        if self.item_id_span != (self.first_real_id, self.num_items):
            raise ContractError(
                f"item_id_span must be {(self.first_real_id, self.num_items)}, "
                f"got {self.item_id_span}"
            )
        if len(self.item_ids) != self.num_items:
            raise ContractError(
                f"item_ids must hold num_items = {self.num_items} entries, "
                f"got {len(self.item_ids)}"
            )
        if sorted(self.item_ids) != list(range(self.first_real_id, self.num_items + 1)):
            raise ContractError(
                "item_ids must be exactly the real item ids "
                f"{self.first_real_id}..{self.num_items}, each once, in backend_row order"
            )
        for name in ("mappings_sha256", "sequences_sha256", "products_sha256"):
            require_sha256(getattr(self, name), name)

    def row_of(self, item_id: int) -> int:
        """``item_id -> backend_row``; refuses ``PAD = 0`` and any out-of-span id."""
        if item_id == self.pad_id:
            raise ContractError("PAD (item id 0) is not an item and has no backend row")
        if not self.first_real_id <= item_id <= self.num_items:
            raise ContractError(
                f"item id {item_id} is outside {self.first_real_id}..{self.num_items}"
            )
        return item_id - self.first_real_id


@dataclass(frozen=True)
class TrainExposureHandoff:
    """Training examples built from ``EvaluationCase.train_history`` **only**.

    There is no field for a target, and the builder that produces this object never has one
    in scope, so the guarantee is structural rather than procedural.
    """

    format: str
    contract_version: str
    protocol: str
    protocol_version: str
    field_source: str
    users: int
    examples: int
    rows: tuple[tuple[int, ...], ...]
    examples_sha256: str
    catalogue_sha256: str

    def validate(self, *, catalogue: CatalogueHandoff | None = None) -> None:
        require_format(self.format, "agentrecx.tiger.train_exposure.v3", "train exposure")
        require_contract_version(self.contract_version)
        if self.field_source != "EvaluationCase.train_history":
            raise ContractError(
                f"field_source must be EvaluationCase.train_history, got {self.field_source!r}"
            )
        if self.protocol != "temporal_leave_two_out":
            raise ContractError(
                f"protocol must be temporal_leave_two_out, got {self.protocol!r}"
            )
        if self.protocol_version != "agentrecx.eval_protocol.v1":
            raise ContractError(
                "protocol_version must be agentrecx.eval_protocol.v1, got "
                f"{self.protocol_version!r}"
            )
        expected_examples = sum(len(row) - 1 for row in self.rows)
        if self.examples != expected_examples:
            raise ContractError(
                f"examples must equal the next-item transition count {expected_examples} over "
                f"{len(self.rows)} rows, got {self.examples}"
            )
        require_sha256(self.examples_sha256, "examples_sha256")
        require_sha256(self.catalogue_sha256, "catalogue_sha256")
        if catalogue is not None and self.catalogue_sha256 != catalogue.mappings_sha256:
            raise ContractError(
                "catalogue_sha256 does not match the catalogue handoff's mappings_sha256"
            )
        for index, row in enumerate(self.rows):
            if len(row) < 2:
                raise ContractError(
                    f"exposure row {index} has {len(row)} items; at least 2 are required to "
                    "form a next-item example"
                )
            assert_real_item_ids(row, catalogue, context=f"exposure row {index}")


@dataclass(frozen=True)
class EvalCohortHandoff:
    """Evaluation histories plus an integer frontier request.  No target, ever.

    ``required_frontier`` is AgentRec-X's own computation (``K_max + |seen(case)|``) and is
    an integer *count*.  The backend receives the count and never the seen-item identities,
    so it cannot mask, cannot reconstruct the set, and cannot observe a target.
    """

    format: str
    contract_version: str
    cohort_seed: int
    cohort_size: int
    protocol_version: str
    k_values: tuple[int, ...]
    case_ids: tuple[int, ...]
    test_histories: tuple[tuple[int, ...], ...]
    required_frontier: tuple[int, ...]
    catalogue_sha256: str

    def validate(self, *, catalogue: CatalogueHandoff | None = None) -> None:
        require_format(self.format, "agentrecx.tiger.eval_cohort.v3", "evaluation cohort")
        require_contract_version(self.contract_version)
        if self.protocol_version != "agentrecx.eval_protocol.v1":
            raise ContractError(
                "protocol_version must be agentrecx.eval_protocol.v1, got "
                f"{self.protocol_version!r}"
            )
        if not self.k_values:
            raise ContractError("k_values must not be empty")
        if len(self.test_histories) != self.cohort_size:
            raise ContractError(
                f"test_histories must hold cohort_size = {self.cohort_size} entries, "
                f"got {len(self.test_histories)}"
            )
        if len(self.required_frontier) != self.cohort_size:
            raise ContractError(
                f"required_frontier must hold {self.cohort_size} entries, "
                f"got {len(self.required_frontier)}"
            )
        if len(self.case_ids) != self.cohort_size:
            raise ContractError(
                f"case_ids must hold {self.cohort_size} entries, got {len(self.case_ids)}"
            )
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ContractError("case_ids must be unique")
        k_max = max(self.k_values)
        if any(int(value) < k_max for value in self.required_frontier):
            raise ContractError(
                f"every required_frontier must be >= max(k_values) = {k_max}"
            )
        require_sha256(self.catalogue_sha256, "catalogue_sha256")
        if catalogue is not None and self.catalogue_sha256 != catalogue.mappings_sha256:
            raise ContractError(
                "catalogue_sha256 does not match the catalogue handoff's mappings_sha256"
            )
        for index, history in enumerate(self.test_histories):
            assert_real_item_ids(history, catalogue, context=f"cohort case {index} history")


@dataclass(frozen=True)
class PerCaseFrontier:
    """The AgentRec-X-side frontier computation, before it reaches an artifact.

    ``seen`` never leaves this process: only :attr:`required` is serialised.
    """

    case_id: int
    history_length: int
    seen_count: int
    k_max: int
    required: int


def required_frontier_for(
    history: Sequence[int], *, case_id: int, k_max: int
) -> PerCaseFrontier:
    """``required_frontier = k_max + |unique seen|`` for one case.

    ``seen`` is the evaluator's own set: every distinct item in the *full* test-time
    history, which may be longer than the backend's truncated model window.  The result is
    therefore not assumed small.
    """
    if k_max < 1:
        raise ContractError(f"k_max must be >= 1, got {k_max}")
    seen = {int(item) for item in history}
    if 0 in seen:
        raise ContractError("history contains item id 0 (PAD); histories must be real items")
    return PerCaseFrontier(
        case_id=case_id,
        history_length=len(history),
        seen_count=len(seen),
        k_max=k_max,
        required=k_max + len(seen),
    )


# --------------------------------------------------------------------------- #
# backend -> AgentRec-X
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ItemFeatureArtifact:
    """Stage-1 output.  Step 2.4 fills it in; Step 2.3 carries the schema only."""

    format: str
    num_items: int
    dim: int
    path: str
    sha256: str
    encoder: Mapping[str, Any] = field(default_factory=dict)
    empty_text_items: int = 0
    truncated_items: int = 0
    catalogue_sha256: str = ""

    def validate(self) -> None:
        require_format(self.format, "agentrecx.tiger.item_features.v3", "item features")
        if self.num_items < 1 or self.dim < 1:
            raise ContractError("num_items and dim must be >= 1")
        require_sha256(self.sha256, "sha256")


@dataclass(frozen=True)
class SemanticIdArtifact:
    """Stage-2 output: the catalogue's Semantic-ID assignment.

    ``assignment`` is indexed by item id and holds ``num_items + 1`` rows.  Row 0 is the
    **reserved PAD row** and must equal ``(PAD_SENTINEL,) * per_item_tokens``; it is not a
    tokenisable SID and the resolver ignores it.
    """

    format: str
    contract_version: str
    num_items: int
    levels: int
    dedup_levels: int
    codebook_size: int
    dedup_vocab_size: int
    pad_row: tuple[int, ...]
    assignment: tuple[tuple[int, ...], ...]
    collision_audit: Mapping[str, Any] = field(default_factory=dict)
    utilisation: tuple[Mapping[str, Any], ...] = ()
    content_features: Mapping[str, Any] = field(default_factory=dict)

    @property
    def per_item_tokens(self) -> int:
        return self.levels + self.dedup_levels

    def layout(self) -> TokenLayout:
        return build_token_layout(
            levels=self.levels,
            codebook_size=self.codebook_size,
            dedup_levels=self.dedup_levels,
            dedup_vocab_size=self.dedup_vocab_size,
        )

    def validate(self) -> None:
        require_format(self.format, "agentrecx.tiger.semantic_ids.v3", "semantic ids")
        require_contract_version(self.contract_version)
        width = self.per_item_tokens
        if self.pad_row != (PAD_SENTINEL,) * width:
            raise ContractError(
                f"pad_row must be the sentinel {(PAD_SENTINEL,) * width}, got {self.pad_row}"
            )
        if len(self.assignment) != self.num_items + 1:
            raise ContractError(
                f"assignment must hold num_items + 1 = {self.num_items + 1} rows, "
                f"got {len(self.assignment)}"
            )
        if self.assignment[0] != self.pad_row:
            raise ContractError("assignment[0] must be the PAD sentinel row")
        layout = self.layout()
        for item_id in range(1, self.num_items + 1):
            row = self.assignment[item_id]
            if len(row) != width:
                raise ContractError(
                    f"item {item_id} has {len(row)} digits, expected {width}"
                )
            for index, digit in enumerate(row):
                size = self.codebook_size if index < self.levels else self.dedup_vocab_size
                if not 0 <= int(digit) < size:
                    raise ContractError(
                        f"item {item_id} digit {index} = {digit} is outside [0, {size})"
                    )
            layout.tokenise(row)

    def real_rows(self) -> tuple[tuple[int, ...], ...]:
        """Every real item's Semantic ID; the PAD row is excluded by construction."""
        return self.assignment[1:]


@dataclass(frozen=True)
class TigerCheckpoint:
    """Stage-3 output.  Step 2.5 fills it in; Step 2.3 carries the schema only."""

    format: str
    contract_version: str
    layout: Mapping[str, Any] = field(default_factory=dict)
    score_rule: Mapping[str, Any] = field(default_factory=dict)
    model: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    exposure_sha256: str = ""
    semantic_ids_sha256: str = ""

    def validate(self) -> None:
        """Check the artifact's own shape.

        The **score-rule** fields are deliberately **not** validated here: the frozen rule has
        exactly one definition, in :mod:`tiger_public.scoring`, and duplicating its literal
        values in a second module is how two components come to disagree about it.  Callers use
        ``scoring.validate_score_rule`` for that check.
        """
        require_format(self.format, "agentrecx.tiger.checkpoint.v3", "tiger checkpoint")
        require_contract_version(self.contract_version)
        if not isinstance(self.layout, Mapping) or "vocab_size" not in self.layout:
            raise ContractError("a checkpoint must carry its token layout (with vocab_size)")
        if not isinstance(self.score_rule, Mapping) or not self.score_rule:
            raise ContractError("a checkpoint must record its score rule by value")


@dataclass(frozen=True)
class ScoreBatch:
    """Raw scores for one batch, in request order, keyed by opaque item id.

    ``case_ids`` is the receipt: row ``i`` of ``scores`` belongs to ``case_ids[i]``.  The
    adapter compares the concatenation of every batch's receipt against the cohort order and
    refuses on any difference, so a silently reordered batch cannot pass as a valid one.
    """

    format: str
    contract_version: str
    case_ids: tuple[int, ...]
    scores: np.ndarray
    reachable: np.ndarray
    reachable_real_count: np.ndarray
    status: str
    certification: Mapping[str, Any] = field(default_factory=dict)
    retrieval: Mapping[str, Any] = field(default_factory=dict)
    generation: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, num_items: int, layout: TokenLayout | None = None) -> None:
        require_format(self.format, "agentrecx.tiger.scores.v3", "scores")
        require_contract_version(self.contract_version)
        if self.status not in ("APPROXIMATE", "CERTIFIED"):
            raise ContractError(f"status must be APPROXIMATE or CERTIFIED, got {self.status!r}")
        if layout is not None:
            layout.validate()
        batch = len(self.case_ids)
        if batch == 0:
            raise ContractError("a ScoreBatch must hold at least one case")
        if len(set(self.case_ids)) != batch:
            raise ContractError("ScoreBatch case_ids must be unique within the batch")
        expected = (batch, num_items + 1)
        if tuple(self.scores.shape) != expected:
            raise ContractError(
                f"scores shape {tuple(self.scores.shape)} must be [batch, num_items + 1] = "
                f"{expected}"
            )
        if tuple(self.reachable.shape) != expected:
            raise ContractError(
                f"reachable shape {tuple(self.reachable.shape)} must equal the scores shape "
                f"{expected}"
            )
        if self.scores.dtype != np.float32:
            raise ContractError(f"scores dtype must be float32, got {self.scores.dtype}")
        if self.reachable.dtype != np.bool_:
            raise ContractError(f"reachable dtype must be bool, got {self.reachable.dtype}")
        if tuple(self.reachable_real_count.shape) != (batch,):
            raise ContractError(
                f"reachable_real_count shape must be [{batch}], "
                f"got {tuple(self.reachable_real_count.shape)}"
            )
        if not np.isfinite(self.scores).all():
            raise ContractError("scores must be finite; NaN/Inf is never a valid score")
        # The PAD column is contract-only: the evaluator masks it positionally.
        if np.any(self.scores[:, 0] != 0.0):
            raise ContractError("scores[:, 0] (the PAD column) must be exactly 0.0")
        if np.any(self.reachable[:, 0]):
            raise ContractError("the PAD column must never be reachable")

    def certifications_ok(self) -> bool:
        """True only when a CERTIFIED batch carries a positive certificate."""
        if self.status != "CERTIFIED":
            return False
        block = self.certification or {}
        return bool(block.get("certificate_holds_for_all_cases")) and int(
            block.get("budget_exhausted_cases", 1)
        ) == 0


# --------------------------------------------------------------------------- #
# The documentation-only Protocol
# --------------------------------------------------------------------------- #


class TigerBackend(Protocol):
    """The whole boundary: four stages, no state escapes.

    Declared for documentation, for structural typing in the backend's own tests, and for one
    in-process test double.  **Production never imports an implementation**: the adapter
    crosses by spawning ``python -m tiger_public.cli <stage>``.
    """

    def build_features(
        self, catalogue: CatalogueHandoff, out_dir: str, config: Mapping[str, Any]
    ) -> ItemFeatureArtifact: ...

    def fit_semantic_ids(
        self,
        catalogue: CatalogueHandoff,
        features: ItemFeatureArtifact,
        out_dir: str,
        config: Mapping[str, Any],
    ) -> SemanticIdArtifact: ...

    def train_generator(
        self,
        exposure: TrainExposureHandoff,
        sid: SemanticIdArtifact,
        out_dir: str,
        config: Mapping[str, Any],
    ) -> TigerCheckpoint: ...

    def score(
        self,
        cohort: EvalCohortHandoff,
        sid: SemanticIdArtifact,
        checkpoint: TigerCheckpoint,
        config: Mapping[str, Any],
    ) -> Iterator[ScoreBatch]: ...


# --------------------------------------------------------------------------- #
# Shared numeric and statistical helpers
# --------------------------------------------------------------------------- #


def compute_scores(
    rows: np.ndarray, case_ids: np.ndarray, *, modulus: int = STUB_MODULUS
) -> np.ndarray:
    """The historical stub score: deterministic, bounded, and explicitly **not a model**.

    ``score = -((row * 2654435761 + case_id * 40503) mod modulus) / modulus``

    Every term is a non-negative integer, so the result lies in ``(-1, 0]`` and higher is
    better.  Step 2.5 replaces this behind the same ``ScoreBatch`` contract; keeping it here
    rather than in the CLI is what lets the boundary be tested with no ML at all.
    """
    rows_array = np.asarray(rows, dtype=np.int64)
    cases_array = np.asarray(case_ids, dtype=np.int64)
    mixed = (rows_array[:, None] * 2654435761 + cases_array[None, :] * 40503) % int(modulus)
    return -(mixed.astype(np.float64) / float(modulus)).astype(np.float32)


def collision_audit(
    real_codes: Sequence[Sequence[int]],
    *,
    dedup_levels: int,
    dedup_vocab_size: int,
) -> dict[str, Any]:
    """Both collision blocks: representation quality, and final identity uniqueness.

    The overflow check reads ``pre_dedup.largest_group`` (a property of the quantizer's code
    space, before the dedup digit masks it) while acceptance reads ``post_dedup``.  Keeping
    both is what preserves the representation-quality diagnostic.
    """
    total = len(real_codes)
    if total == 0:
        raise ContractError("collision_audit needs at least one item")
    pre_groups: dict[tuple[int, ...], int] = {}
    for codes in real_codes:
        key = tuple(int(value) for value in codes)
        pre_groups[key] = pre_groups.get(key, 0) + 1
    pre_collisions = {key: count for key, count in pre_groups.items() if count > 1}
    pre_items = sum(pre_collisions.values())
    pre_largest = max(pre_groups.values())

    overflow = pre_largest > dedup_vocab_size
    # The dedup digit is a per-group ordinal in first-come item order, exactly as the build
    # assigns it, so the post-dedup block reports the assignment that would actually be built.
    post_groups: dict[tuple[int, ...], int] = {}
    ordinals: dict[tuple[int, ...], int] = {}
    for codes in real_codes:
        key = tuple(int(value) for value in codes)
        if dedup_levels:
            ordinal = ordinals.get(key, 0)
            ordinals[key] = ordinal + 1
            full = key + (ordinal,)
        else:
            full = key
        post_groups[full] = post_groups.get(full, 0) + 1
    post_collisions = {key: count for key, count in post_groups.items() if count > 1}
    post_items = sum(post_collisions.values())

    def rate(count: int) -> float:
        return round(count / total, 6)

    return {
        "pre_dedup": {
            "collision_groups": len(pre_collisions),
            "items_in_collision": pre_items,
            "collision_rate": rate(pre_items),
            "largest_group": pre_largest,
            "distinct_sids": len(pre_groups),
        },
        "post_dedup": {
            "collision_groups": len(post_collisions),
            "items_in_collision": post_items,
            "collision_rate": rate(post_items),
            "largest_group": max(post_groups.values()) if post_groups else 0,
            "distinct_sids": len(post_groups),
        },
        "dedup_vocab_size": int(dedup_vocab_size),
        "dedup_scheme": "collision_ordinal",
        "first_member_zero": True,
        "largest_group_le_dedup_vocab": not overflow,
        "overflow_refusal_triggered": overflow,
    }


# --------------------------------------------------------------------------- #
# Contract guards
# --------------------------------------------------------------------------- #


def require_format(value: str, expected: str, context: str) -> None:
    if value != expected:
        raise ContractError(f"{context} format must be {expected!r}, got {value!r}")


def require_contract_version(value: str) -> None:
    if value != CONTRACT_VERSION:
        raise ContractError(
            f"contract_version must be {CONTRACT_VERSION!r}, got {value!r}; a backend and an "
            "adapter must agree on the schema before any artifact crosses the boundary"
        )


def require_sha256(value: str, name: str) -> None:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ContractError(f"{name} must be a 64-character hex sha256 digest")


def assert_real_item_ids(
    item_ids: Sequence[int],
    catalogue: CatalogueHandoff | None,
    *,
    context: str,
) -> None:
    """Refuse ``PAD = 0`` and any id outside the declared span.

    A ``0`` here means the backend would be shown an item that does not exist, so it is a
    refusal rather than a warning.
    """
    if catalogue is None:
        for item_id in item_ids:
            if int(item_id) == 0:
                raise ContractError(f"{context} contains item id 0 (PAD)")
        return
    for item_id in item_ids:
        value = int(item_id)
        if value == catalogue.pad_id:
            raise ContractError(f"{context} contains item id 0 (PAD)")
        if not catalogue.first_real_id <= value <= catalogue.num_items:
            raise ContractError(
                f"{context} contains item id {value}, outside "
                f"{catalogue.first_real_id}..{catalogue.num_items}"
            )
