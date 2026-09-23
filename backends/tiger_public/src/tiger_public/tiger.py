"""TIGER seq2seq generator with catalogue-constrained decoding (Step 2.5, stage 3).

```text
train_history item_ids  ->  Semantic-ID token sequences  ->  encoder-decoder
                        ->  next-item SID generation      ->  constrained decoding
```

Scope and ownership
-------------------
The backend owns SID representation, generator training, constrained generation and raw scores.
It does **not** own canonical identity, the split, the populations, targets, seen-item masking,
tie-breaking, ranking or any metric.  Training reads
``train_exposure.jsonl`` (rows already verified to equal frozen
``EvaluationCase.train_history``) and never a validation or test target.

Training-example semantics (frozen, and asserted by tests)
----------------------------------------------------------
For a train history ``i1..in`` the examples are, for each ``k`` in ``1..n-1``:

```text
input  tokens  = BOS, [i1 block], SEP, ..., [ik block], SEP
target tokens  = [i(k+1) block], EOS
```

Two properties follow and both are load-bearing:

* **2 263 252 examples** for the accepted exposure — one per (history position, next item), which
  is ``sum(max(0, len(row) - 1))`` over its 412 445 rows.  This is the count registered in
  ``docs/TIGER_BACKEND.md`` 17.4 and it reproduces exactly.  ``2 675 697`` is ``sum(len(row))``,
  the number of item *occurrences*, and ``1 850 807`` is ``sum(len(row) - 2)``, the GenRec-v0 /
  Two-Tower pair count — neither is an example count, and conflating them is the error Gate B.1
  corrected;
* the **last** item of a row is a target and never an input, so no row-final item is fed as
  history.  A target item may legitimately recur earlier in its own source: a repurchase is a legal
  next item, and an item-set disjointness test would wrongly delete every repurchase example.  The
  invariant that holds is positional — the source is exactly
  ``train_history[max(0, k - max_hist_items):k]`` and no position at or after ``k`` enters it.
  The sequence is causal and contains no validation or test interaction.

Input construction is **batch-complete**: every example's full input is embedded, with padding
placed so padding is never distinguishable from a position the model was trained on.  Because the
vocabulary has no pad token (special tokens sit above the code space), padding uses the *token*
``0`` — the legal code ``(level 0, code 0)`` — and is hidden from attention by an explicit mask.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from tiger_public.generator_layout import GENERATOR_LAYOUT_FORMAT
from tiger_public.scoring import SCORE_RULE, validate_score_rule
from tiger_public.trie import CatalogueTrie, TrieError, item_token_path

__all__ = [
    "PAD_TOKEN",
    "TigerConfig",
    "TigerDataset",
    "TigerError",
    "TigerGenerator",
    "TigerTrainer",
    "build_examples",
    "describe_examples",
    "load_generator",
    "save_generator",
]


class TigerError(ValueError):
    """Raised when the generator is configured, built or loaded unusably."""


#: Padding uses token ``0``, which is the legal code ``(level 0, code 0)``.  The vocabulary has
#: no dedicated pad token (specials sit above the code space), so padding is expressed by the
#: attention mask rather than by a reserved id.  It is never a *target*: targets are catalogue
#: SIDs followed by EOS.
PAD_TOKEN = 0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TigerConfig:
    """Generator configuration.

    Model dimensions are the values registered in ``docs/TIGER_BACKEND.md`` H5 / section 7.4
    (``d_model=256``, 6 layers, 4 heads, ``d_ff=1024``, dropout 0.1, batch 512, bf16 on a CUDA
    host).  ``epochs``, ``learning_rate`` and the rest of the training values are carried at
    ``REGISTERED_TRAINING`` below, which is the Gate-B.1 registration in section 17.4; that section
    also records the two items still open (the H5 per-epoch checkpoint cadence and the accepted
    layout's missing ``sep``).

    ``max_hist_items`` is the model's history window, not the example count: it bounds only how far
    back a single prediction may look, and truncating it does not change how many examples a row
    yields.  The registered production example count is 2 263 252 (§17.4).
    """

    # -- architecture (registered) --------------------------------------- #
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    #: History window, in items.
    max_hist_items: int = 20
    # -- training (registered budget + section 7.4 example values) -------- #
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.05
    bf16: bool = True
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise TigerError(
                f"d_model {self.d_model} must be divisible by num_heads {self.num_heads}"
            )
        if self.d_model // self.num_heads < 1:
            raise TigerError(
                f"num_heads {self.num_heads} exceeds d_model {self.d_model}"
            )
        for name in ("d_model", "num_layers", "num_heads", "d_ff", "max_hist_items",
                     "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TigerError(f"{name} must be a positive int, got {value!r}")
        if self.learning_rate <= 0.0:
            raise TigerError(f"learning_rate must be > 0, got {self.learning_rate}")

    #: The dimensions registered in ``docs/TIGER_BACKEND.md`` H5 / section 7.4.  A configuration
    #: that differs is *constructible* - the smoke and the tests need small models - but it is
    #: reported as non-registered so it can never be mistaken for the production run.
    REGISTERED = {
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "batch_size": 512,
    }

    def matches_registered_architecture(self) -> bool:
        """True when every registered architecture dimension matches."""
        return all(getattr(self, name) == value for name, value in self.REGISTERED.items())

    #: The registered Gate-C training configuration, as frozen in ``docs/TIGER_BACKEND.md`` 17.4.
    #: This is the full set of values, not only the architecture ones: an unregistered learning
    #: rate or epoch budget is exactly how two runs become incomparable.  ``matches_registered_
    #: training`` checks the *instance* against it, so a diverging default fails a test rather
    #: than being discovered after a 3-6 hour GPU run.
    REGISTERED_TRAINING = {
        "max_hist_items": 20,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "batch_size": 512,
        "epochs": 20,
        "learning_rate": 5e-4,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_fraction": 0.05,
        "bf16": True,
        "seed": 2026,
    }

    def training_divergence(self) -> dict[str, tuple[Any, Any]]:
        """Registered values this instance does not match, as ``{field: (registered, actual)}``."""
        return {
            name: (value, getattr(self, name))
            for name, value in self.REGISTERED_TRAINING.items()
            if getattr(self, name) != value
        }

    def matches_registered_training(self) -> bool:
        """True when every registered training and architecture value matches this instance."""
        return not self.training_divergence()

    def as_dict(self, *, vocab_size: int, per_item_tokens: int) -> dict[str, Any]:
        return {
            "family": "T5ForConditionalGeneration-style encoder-decoder",
            "init": "random",
            "registered_architecture": self.matches_registered_architecture(),
            "registered_training": self.matches_registered_training(),
            "training_divergence": {
                name: {"registered": registered, "actual": actual}
                for name, (registered, actual) in self.training_divergence().items()
            },
            "vocab_size": vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_kv": self.d_model // self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "max_hist_items": self.max_hist_items,
            "per_item_tokens": per_item_tokens,
            "in_len_items": self.max_hist_items,
        }

    def as_training_dict(self) -> dict[str, Any]:
        return {
            "objective": "seq2seq cross-entropy over one SID item block + EOS",
            "optimizer": "AdamW",
            "lr": self.learning_rate,
            "scheduler": "OneCycleLR",
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "bf16_requested": self.bf16,
            "examples": 0,
            "steps": 0,
            "seconds": 0.0,
            "labels_from": "agentrecx.tiger.train_exposure.v3 (train_history only)",
            # An assertion, not a report field: the backend has no validation split and no code
            # path that could compute one.
            "validation_used": False,
        }


# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TigerDataset:
    """Generator examples, flattened as flat token lists.

    Histories are stored per example (not as a padded matrix) so the window truncation is visible
    and testable; padding happens per batch in the trainer.
    """

    #: ``inputs[e]`` ends with SEP; ``targets[e]`` is one item block followed by EOS.
    inputs: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]
    user_indices: tuple[int, ...]
    examples: int
    users: int
    min_history_tokens: int
    max_history_tokens: int
    min_history_items: int
    max_history_items: int
    truncated_examples: int
    bos: int
    eos: int
    sep: int
    per_item_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "users": self.users,
            "min_history_items": self.min_history_items,
            "max_history_items": self.max_history_items,
            "min_history_tokens": self.min_history_tokens,
            "max_history_tokens": self.max_history_tokens,
            "truncated_examples": self.truncated_examples,
            "per_item_tokens": self.per_item_tokens,
            "bos": self.bos,
            "eos": self.eos,
            "sep": self.sep,
        }


def build_examples(
    rows: Sequence[Sequence[int]],
    *,
    assignment: Sequence[Sequence[int]],
    layout: Mapping[str, Any],
    max_hist_items: int,
) -> TigerDataset:
    """Build next-item examples from train histories.

    ``rows`` must be **train histories only**.  The function has no parameter that could carry a
    target, and it never reads beyond a row, so a validation or test target cannot enter by
    construction.

    A row of ``n`` items yields ``n - 1`` examples: for each ``k`` in ``1..n-1`` the input is the
    (windowed) prefix ``i1..ik`` and the target is ``i(k+1)``.  The final item of a row is
    therefore always a target and never an input.
    """
    if max_hist_items < 1:
        raise TigerError(f"max_hist_items must be >= 1, got {max_hist_items}")
    # The generator vocabulary is a *derived* artifact, not the accepted SID layout.  Requiring the
    # tag here means the two cannot be confused: the accepted layout has no SEP and a vocab_size
    # one short, so using it would either fail later or silently mis-tokenise a boundary.
    declared_format = layout.get("format")
    if declared_format != GENERATOR_LAYOUT_FORMAT:
        raise TigerError(
            f"the generator needs a {GENERATOR_LAYOUT_FORMAT!r} layout, got {declared_format!r}. "
            "Materialise it from the accepted SID layout (tiger_public.generator_layout) rather "
            "than passing the accepted layout.json, which declares no SEP."
        )
    levels = int(layout["levels"])
    total_levels = levels + int(layout["dedup_levels"])
    if len(assignment) < 2:
        raise TigerError("the assignment holds no real items")
    bos = int(layout["special"]["bos"])
    eos = int(layout["special"]["eos"])
    special = layout.get("special") or {}
    if "sep" not in special:
        raise TigerError(
            "the accepted layout declares no 'sep' special token; the generator cannot mark "
            "item boundaries without it. Re-derive the layout (additive 'sep' key) rather than "
            "guessing a token id, because a guessed id could alias a code."
        )
    sep = int(special["sep"])
    if sep >= int(layout["vocab_size"]):
        raise TigerError(
            f"SEP token {sep} is outside the vocabulary [0, {layout['vocab_size']})"
        )

    # Precompute every item's token block once; the corpus references items millions of times.
    blocks: list[tuple[int, ...]] = [()]
    for item_id in range(1, len(assignment)):
        blocks.append(item_token_path(assignment[item_id], layout))

    inputs: list[tuple[int, ...]] = []
    targets: list[tuple[int, ...]] = []
    user_indices: list[int] = []
    min_tokens = math.inf
    max_tokens = 0
    min_items = math.inf
    max_items = 0
    truncated = 0

    for user_index, row in enumerate(rows):
        items = [int(item) for item in row]
        if len(items) < 2:
            continue
        for position in range(1, len(items)):
            prefix = items[:position]
            window = prefix[-max_hist_items:]
            if len(window) < len(prefix):
                truncated += 1
            flat: list[int] = [bos]
            for item_id in window:
                if not 0 < item_id < len(blocks):
                    raise TigerError(
                        f"history item {item_id} is outside the assignment range "
                        f"1..{len(blocks) - 1}"
                    )
                flat.extend(blocks[item_id])
                flat.append(sep)
            target_item = items[position]
            if not 0 < target_item < len(blocks):
                raise TigerError(
                    f"target item {target_item} is outside the assignment range "
                    f"1..{len(blocks) - 1}"
                )
            target = (*blocks[target_item], eos)
            inputs.append(tuple(flat))
            targets.append(target)
            user_indices.append(user_index)
            min_tokens = min(min_tokens, len(flat))
            max_tokens = max(max_tokens, len(flat))
            min_items = min(min_items, len(window))
            max_items = max(max_items, len(window))

    if not targets:
        raise TigerError("no examples were built; every history row had fewer than 2 items")

    return TigerDataset(
        inputs=tuple(inputs),
        targets=tuple(targets),
        user_indices=tuple(user_indices),
        examples=len(targets),
        # Every example already carries the index of the row it came from, so the distinct-user
        # count is a set size rather than a second pass over the corpus.
        users=len(set(user_indices)),
        min_history_tokens=int(min_tokens),
        max_history_tokens=int(max_tokens),
        min_history_items=int(min_items),
        max_history_items=int(max_items),
        truncated_examples=truncated,
        bos=bos,
        eos=eos,
        sep=sep,
        per_item_tokens=total_levels,
    )


def describe_examples(dataset: TigerDataset) -> dict[str, Any]:
    """The example-shape report required by the Step-2.5 handoff."""
    return dataset.as_dict()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class TigerGenerator(nn.Module):
    """A randomly initialised T5-style encoder-decoder over SID tokens.

    Token semantics come from the accepted layout: each level owns a disjoint offset block, so a
    code's *level* is a property of the token id and needs no separate level embedding.  The dedup
    level is its own block, which is what keeps it from aliasing the last semantic level.
    """

    def __init__(
        self,
        *,
        config: TigerConfig,
        layout: Mapping[str, Any],
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layout = dict(layout)
        self.vocab_size = int(vocab_size)
        self.per_item_tokens = int(layout["levels"]) + int(layout["dedup_levels"])
        self._build()

    def _build(self) -> None:
        """Construct the T5 stack.

        ``transformers`` is imported lazily so the trie, the example builder and the checkpoint
        reader stay usable in an environment without the ML stack - which is what lets the
        boundary be tested where the model cannot run.
        """
        try:
            from transformers import T5Config, T5ForConditionalGeneration
        except Exception as error:  # pragma: no cover - depends on the backend venv
            raise TigerError(
                f"the generator needs transformers, which is not importable: {error}. Install "
                "the backend ML requirements, or use the smoke path that does not build a model."
            ) from error
        cfg = self.config
        t5 = T5Config(
            vocab_size=self.vocab_size,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            num_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_kv=cfg.d_model // cfg.num_heads,
            dropout_rate=cfg.dropout,
            # The vocabulary has no pad token: token 0 is the legal code (level 0, code 0).
            # Padding is expressed by the attention mask, so `pad_token_id` here is used only by
            # the framework for its own book-keeping and never as a target.
            pad_token_id=PAD_TOKEN,
            eos_token_id=int(self.layout["special"]["eos"]),
            decoder_start_token_id=int(self.layout["special"]["bos"]),
        )
        self.model = T5ForConditionalGeneration(t5)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    @torch.no_grad()
    def constrained_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        trie: CatalogueTrie,
        beam: int = 1,
    ) -> list[tuple[int, ...]]:
        """Generate one SID token path per row, restricted to catalogue prefixes.

        The trie is a **search constraint only**: at each step the model's own log-softmax over
        the *full* vocabulary is masked to the tokens that keep the path extendable to a real
        catalogue item.  No probability is renormalised, so the ranking semantics are exactly the
        frozen rule in :mod:`tiger_public.scoring`.

        Returns token paths.  A path is complete when the trie says so; incompleteness is
        reported rather than repaired.
        """
        if beam < 1:
            raise TigerError(f"beam must be >= 1, got {beam}")
        self.model.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch = input_ids.shape[0]
        results: list[tuple[int, ...]] = []

        encoder_outputs = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        for row in range(batch):
            row_encoder = type(encoder_outputs)(
                last_hidden_state=encoder_outputs.last_hidden_state[row : row + 1]
            )
            row_mask = attention_mask[row : row + 1]
            frontier: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
            for _level in range(trie.per_item_tokens):
                candidates: list[tuple[float, tuple[int, ...]]] = []
                for score, prefix in frontier:
                    legal = trie.children(prefix)
                    if not legal:
                        continue
                    log_probs = self._next_log_probs(
                        row_encoder, row_mask, prefix, device
                    )
                    for token in legal:
                        candidates.append((score + float(log_probs[token]), prefix + (token,)))
                if not candidates:
                    break
                candidates.sort(key=lambda item: (-item[0], item[1]))
                frontier = candidates[:beam]
            results.append(frontier[0][1] if frontier else ())
        return results

    def _next_log_probs(
        self,
        encoder_outputs: Any,
        attention_mask: torch.Tensor,
        prefix: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Full-vocabulary log-softmax for the next token after ``prefix``."""
        start = int(self.layout["special"]["bos"])
        decoder_input = torch.tensor(
            [[start, *prefix]], dtype=torch.long, device=device
        )
        decoder_mask = torch.ones_like(decoder_input)
        logits = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=decoder_mask,
        ).logits
        return torch.log_softmax(logits[0, -1, :], dim=-1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TigerTrainingReport:
    """What the trainer observed."""

    history: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    steps: int = 0
    examples: int = 0
    device: str = "cpu"
    precision: str = "fp32"
    peak_allocated_bytes: int | None = None
    resumed_from_epoch: int | None = None

    @property
    def final(self) -> dict[str, Any]:
        return self.history[-1] if self.history else {}


class TigerTrainer:
    """Deterministic local trainer: no Lightning, no Hydra, no distributed infrastructure.

    Batching is a seeded permutation of example indices (a CPU generator, because the examples
    are CPU-resident) followed by per-batch padding to that batch's own width.  That keeps the
    padding minimal without letting it become a token the model is trained to predict.
    """

    def __init__(
        self,
        model: TigerGenerator,
        dataset: TigerDataset,
        *,
        config: TigerConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device(device)
        self._rng = torch.Generator(device="cpu").manual_seed(config.seed)

    def _batches(self, *, epoch: int) -> Iterator[list[int]]:
        order = torch.randperm(
            self.dataset.examples,
            generator=torch.Generator(device="cpu").manual_seed(
                self.config.seed * 1_000_003 + epoch
            ),
        ).tolist()
        size = self.config.batch_size
        for start in range(0, len(order), size):
            yield order[start : start + size]

    def _collate(self, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(len(self.dataset.inputs[index]) for index in indices)
        rows = torch.full((len(indices), width), PAD_TOKEN, dtype=torch.long)
        mask = torch.zeros((len(indices), width), dtype=torch.long)
        targets = torch.stack(
            [torch.tensor(self.dataset.targets[index], dtype=torch.long) for index in indices]
        )
        for position, index in enumerate(indices):
            tokens = self.dataset.inputs[index]
            rows[position, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[position, : len(tokens)] = 1
        return rows, mask, targets

    def train(self, *, resume_state: Mapping[str, Any] | None = None) -> TigerTrainingReport:
        report = TigerTrainingReport(device=str(self.device), examples=self.dataset.examples)
        self.model.to(self.device)
        model = self.model.model
        use_bf16 = bool(self.config.bf16) and self.device.type == "cuda"
        report.precision = "bf16" if use_bf16 else "fp32"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        steps_per_epoch = math.ceil(self.dataset.examples / self.config.batch_size)
        total_steps = max(1, steps_per_epoch * self.config.epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_fraction,
        )
        start_epoch = 0
        if resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            start_epoch = int(resume_state["epoch"]) + 1
            report.resumed_from_epoch = int(resume_state["epoch"])
            report.history = list(resume_state.get("history", []))

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            total = 0.0
            steps = 0
            for indices in self._batches(epoch=epoch):
                if len(indices) < 2:
                    # BatchNorm-free T5 tolerates a singleton, but a 1-row batch makes the
                    # reported mean noisy for no benefit; skip a trailing singleton.
                    continue
                rows, mask, targets = self._collate(indices)
                rows = rows.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else torch.autocast("cpu", enabled=False)
                )
                with context:
                    output = self.model(rows, mask, targets)
                    loss = output.loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach())
                steps += 1
                report.steps += 1
            if steps == 0:
                raise TigerError("an epoch produced no training step; batch_size is too large")
            mean_loss = total / steps
            if not math.isfinite(mean_loss):
                raise TigerError(f"epoch {epoch} produced a non-finite loss ({mean_loss})")
            report.history.append(
                {
                    "epoch": epoch,
                    "steps": steps,
                    "loss": round(mean_loss, 6),
                    "lr": round(float(scheduler.get_last_lr()[0]), 8),
                }
            )
        report.seconds = time.perf_counter() - started
        if self.device.type == "cuda" and torch.cuda.is_available():
            report.peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        return report


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_hashes(
    sid_dir: Path, exposure_path: Path, catalogue_dir: Path, generator_layout_path: Path
) -> dict[str, str]:
    """Everything a checkpoint depends on, so an incompatible load can be refused.

    Two layout artifacts appear, deliberately.  ``layout`` is the **accepted** Step-2.4F SID
    layout, and ``generator_layout`` is the derived generator vocabulary.  Binding both means a
    checkpoint refuses when either drifts: a re-derived generator layout over the same accepted
    layout, or an accepted layout that changed underneath the same generator layout.
    """
    resolved: dict[str, str] = {}
    for name, path in (
        ("semantic_ids", sid_dir / "semantic_ids.json"),
        ("layout", sid_dir / "layout.json"),
        ("tokenizer", sid_dir / "tokenizer.pt"),
        ("train_exposure", exposure_path),
        ("catalogue", catalogue_dir / "catalogue.json"),
        ("catalogue_items", catalogue_dir / "catalogue_items.jsonl"),
        ("generator_layout", generator_layout_path),
    ):
        if not path.is_file():
            raise TigerError(f"missing dependency {path}")
        resolved[name] = _sha256_file(path)
    return resolved


def save_generator(
    *,
    model: TigerGenerator,
    report: TigerTrainingReport,
    sid_dir: Path,
    exposure_path: Path,
    catalogue_dir: Path,
    generator_layout_path: Path,
    out_dir: Path,
    resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``tiger.pt``/``tiger.json``/``generator_layout.json``/``score_rule.json``.

    The metadata binds the checkpoint to the **exact** accepted SID assignment, accepted SID
    layout, derived generator layout, tokenizer, training exposure and catalogue it was trained
    against.  Loading a checkpoint against a different one of any of those is refused rather than
    silently accepted, which is what stops a generator trained on one token space from being used
    with another.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dependencies = _dependency_hashes(sid_dir, exposure_path, catalogue_dir, generator_layout_path)
    validate_score_rule(dict(SCORE_RULE))
    layout = model.layout
    # The generator layout travels with the weights, so a checkpoint directory is self-describing.
    # The dependency hash is over the caller's artifact, so writing it is idempotent when the
    # caller already placed it there (the CLI does) and a copy otherwise.
    written_layout = out_dir / "generator_layout.json"
    if Path(generator_layout_path).resolve() != written_layout.resolve():
        written_layout.write_bytes(Path(generator_layout_path).read_bytes())
    payload = {
        "state_dict": model.model.state_dict(),
        "model_config": model.config.as_dict(
            vocab_size=model.vocab_size, per_item_tokens=model.per_item_tokens
        ),
        "layout": layout,
        "dependency_hashes": dependencies,
        "resource": resume_state or {},
    }
    checkpoint_path = out_dir / "tiger.pt"
    torch.save(payload, checkpoint_path)

    score_rule_path = out_dir / "score_rule.json"
    score_rule_path.write_text(
        json.dumps(dict(SCORE_RULE), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    metadata = {
        "format": "agentrecx.tiger.checkpoint.v3",
        "layout": dict(layout),
        "layout_sha256": _sha256_file(sid_dir / "layout.json"),
        "generator_layout_sha256": dependencies["generator_layout"],
        "score_rule": dict(SCORE_RULE),
        "score_rule_sha256": _sha256_file(score_rule_path),
        "model": {
            **model.config.as_dict(
                vocab_size=model.vocab_size, per_item_tokens=model.per_item_tokens
            ),
            "params": model.parameter_count(),
        },
        "training": {
            **model.config.as_training_dict(),
            "history": report.history,
            "steps": report.steps,
            "seconds": round(report.seconds, 3),
            "device": report.device,
            "precision": report.precision,
            "peak_allocated_bytes": report.peak_allocated_bytes,
            "resumed_from_epoch": report.resumed_from_epoch,
        },
        "dependency_hashes": dependencies,
        "exposure_sha256": dependencies["train_exposure"],
        "semantic_ids_sha256": dependencies["semantic_ids"],
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "determinism": {
            "seed": model.config.seed,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    (out_dir / "tiger.json").write_text(
        json.dumps(metadata, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_generator(
    *,
    out_dir: Path,
    sid_dir: Path,
    exposure_path: Path,
    catalogue_dir: Path,
    generator_layout_path: Path,
    device: str = "cpu",
) -> tuple[TigerGenerator, dict[str, Any], dict[str, Any]]:
    """Load a checkpoint, refusing any dependency mismatch.

    Returns ``(model, metadata, resume_state)``.  Four refusals matter:

    * a dependency hash that does not match the artifacts on disk - the checkpoint would be applied
      to a different token space, exposure or catalogue.  Both layout artifacts are covered: the
      accepted SID layout and the derived generator layout;
    * a layout whose ``vocab_size`` disagrees with the checkpoint's own model config;
    * a checkpoint directory that does not carry the generator layout it was trained with;
    * a missing checkpoint file.
    """
    checkpoint_path = out_dir / "tiger.pt"
    metadata_path = out_dir / "tiger.json"
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise TigerError(f"no checkpoint at {out_dir}")
    if not (out_dir / "generator_layout.json").is_file():
        raise TigerError(
            f"the checkpoint at {out_dir} carries no generator_layout.json; it names a vocabulary "
            "that cannot be reconstructed, so it is refused rather than guessed"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("dependency_hashes") or {}
    actual = _dependency_hashes(sid_dir, exposure_path, catalogue_dir, generator_layout_path)
    mismatched = sorted(
        name for name, digest in expected.items() if actual.get(name) != digest
    )
    if mismatched:
        raise TigerError(
            "the checkpoint was trained against different dependencies; refusing to load. "
            f"Mismatched: {mismatched}. A generator trained on one Semantic-ID layout, generator "
            "layout, exposure or catalogue must not be silently reused with another."
        )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    layout = payload["layout"]
    recorded = payload["model_config"]
    if int(layout["vocab_size"]) != int(recorded["vocab_size"]):
        raise TigerError(
            f"the checkpoint's layout vocab_size {layout['vocab_size']} disagrees with its "
            f"model config vocab_size {recorded['vocab_size']}"
        )
    # The embedded copy and the dependency file must be the same artifact, and it must be a
    # *derived generator* layout.  Accepting the accepted SID layout here would let a checkpoint
    # name a vocabulary with no SEP.
    embedded_raw = (out_dir / "generator_layout.json").read_text(encoding="utf-8")
    embedded = json.loads(embedded_raw)
    if embedded.get("format") != GENERATOR_LAYOUT_FORMAT:
        raise TigerError(
            f"the checkpoint's generator layout declares format {embedded.get('format')!r}, "
            f"expected {GENERATOR_LAYOUT_FORMAT!r}"
        )
    if int(embedded["vocab_size"]) != int(recorded["vocab_size"]):
        raise TigerError(
            f"the checkpoint's embedded generator layout declares vocab_size "
            f"{embedded['vocab_size']} but its model config says {recorded['vocab_size']}"
        )
    config = TigerConfig(
        d_model=int(recorded["d_model"]),
        num_layers=int(recorded["num_layers"]),
        num_heads=int(recorded["num_heads"]),
        d_ff=int(recorded["d_ff"]),
        dropout=float(recorded["dropout"]),
        max_hist_items=int(recorded["max_hist_items"]),
        seed=int(metadata["determinism"]["seed"]),
    )
    model = TigerGenerator(config=config, layout=layout, vocab_size=int(layout["vocab_size"]))
    model.model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    return model, metadata, payload.get("resource", {})
