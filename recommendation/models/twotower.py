"""Two-Tower (DSSM-style) retrieval model.

What this is
------------
A standard dual-encoder retrieval baseline: a **user tower** that encodes a
chronological interaction history into one user vector, and an **item tower** that
encodes a catalogue item into one item vector.  Relevance is the dot product of the
two, so the whole catalogue can be scored as a single matrix multiply and item
vectors can be precomputed once per catalogue.

Why it is deliberately *not* another SASRec
-------------------------------------------
SASRec is an autoregressive next-item transformer: it summarises the sequence with
causal self-attention and shares the embedding table between input and output.  This
model is a **pooled sequence encoder plus an explicit projection into a separate item
space**:

* the user tower pools a lightweight recurrent encoder over the history (or a plain
  mean, via ``pooling="mean"``) rather than re-running self-attention;
* the item tower is a *different* module with its own embedding table, and it also
  consumes catalogue attributes, so the two sides are not tied;
* the training objective is contrastive retrieval (in-batch sampled softmax, see
  :mod:`recommendation.training.twotower_loss`) rather than per-position binary
  logistic loss.

Duplicating SASRec inside a tower would make the comparison uninformative: two
identical architectures differ only by seed.  The two models are therefore trained on
the same data under different objectives, which is what makes the complementarity
analysis in the Phase-5 benchmark meaningful.

Metadata
--------
The item tower can take **categorical** catalogue attributes (main category, store,
colour, material, price band).  Those are the fields the normalised catalogue carries
cleanly for every record, and they are encoded as integer category indices produced
once by :func:`recommendation.datasets.twotower.build_metadata_categories`.  Free-text
fields are deliberately *not* embedded here: a text encoder would be a much larger
model and would blur what this baseline is measuring (interaction structure), and the
repository already has a lexical text retriever as its own Phase-5 arm.
"""

from __future__ import annotations

import torch
from torch import nn

from recommendation.datasets.sasrec import PAD_ID

__all__ = [
    "TWOTOWER_POOLINGS",
    "TwoTower",
    "TwoTowerConfig",
    "TwoTowerError",
]

#: Pooling strategies the user tower supports.  ``gru`` keeps order; ``mean`` ignores it.
TWOTOWER_POOLINGS: tuple[str, ...] = ("gru", "mean")


class TwoTowerError(ValueError):
    """Raised when the two-tower configuration is invalid."""


class TwoTowerConfig:
    """Configuration for the two-tower retrieval model.

    Attributes
    ----------
    num_items:
        Catalogue size.  Item id ``i`` maps to row ``i`` of the item embedding table,
        with row ``PAD_ID`` reserved and kept at zero.
    max_seq_len:
        Fixed history window length for both training and inference.
    embedding_size:
        Width of the shared retrieval space (the dimension both towers project into).
    hidden_size:
        Width of the user tower's recurrent encoder.  Decoupled from
        ``embedding_size`` because a pooled encoder and a retrieval space need not have
        the same width.
    pooling:
        ``"gru"`` (order-aware) or ``"mean"`` (order-insensitive).  Both are legitimate
        baselines; ``"gru"`` is the default because a sequential recommender's premise is
        that order carries signal, and a mean-pooled tower would confound "two-tower is
        weak" with "order was thrown away".
    num_categories:
        Per-attribute cardinality for the optional categorical metadata features, in the
        fixed attribute order.  Empty means the item tower uses identity only.
    dropout:
        Applied to tower inputs and inside the user encoder.
    logq_correction:
        Apply the sampled-softmax logQ correction (subtract the log of each item's
        in-batch sampling probability).  See the training module for why this matters
        and why it is on by default.
    initializer_range:
        Standard deviation for normal initialisation.
    """

    def __init__(
        self,
        *,
        num_items: int,
        max_seq_len: int = 50,
        embedding_size: int = 128,
        hidden_size: int = 128,
        pooling: str = "gru",
        num_categories: tuple[int, ...] = (),
        dropout: float = 0.1,
        logq_correction: bool = True,
        initializer_range: float = 0.02,
    ) -> None:
        if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
            raise TwoTowerError(f"num_items must be a positive int, got {num_items!r}")
        for name, value in (
            ("max_seq_len", max_seq_len),
            ("embedding_size", embedding_size),
            ("hidden_size", hidden_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TwoTowerError(f"{name} must be a positive int, got {value!r}")
        if pooling not in TWOTOWER_POOLINGS:
            raise TwoTowerError(
                f"pooling must be one of {TWOTOWER_POOLINGS}, got {pooling!r}"
            )
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TwoTowerError(f"dropout must be a float, got {type(dropout).__name__}")
        if not 0.0 <= float(dropout) < 1.0:
            raise TwoTowerError(f"dropout must be in [0, 1), got {dropout!r}")
        if initializer_range <= 0.0:
            raise TwoTowerError(
                f"initializer_range must be > 0, got {initializer_range!r}"
            )
        for cardinality in num_categories:
            if isinstance(cardinality, bool) or not isinstance(cardinality, int) or cardinality < 2:
                raise TwoTowerError(
                    "every metadata cardinality must be an int >= 2 "
                    f"(0 is reserved for 'unknown'), got {cardinality!r}"
                )

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.num_categories = tuple(num_categories)
        self.dropout = float(dropout)
        self.logq_correction = bool(logq_correction)
        self.initializer_range = float(initializer_range)

    @property
    def num_metadata_fields(self) -> int:
        """How many categorical metadata attributes the item tower consumes."""
        return len(self.num_categories)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view, stored in the checkpoint."""
        return {
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "pooling": self.pooling,
            "num_categories": list(self.num_categories),
            "dropout": self.dropout,
            "logq_correction": self.logq_correction,
            "initializer_range": self.initializer_range,
        }


class UserTower(nn.Module):
    """Encode a left-padded interaction history into one L2-normalised user vector.

    The history arrives exactly as the accepted dataset builds it: left-padded with
    ``PAD_ID`` and truncated to ``max_seq_len``, so the newest interaction is the last
    real position.  That is the same convention SASRec is trained and served with, which
    is what makes the two models' inputs comparable.
    """

    def __init__(self, config: TwoTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(
            config.num_items + 1, config.hidden_size, padding_idx=PAD_ID
        )
        self.encoder: nn.Module | None
        if config.pooling == "gru":
            self.encoder = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                batch_first=True,
            )
        else:
            self.encoder = None
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, embedding_size]`` unit user vectors.

        ``input_ids`` is ``[batch, max_seq_len]`` with ``PAD_ID`` for padding.  A row that
        is entirely padding cannot be encoded into a meaningful preference and is
        rejected rather than silently mapped to the zero vector, which would score every
        item identically.
        """
        if input_ids.dim() != 2:
            raise TwoTowerError(
                f"input_ids must be 2-D [batch, max_seq_len], got {tuple(input_ids.shape)}"
            )
        if input_ids.shape[1] != self.config.max_seq_len:
            raise TwoTowerError(
                f"input_ids width must be max_seq_len={self.config.max_seq_len}, "
                f"got {input_ids.shape[1]}"
            )
        valid = input_ids != PAD_ID
        if not bool(valid.any(dim=1).all()):
            raise TwoTowerError("every history must contain at least one non-PAD item")
        # Padding must be a **prefix**.  The pooled encoder summarises the sequence at its last
        # real position, so a trailing PAD would be read as the user's most recent interaction
        # and the vector would describe padding rather than behaviour.  This is rejected loudly
        # rather than scored, because the failure is silent: every shape and finiteness check
        # still passes and the metric merely collapses.  The accepted dataset builder
        # (``datasets.sasrec.build_arrays``) and the benchmark's ``left_pad_histories`` both
        # left-pad, so a correct caller never trips this.
        transitions = valid.to(torch.int8).diff(dim=1)
        if bool((transitions < 0).any()):
            raise TwoTowerError(
                "histories must be left-padded: found a real item after a PAD position. "
                "Right-padding makes the pooled encoder read padding as the latest item"
            )

        embedded = self.dropout(self.item_embedding(input_ids))
        hidden = self._encode(embedded, valid)
        return nn.functional.normalize(self.projection(hidden), dim=-1)

    def _encode(self, embedded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Pool the embedded history into one hidden vector per row.

        The recurrent path reads the hidden state at each row's **last real position**, which
        is the sequence summary the pooled design calls for.  It reads the full padded tensor
        and gathers by index rather than using ``pack_padded_sequence``: packing expects a
        *raw* sequence tensor, and passing already-embedded rows produced a type error only on
        non-CPU devices - a portability trap for a model that may be trained on a GPU later.
        """
        if self.encoder is None:
            # Masked mean over the real positions only, so padding cannot dilute the mean.
            weights = valid.unsqueeze(-1).to(embedded.dtype)
            return (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        output, _hidden = self.encoder(embedded)
        # The last real position is the *highest* valid index, which is where left-padding puts
        # it.  Using the valid *count* instead (``sum - 1``) is the natural mistake and is wrong
        # for left-padding: with 3 real items in an 8-wide window it points at index 2, i.e. at
        # padding, and the tower returns a zero vector that L2-normalisation then turns into
        # NaN-or-zero.  The accepted SASRec serving path computes it the same way:
        # ``(positions * valid).max(dim=1)``.
        positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
        last = (positions * valid.long()).max(dim=1).values
        gather = last.unsqueeze(1).unsqueeze(2).expand(-1, 1, output.shape[-1])
        return output.gather(1, gather).squeeze(1)

    def encode_history(self, history: tuple[int, ...] | list[int]) -> torch.Tensor:
        """Encode one already-left-padded history window into a user vector."""
        tensor = torch.tensor([list(history)], dtype=torch.long, device=self._device())
        with torch.no_grad():
            return self.forward(tensor)[0]

    def _device(self) -> torch.device:
        return next(self.parameters()).device


class ItemTower(nn.Module):
    """Encode a catalogue item into one L2-normalised item vector.

    The tower is a pure function of static item features - the item id and the optional
    categorical attributes - so **every item vector can be precomputed once** and cached
    as a ``[num_items + 1, embedding_size]`` matrix.  Serving then costs one matrix
    multiply against the user vector, which is what makes this architecture the retrieval
    baseline the benchmark needs.
    """

    def __init__(self, config: TwoTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.id_embedding = nn.Embedding(
            config.num_items + 1, config.hidden_size, padding_idx=PAD_ID
        )
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.hidden_size)
            for cardinality in config.num_categories
        )
        self.dropout = nn.Dropout(config.dropout)
        input_width = config.hidden_size * (1 + config.num_metadata_fields)
        self.projection = nn.Linear(input_width, config.embedding_size)

    def forward(self, item_ids: torch.Tensor, categories: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``[batch, embedding_size]`` unit item vectors for the given item ids.

        ``categories`` is ``[batch, num_metadata_fields]`` of category indices, or
        ``None`` when the model was configured without metadata.
        """
        if item_ids.dim() != 1:
            raise TwoTowerError(
                f"item_ids must be 1-D [batch], got {tuple(item_ids.shape)}"
            )
        parts = [self.id_embedding(item_ids)]
        if self.config.num_metadata_fields:
            if categories is None:
                raise TwoTowerError(
                    f"this model expects {self.config.num_metadata_fields} categorical "
                    "metadata field(s); pass categories"
                )
            if categories.shape != (item_ids.shape[0], self.config.num_metadata_fields):
                raise TwoTowerError(
                    "categories must be "
                    f"[batch, {self.config.num_metadata_fields}], got {tuple(categories.shape)}"
                )
            for field_index, embedding in enumerate(self.category_embeddings):
                parts.append(embedding(categories[:, field_index]))
        elif categories is not None:
            raise TwoTowerError("this model was configured without categorical metadata")

        combined = self.dropout(torch.cat(parts, dim=-1))
        return nn.functional.normalize(self.projection(combined), dim=-1)


class TwoTower(nn.Module):
    """A dual encoder whose relevance score is the dot product of its two towers."""

    def __init__(self, config: TwoTowerConfig) -> None:
        super().__init__()
        if not isinstance(config, TwoTowerConfig):
            raise TwoTowerError(f"config must be a TwoTowerConfig, got {type(config).__name__}")
        self.config = config
        #: Log-scale of the softmax temperature.  Learned, so the model chooses its own
        #: sharpness; clamped on use so an unbounded value cannot produce inf/NaN logits.
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.user_tower = UserTower(config)
        self.item_tower = ItemTower(config)
        #: Categorical metadata, when the model was configured with it.  Registered as a
        #: buffer so it moves with the module and is saved in the checkpoint, but *not* a
        #: parameter: catalogue attributes are data, not something training should adjust.
        self.register_buffer(
            "_categories",
            torch.zeros(
                (config.num_items + 1, config.num_metadata_fields), dtype=torch.long
            )
            if config.num_metadata_fields
            else torch.zeros((0, 0), dtype=torch.long),
            persistent=False,
        )
        self.apply(self._init_weights)
        with torch.no_grad():
            self.user_tower.item_embedding.weight[PAD_ID].zero_()
            self.item_tower.id_embedding.weight[PAD_ID].zero_()

    # -- initialization ---------------------------------------------------- #

    def _init_weights(self, module: nn.Module) -> None:
        """Normal-initialise weights; zero biases and PAD rows."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.GRU):
            for name, parameter in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.normal_(parameter, mean=0.0, std=self.config.initializer_range)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    # -- properties -------------------------------------------------------- #

    @property
    def num_items(self) -> int:
        """Catalogue size."""
        return self.config.num_items

    @property
    def max_seq_len(self) -> int:
        """History window length."""
        return self.config.max_seq_len

    @property
    def score_vector_length(self) -> int:
        """Width of a full-catalogue score vector (``num_items + 1``, PAD included)."""
        return self.config.num_items + 1

    def temperature(self) -> torch.Tensor:
        """The positive softmax temperature, clamped away from zero and infinity."""
        return self.log_temperature.clamp(-2.0, 4.0).exp()

    # -- forward ----------------------------------------------------------- #

    def encode_users(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode histories into unit user vectors."""
        return self.user_tower(input_ids)

    def encode_items(
        self, item_ids: torch.Tensor, categories: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode item ids into unit item vectors."""
        return self.item_tower(item_ids, categories)

    def scores_from_embeddings(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        *,
        scale: bool = True,
    ) -> torch.Tensor:
        """Return ``[batch, num_items + 1]`` scores from precomputed embeddings.

        ``scale=False`` returns the raw dot product.  Scaling by the learned temperature
        changes only the spread of a row, never its order, so the benchmark's ranking is
        unaffected either way; it is exposed so a test can assert exactly that.
        """
        scores = user_embeddings @ item_embeddings.t()
        if not scale:
            return scores
        return scores * self.temperature()

    def set_categories(self, categories: torch.Tensor | None) -> None:
        """Attach the ``[num_items + 1, num_fields]`` categorical metadata matrix.

        Kept out of the parameter set (it is data, not a learned weight) but attached to
        the module so encoding an item does not require re-passing it through every call
        site, and so it cannot drift between the training and scoring paths.
        """
        if not self.config.num_metadata_fields:
            if categories is not None:
                raise TwoTowerError("this model was configured without categorical metadata")
            return
        if categories is None:
            raise TwoTowerError("this model requires categorical metadata")
        expected = (self.config.num_items + 1, self.config.num_metadata_fields)
        if tuple(categories.shape) != expected:
            raise TwoTowerError(f"categories must be {expected}, got {tuple(categories.shape)}")
        self._categories = categories.to(dtype=torch.long)

    @property
    def categories(self) -> torch.Tensor | None:
        """The attached categorical metadata matrix, or ``None`` when unused."""
        if not self.config.num_metadata_fields:
            return None
        return self._categories

    def full_catalog_scores(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score the whole catalogue for a batch of histories.

        Returns ``[batch, num_items + 1]`` with index ``i`` holding item ``i``'s score.
        Like the accepted SASRec serving path, the model performs **no** masking: PAD,
        seen items and the target all receive an ordinary finite score, and the evaluator
        owns candidate legality.  That is what keeps the two arms comparable.
        """
        user = self.encode_users(input_ids)
        item_embeddings = self.precompute_item_embeddings()
        return self.scores_from_embeddings(user, item_embeddings)

    def precompute_item_embeddings(
        self, categories: torch.Tensor | None = None, *, batch_size: int = 8192
    ) -> torch.Tensor:
        """Return the cached ``[num_items + 1, embedding_size]`` item matrix.

        Every item is encoded exactly once per call, in catalogue order, so the result is
        deterministic and independent of the caller's batching.  Row ``PAD_ID`` is encoded
        like any other row: it is never a candidate, and the evaluator masks it
        positionally, so giving it a special value would only make the matrix less
        honest.
        """
        device = next(self.parameters()).device
        resolved = categories if categories is not None else self.categories
        if self.config.num_metadata_fields:
            if resolved is None:
                raise TwoTowerError(
                    "item embeddings need the categorical metadata matrix; pass one or "
                    "call set_categories first"
                )
            if resolved.shape[0] != self.config.num_items + 1:
                raise TwoTowerError(
                    "categories must have num_items + 1 rows, got "
                    f"{resolved.shape[0]}"
                )
        rows: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, self.config.num_items + 1, batch_size):
                stop = min(start + batch_size, self.config.num_items + 1)
                item_ids = torch.arange(start, stop, dtype=torch.long, device=device)
                block = None if resolved is None else resolved[start:stop].to(device)
                rows.append(self.encode_items(item_ids, block))
        return torch.cat(rows, dim=0)
