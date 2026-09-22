"""Train the semantic tokenizer and the Semantic-ID generator.

Two stages, each independently reproducible and each writing its own artifact:

1. **Tokenizer.**  Train the RQ-VAE on catalogue item features, assign every item a Semantic ID,
   audit the assignment (coverage, collisions, codebook utilisation) and write
   ``tokenizer.pt`` + ``semantic_ids.json``.
2. **Generator.**  Build next-item examples from ``train_history`` only, train the autoregressive
   Transformer, and write ``generator.pt`` + ``run.json``.

The tokenizer is trained on **catalogue features**, which contain no interaction data at all, so
there is nothing for it to leak.  The generator reads ``train_history`` only.

Usage::

    python -m experiments.train_semantic_id --out runs/semantic_id_public_2026
    python -m experiments.train_semantic_id --tokenizer-only
    python -m experiments.train_semantic_id --generator-only --out runs/semantic_id_public_2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402
from recommendation.semantic_id.dataset import (  # noqa: E402
    GeneratorDatasetConfig,
    build_generator_dataset,
)
from recommendation.semantic_id.generator import (  # noqa: E402
    SemanticIdGenerator,
    SemanticPrefixTrie,
    build_token_layout,
)
from recommendation.semantic_id.tokenizer import (  # noqa: E402
    QuantizerConfig,
    RqVaeTokenizer,
    audit_tokenizer,
    build_item_features,
)
from recommendation.semantic_id.training import (  # noqa: E402
    GeneratorTrainerConfig,
    train_generator,
)
from recommendation.training.checkpoint import (  # noqa: E402
    TrainingState,
    environment_metadata,
    git_metadata,
    save_checkpoint,
    sha256_file,
)

SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
CATEGORIES = REPO_ROOT / "runs" / "twotower_public_2026_full" / "metadata_categories.json"
DEFAULT_OUT = REPO_ROOT / "runs" / "semantic_id_public_2026"


def train_tokenizer(
    *,
    num_items: int,
    out_dir: Path,
    levels: int,
    codebook_size: int,
    latent_size: int,
    item_embedding_dim: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    quiet: bool,
) -> dict[str, Any]:
    """Train the RQ-VAE on catalogue features and return its audit block."""
    import torch

    categories = None
    if CATEGORIES.exists():
        payload = json.loads(CATEGORIES.read_text(encoding="utf-8"))
        categories = torch.tensor(payload["matrix"], dtype=torch.long)
        if not quiet:
            print(f"metadata features: {payload['attributes']} cardinalities "
                  f"{payload['cardinalities']}")

    features, feature_note = build_item_features(
        num_items=num_items,
        item_embedding_dim=item_embedding_dim,
        categories=categories,
        seed=seed,
    )
    config = QuantizerConfig(
        input_size=int(features.shape[1]),
        latent_size=latent_size,
        levels=levels,
        codebook_size=codebook_size,
        hidden_size=hidden_size,
        seed=seed,
    )
    torch.manual_seed(seed)
    model = RqVaeTokenizer(config)

    trainable = features[1:]  # PAD is a fixed zero row and carries no item to reconstruct.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        order = torch.randperm(trainable.shape[0], generator=generator)
        total = 0.0
        steps = 0
        for start in range(0, order.shape[0], batch_size):
            index = order[start : start + batch_size]
            batch = trainable[index]
            out = model(batch)
            reconstruction_loss = torch.nn.functional.mse_loss(
                out["reconstruction"], batch
            )
            loss = (
                reconstruction_loss
                + out["commitment_loss"] * config.commitment_weight
                + out["codebook_loss"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        metrics = {
            "epoch": epoch,
            "loss": round(total / max(1, steps), 6),
            "reconstruction_loss": round(float(reconstruction_loss.detach()), 6),
            "steps": steps,
        }
        history.append(metrics)
        if not quiet:
            print(f"tokenizer epoch {epoch:>2} loss={metrics['loss']:.6f} "
                  f"recon={metrics['reconstruction_loss']:.6f}", flush=True)

    codes = model.codes_for_features(features).tolist()
    histograms: list[list[int]] = []
    for level in range(levels):
        counts = [0] * codebook_size
        for item_id in range(1, num_items + 1):
            counts[codes[item_id][level]] += 1
        histograms.append(counts)

    audit = audit_tokenizer(
        codes, num_items=num_items, config=config, codebook_histograms=histograms
    )
    audit["training"] = history
    audit["features"] = feature_note

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "quantizer_config": config.as_dict(),
            "feature_note": feature_note,
            "seed": seed,
        },
        out_dir / "tokenizer.pt",
    )
    (out_dir / "semantic_ids.json").write_text(
        json.dumps(
            {
                "levels": levels,
                "codebook_size": codebook_size,
                "num_items": num_items,
                "assignment": codes,
                "audit": audit,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if not quiet:
        print(
            f"tokenizer: codes={audit['distinct_semantic_ids']:,} "
            f"collisions={audit['items_in_a_collision_group']:,} "
            f"({audit['collision_rate']:.4%}) largest={audit['largest_collision_group']}"
        )
    return audit


def train_generator_stage(
    *,
    cases: Any,
    item_codes: list[list[int]],
    num_items: int,
    out_dir: Path,
    levels: int,
    codebook_size: int,
    max_items: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_examples: int | None,
    quiet: bool,
) -> dict[str, Any]:
    """Train the generator on train histories and return its run record."""
    import torch

    dataset = build_generator_dataset(
        cases,
        item_codes=item_codes,
        levels=levels,
        config=GeneratorDatasetConfig(
            max_items=max_items, seed=seed, max_examples=max_examples
        ),
    )
    if not quiet:
        print(f"generator examples: {len(dataset):,} users={dataset.stats['distinct_users']:,} "
              f"stats={json.dumps(dataset.stats, sort_keys=True)}", flush=True)

    layout = build_token_layout(levels=levels, codebook_size=codebook_size)
    torch.manual_seed(seed)
    model = SemanticIdGenerator(
        layout=layout,
        max_items=max_items,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    trie = SemanticPrefixTrie(item_codes, levels=levels)

    def on_epoch(epoch: int, metrics: dict[str, Any]) -> None:
        if not quiet:
            print(
                f"generator epoch {metrics['epoch']:>2} loss={metrics['loss']:.6f} "
                f"code_acc={metrics['code_accuracy']:.4f} steps={metrics['steps']}",
                flush=True,
            )

    result = train_generator(
        model,
        dataset,
        config=GeneratorTrainerConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            device="cpu",
        ),
        on_epoch=on_epoch,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / "generator.pt",
        model=model,
        optimizer=None,
        # ``train_generator`` records the step count on the *result's* stats copy, not on the
        # dataset object, so read it from there.
        state=TrainingState(
            epoch=epochs - 1, global_step=int(result.dataset_stats.get("total_steps", 0))
        ),
        model_config={
            "architecture": "semantic_id_generator",
            "levels": levels,
            "codebook_size": codebook_size,
            "max_items": max_items,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        },
        trainer_config=result.config.as_dict(),
        max_seq_len=model.max_sequence_length,
        seed=seed,
        num_items=num_items,
        dataset_identity={
            "source": "Amazon Reviews 2023 / Sports_and_Outdoors",
            "examples": len(dataset),
            "stats": dataset.stats,
            "labels_from": "EvaluationCase.train_history only",
        },
        extra={
            "architecture": "semantic_id_generator",
            "objective": "per_position_code_cross_entropy",
            "trie_nodes": trie.node_count,
            "trie_depth_histogram": trie.depth_histogram(),
        },
    )
    record = {
        "run_id": f"semantic-id-generator-seed{seed}",
        "training": result.as_dict(),
        "trie": {"nodes": trie.node_count, "depth_histogram": trie.depth_histogram()},
        "labels_from": "EvaluationCase.train_history only",
    }
    (out_dir / "generator_run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the semantic tokenizer and generator")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cohort", type=int, default=412_445,
                        help="eligible users to TRAIN on (default: all of them)")
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--item-embedding-dim", type=int, default=64)
    parser.add_argument("--tokenizer-hidden", type=int, default=256)
    parser.add_argument("--tokenizer-epochs", type=int, default=30)
    parser.add_argument("--tokenizer-batch-size", type=int, default=4096)
    parser.add_argument("--tokenizer-lr", type=float, default=1e-3)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--generator-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    cases, split_report = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    num_items = split_report.catalog_size

    if not args.generator_only:
        audit = train_tokenizer(
            num_items=num_items,
            out_dir=args.out,
            levels=args.levels,
            codebook_size=args.codebook_size,
            latent_size=args.latent_size,
            item_embedding_dim=args.item_embedding_dim,
            hidden_size=args.tokenizer_hidden,
            epochs=args.tokenizer_epochs,
            batch_size=args.tokenizer_batch_size,
            learning_rate=args.tokenizer_lr,
            seed=args.seed,
            quiet=args.quiet,
        )
        del audit
    if args.tokenizer_only:
        print(f"tokenizer written to {args.out} in {time.time() - started:.1f}s")
        return 0

    payload = json.loads((args.out / "semantic_ids.json").read_text(encoding="utf-8"))
    item_codes = payload["assignment"]

    selection = None
    if args.cohort < len(cases):
        from experiments.benchmark_public import cohort_from_cases

        selection = cohort_from_cases(cases, size=args.cohort)
        cases = list(selection["cases"])

    record = train_generator_stage(
        cases=cases,
        item_codes=item_codes,
        num_items=num_items,
        out_dir=args.out,
        levels=payload["levels"],
        codebook_size=payload["codebook_size"],
        max_items=args.max_items,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_examples=args.max_examples,
        quiet=args.quiet,
    )
    del record

    run_record = {
        "run_id": f"semantic-id-seed{args.seed}",
        "tokenizer": json.loads((args.out / "semantic_ids.json").read_text(encoding="utf-8"))[
            "audit"
        ],
        "generator": json.loads((args.out / "generator_run.json").read_text(encoding="utf-8")),
        "dataset": {
            "name": "Amazon Reviews 2023 / Sports_and_Outdoors",
            "catalog_size": num_items,
            "eligible_users": split_report.num_users_eligible,
            "training_users": len(cases),
            "split_report": split_report.as_dict(),
            "artifact_hashes": {
                "sequences_sha256": sha256_file(SEQUENCES),
                "mappings_sha256": sha256_file(MAPPINGS),
                "catalog_metadata_sha256": sha256_file(PRODUCTS),
            },
        },
        "environment": environment_metadata(),
        "git": git_metadata(REPO_ROOT),
        "wall_seconds": round(time.time() - started, 3),
    }
    (args.out / "run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out} in {run_record['wall_seconds']}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
