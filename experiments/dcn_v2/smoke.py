"""Forward/backward smoke for the DCN-V2 skeleton on the real DCN feature table.

No formal training is performed.  The loss here is only a gradient-flow probe because
the dataset does not have enough independent Memory-Conditioned supervision.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
DATA = REPO / "data/dcn_v2_ranking_v1"
sys.path.insert(0, str(REPO))
from experiments.dcn_v2.model import DCNConfig, DCNV2Ranker


def main():
    frame = pd.read_parquet(DATA / "groups.parquet")
    frame = frame[frame["source"] == "p20_fixed_pool"].reset_index(drop=True)
    schema = json.loads((DATA / "feature_schema.json").read_text())
    memory = set(schema["memory_features"])
    all_features = schema["numeric_features"]
    off_features = [name for name in all_features if name not in memory]
    on_features = all_features
    train = frame[frame["split"] == "train"]
    if len(train) < 8:
        train = frame
    x = train[all_features].astype(np.float32).fillna(0.0).to_numpy()
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = np.clip((x - mean) / std, -8, 8).astype(np.float32)
    index = {name: idx for idx, name in enumerate(all_features)}
    off_idx = [index[name] for name in off_features]
    device = torch.device("cuda")

    results = {}
    for memory_on in (False, True):
        dim = len(on_features) if memory_on else len(off_features)
        model = DCNV2Ranker(DCNConfig(input_dim=dim, hidden=64, cross_layers=3)).to(device)
        batch = torch.from_numpy(x[: min(64, len(x))][:, off_idx if not memory_on else slice(None)]).to(device)
        started = time.perf_counter()
        scores = model(batch)
        # Engineering smoke only: no label, no ranking supervision, no model-quality claim.
        loss = scores.float().pow(2).mean()
        loss.backward()
        elapsed = (time.perf_counter() - started) * 1000.0
        results["memory_on" if memory_on else "memory_off"] = {
            "input_dim": dim,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "batch": int(batch.shape[0]),
            "scores_shape": list(scores.shape),
            "smoke_loss": round(float(loss.detach()), 6),
            "backward_ok": all(parameter.grad is not None for parameter in model.parameters()),
            "forward_backward_ms": round(elapsed, 3),
        }
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    payload = {
        "schema": "agentrecx.dcn_v2.smoke.v1",
        "data": {
            "rows": int(len(train)),
            "features": len(all_features),
            "memory_features": len(memory),
            "memory_off_features": len(off_features),
            "memory_on_features": len(on_features),
        },
        "models": results,
        "gpu_peak_memory_mb": round(peak, 2),
        "training_performed": False,
        "reason": "Insufficient independent Query+Event-Sourced Memory supervision; only gradient-flow smoke was run.",
    }
    (DATA / "dcn_smoke.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
