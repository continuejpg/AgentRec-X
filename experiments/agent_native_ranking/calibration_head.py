"""GPU forward/backward smoke for the optional Agent-native calibration head.

No labels are used and no model-quality claim is made.  This is the interface for a
future learned calibration step once independent Agent preference supervision exists.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn

OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/agent_native_ranking_phase42")


class CalibrationHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, semantic: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([semantic.unsqueeze(-1), memory], dim=-1)).squeeze(-1)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda")
    model = CalibrationHead(input_dim=1 + 8).to(device)
    semantic = torch.randn(256, device=device)
    memory = torch.randn(256, 8, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    logits = model(semantic, memory)
    loss = logits.pow(2).mean()  # gradient-flow probe only
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    elapsed = (time.perf_counter() - started) * 1000
    payload = {
        "schema": "agentrecx.agent_native_ranking_phase42.calibration_smoke.v1",
        "interface": "semantic_score + structured_memory_features -> learned_ranking_score",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch": 256,
        "backward_ok": all(parameter.grad is not None for parameter in model.parameters()),
        "forward_backward_ms": round(elapsed, 3),
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        "training_performed": False,
        "reason": "No independent Agent-native preference supervision; smoke only.",
    }
    (OUT / "calibration_smoke.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
