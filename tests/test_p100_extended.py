from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p100_extended_benchmark"
RUN = REPO / "runs/p100_extended_evaluation"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


pytestmark = pytest.mark.skipif(not (DATA / "p100_selection_manifest.json").exists(), reason="P100 data not present")


def test_p100_selection_and_tracks():
    selection = _json(DATA / "p100_selection_manifest.json")
    assert selection["funnel"]["selected"] == 100
    assert len(selection["queries"]) == 100
    pools = _jsonl(DATA / "p100_bge_pools.jsonl")
    assert len(pools) == 100
    assert sum(len(pool["candidate_pool_ids"]) for pool in pools) == 2000
    judged = _jsonl(DATA / "p100_judged_tasks.jsonl")
    assert len(judged) == 100
    coverage = _json(DATA / "p100_label_coverage.json")
    assert coverage["human_memory_preference_gold"] == 0
    assert coverage["agent_unjudged_fraction"] > 0.9


def test_p100_ranker_and_judged_metrics():
    metrics = _json(RUN / "p100_ranker_metrics.json")
    assert set(metrics["arms"]) == {"existing", "natural", "hybrid"}
    assert all(metrics["arms"][arm]["queries"] == 100 for arm in metrics["arms"])
    judged = _json(RUN / "p100_judged_metrics.json")
    assert judged["arms"]["shopranker_query_only"]["valid_queries"] == 100
    assert judged["arms"]["shopranker_query_only"]["ndcg@10"] is not None


def test_p100_memory_off_on_exists():
    data = _json(RUN / "p100_memory_off_on.json")
    assert set(data["arms"]) == {"existing", "natural", "hybrid"}


def test_p100_agent_e2e_complete():
    path = RUN / "p100_agent_e2e.jsonl"
    if not path.exists():
        pytest.skip("P100 E2E not complete yet")
    rows = _jsonl(path)
    if len(rows) < 300:
        pytest.skip(f"P100 E2E has {len(rows)}/300 trajectories")
    assert len(rows) == 300
    assert {row["arm"] for row in rows} == {"existing", "natural", "hybrid"}
    assert sum(1 for row in rows if row["status"] == "error") <= 10
    assert all(row["hard_violations"] == 0 for row in rows)
