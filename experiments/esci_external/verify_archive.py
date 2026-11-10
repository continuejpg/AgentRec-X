from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/esci_external"
RESULTS = ARCHIVE / "results"
MANIFEST = json.loads((ARCHIVE / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
METRICS = json.loads((RESULTS / "esci_metrics.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            # Text files are checked out as CRLF on Windows and LF in git archive;
            # hash the canonical LF form so both environments verify the same archive.
            if not path.name.endswith(".gz"):
                chunk = chunk.replace(b"\r\n", b"\n")
            digest.update(chunk)
    return digest.hexdigest()


def load_gzip_jsonl(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def check_hashes():
    for name, entry in MANIFEST["archived_results"].items():
        path = RESULTS / name
        if not path.is_file():
            raise AssertionError(f"missing archived result: {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise AssertionError(f"sha mismatch {name}: {actual} != {entry['sha256']}")
    for name, entry in MANIFEST["archived_sources"].items():
        path = ARCHIVE / name
        if not path.is_file():
            raise AssertionError(f"missing archived source: {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise AssertionError(f"sha mismatch {name}: {actual} != {entry['sha256']}")
    return {"result_files": len(MANIFEST["archived_results"]), "source_files": len(MANIFEST["archived_sources"])}


def check_metrics_and_results():
    per_query = load_gzip_jsonl(RESULTS / "esci_per_query.jsonl.gz")
    failures = load_gzip_jsonl(RESULTS / "esci_failures.jsonl.gz")
    if len(per_query) != 8956:
        raise AssertionError(f"per-query count {len(per_query)} != 8956")
    if len(failures) != 1529:
        raise AssertionError(f"failure count {len(failures)} != 1529")
    if len({row["query_id"] for row in per_query}) != 8956:
        raise AssertionError("per-query query IDs are not unique")
    if any(row["bm25_ndcg10_primary"] is None or row["bge_ndcg10_primary"] is None for row in per_query):
        raise AssertionError("missing nDCG@10 values")

    bm25_10 = [row["bm25_ndcg10_primary"] for row in per_query]
    bge_10 = [row["bge_ndcg10_primary"] for row in per_query]
    bm25_all = [row["bm25_ndcg_all_primary"] for row in per_query]
    bge_all = [row["bge_ndcg_all_primary"] for row in per_query]
    aggregate = METRICS["aggregate"]
    checks = [
        ("bm25_ndcg10_primary", mean(bm25_10), aggregate["bm25_ndcg10_primary"]["mean"]),
        ("bge_ndcg10_primary", mean(bge_10), aggregate["bge_ndcg10_primary"]["mean"]),
        ("bm25_ndcg_all_primary", mean(bm25_all), aggregate["bm25_ndcg_all_primary"]["mean"]),
        ("bge_ndcg_all_primary", mean(bge_all), aggregate["bge_ndcg_all_primary"]["mean"]),
        ("paired_ndcg10_primary", mean(b - a for a, b in zip(bm25_10, bge_10)), aggregate["paired_ndcg10_primary"]["mean_difference"]),
        ("paired_ndcg_all_primary", mean(b - a for a, b in zip(bm25_all, bge_all)), aggregate["paired_ndcg_all_primary"]["mean_difference"]),
    ]
    for name, recomputed, archived in checks:
        if abs(recomputed - archived) > 1e-6:
            raise AssertionError(f"{name} aggregate mismatch: {recomputed} != {archived}")
    computed_failures = sum(
        1 for row in per_query
        if (
            row["bm25_ndcg10_primary"] == 0.0 and row["bge_ndcg10_primary"] == 0.0
        ) or (
            row["bge_ndcg10_primary"] + 1e-12 < row["bm25_ndcg10_primary"] - 0.10
        )
    )
    if computed_failures != len(failures):
        raise AssertionError(f"failure recomputation {computed_failures} != {len(failures)}")
    if METRICS["data"]["queries"] != 8956 or METRICS["data"]["candidate_rows"] != 181701:
        raise AssertionError("metrics data counts mismatch")
    return {"per_query": len(per_query), "failures": len(failures), "metrics": "ok"}


def check_official_gain_audit():
    audit = json.loads((ARCHIVE / "reference/official_gain_audit.json").read_text(encoding="utf-8"))
    if audit["pinned_commit"] != "7916cdf6ab75a462e77f20ab40428a10923998d5":
        raise AssertionError("official commit mismatch")
    if audit["official_script_gain"] != {"E": 1.0, "S": 0.01, "C": 0.1, "I": 0.0}:
        raise AssertionError("official script gain audit mismatch")
    if audit["semantic_gain_primary"] != {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}:
        raise AssertionError("semantic gain audit mismatch")
    return {"official_commit": audit["pinned_commit"], "gain_mapping": "checked"}


def check_imports():
    sys.path.insert(0, str(ROOT))
    modules = [
        "experiments.esci_external.prepare_esci",
        "experiments.esci_external.export_gpu_inputs",
        "experiments.esci_external.encode_esci_gpu",
        "experiments.esci_external.evaluate_esci",
        "experiments.esci_external.cpu_gpu_product_sample",
        "experiments.esci_external.encode_esci_bge",
    ]
    imported = []
    for name in modules:
        importlib.import_module(name)
        imported.append(name)
    return {"imported": imported}


def main() -> None:
    checks = {
        "hashes": check_hashes(),
        "results_and_metrics": check_metrics_and_results(),
        "official_gain_audit": check_official_gain_audit(),
        "imports": check_imports(),
        "frozen_test": {
            "queries": MANIFEST["frozen_test"]["queries"],
            "candidates": MANIFEST["frozen_test"]["candidates"],
            "test_tuning": MANIFEST["frozen_test"]["test_tuning"],
        },
        "scope": "judged-candidate reranking, not full-catalogue retrieval",
        "gpu_encoding_rerun": False,
        "status": "PASS",
    }
    print(json.dumps(checks, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
