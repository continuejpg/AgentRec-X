from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/esci_external"
SOURCE = OUT / "esci_us_test_candidates.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    product_titles = {}
    query_texts = {}
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            product_titles.setdefault(str(row["product_id"]), str(row["product_title"]))
            query_texts.setdefault(str(row["query_id"]), str(row["query"]))
    product_ids = sorted(product_titles)
    query_ids = sorted(query_texts, key=int)
    (OUT / "esci_product_texts.jsonl").write_text(
        "\n".join(json.dumps({"product_id": pid, "product_title": product_titles[pid]}, ensure_ascii=False, sort_keys=True) for pid in product_ids) + "\n",
        encoding="utf-8",
    )
    (OUT / "esci_query_texts.jsonl").write_text(
        "\n".join(json.dumps({"query_id": qid, "query": query_texts[qid]}, ensure_ascii=False, sort_keys=True) for qid in query_ids) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "agentrecx.esci_external.gpu_inputs.v1",
        "source_candidates_sha256": sha256(SOURCE),
        "product_count": len(product_ids),
        "query_count": len(query_ids),
        "product_texts_sha256": sha256(OUT / "esci_product_texts.jsonl"),
        "query_texts_sha256": sha256(OUT / "esci_query_texts.jsonl"),
        "product_text_field": "product_title",
        "query_field": "query",
        "protocol": {
            "model": "BAAI/bge-base-en-v1.5",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "product_prefix": "",
            "pooling": "CLS",
            "normalization": "L2",
            "max_length": 512,
            "torch_dtype": "float32",
        },
        "labels_sent_to_gpu": False,
        "candidate_pairs_sent_to_gpu": False,
        "note": "Only unique product title and query texts are sent for embedding; labels remain local in esci_us_test_candidates.jsonl.",
    }
    (OUT / "gpu_inputs_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
