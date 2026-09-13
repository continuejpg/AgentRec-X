# Models

Milestone 3 provides one architecture: [`SASRec`](sasrec.py).

* Models expose **training logits** and **raw full-catalog scores** only.
* Models contain no evaluation logic, no optimizer and no training loop.
* Evaluation (PAD exclusion, seen-item masking, target retention, tie handling,
  ranking, metrics) belongs to `recommendation/evaluation/` and is never duplicated.

The SASRec data contract, architecture details and usage live in
[`../datasets/README.md`](../datasets/README.md).
