# AgentRec-X Archive Manifest — dependencies and external resources

> Scope: the local Git archive `archive/agentrecx-web-demo-v1` of the current AgentRec-X
> worktree. It records **what is in Git**, **what must exist outside Git to run the Web Demo**,
> how to verify those external bytes, and how far the archive is reproducible on another
> machine.  It complements [`GIT_ARCHIVE_REPORT.md`](../GIT_ARCHIVE_REPORT.md) and
> [`WEB_DEMO_GUIDE.md`](../WEB_DEMO_GUIDE.md).

## 1. Environment used to verify the archive

| Component | Verified version |
|---|---|
| Host | Windows 11 + WSL2 Ubuntu-22.04 |
| Python (WSL demo/runtime) | **3.10.12** in `.venv` (not committed) |
| Python packages | `docs/archive_python_freeze.txt` (84 packages, `pip freeze` snapshot) |
| Node.js (Windows) | **24.14.1** |
| pnpm | **12.4.1** |
| Frontend | Next.js **16.4.5**, React **19.3.1**, TypeScript **5.9.3**, Tailwind CSS **4.3.5** |
| Browser | Google Chrome **156.0.8078.80** (Playwright `channel=chrome`, no bundled browser) |
| E2E runner | Playwright **1.65.0** (`e2e/`) |
| GPU | **not used**; CPU only |

Python dependency files: `requirements.txt` (service/agent), `requirements-cpu.txt` (CPU torch
2.1.2+cpu), `requirements-mcp.txt` (`mcp==2.3.0`), `requirements-dev.txt` (pytest/httpx), plus the
freeze above. Frontend/E2E dependency locks: `web/pnpm-lock.yaml`, `e2e/pnpm-lock.yaml`.

## 2. External runtime assets required by the Web Demo

The P17 MCP server loads the **Electronics** catalogue, the **BGE dense index**, the **BGE model**
and the P17 **memory fixture** from fixed repo-relative paths. These files are huge and are
intentionally **not** in Git. The archive records their exact bytes via SHA-256.

| Role | Repo path | Bytes | SHA-256 | Source / provenance | How to obtain or rebuild |
|---|---|---:|---|---|---|
| Electronics catalogue | `data/p13r/electronics/catalog_meta.jsonl` | 32,253,569 | `14f27eedd94915ad3c5372ab3a8e13cd13f432240fd572db37e4681c49aaa1e6` | Amazon Reviews 2023 Amazon-C4 **Electronics** sampled metadata subset, prepared by P13-R (see `P13_R_REPORT.md`, `P14_PREP_REPORT.md`). `data/p13r/electronics/manifest.json` records the frozen source descriptor. | Rebuild with `python experiments/p13r/prepare_data.py` after obtaining the authorised Amazon-C4 raw inputs (`--c4-meta`, `--queries`, `--mem`, `--ts-file`; defaults point at local/author paths). No raw Amazon data is distributed here. Verify the local file against the SHA above. |
| BGE dense index | `runs/p14d2/bge_index.npz` | 213,173,605 | `fce84987083452abd84b7fdbf09b1d55e35305e7c4983b03343034dca3f7a69d` | P14-D2 repaired identity-aligned index for the 76,263 accepted Electronics records. Documented in `P14D2_REPORT.md` §1: 76,263 identities, 768-d, CLS pooling + L2 normalize, text `title + "\n" + catalogue text`, max_length 512, batch 32, model `BAAI/bge-base-en-v1.5` revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. `experiments/p14c1/search_backend.py` is the consumer (`DEFAULT_INDEX`). | The accepted artifact must be supplied out-of-band (or re-derived by encoding the catalogue with the documented parameters and verifying the identity-set equality). **A standalone builder script for this exact P14-D2 index is not present in the repository snapshot**; the published SHA is the contract. Without it, the server falls back to BM25 (still a real Web Demo, but not the verified BGE path). |
| BGE model | `models/bge-base-en-v1.5/` (`pytorch_model.bin` 437,997,357 + tokenizer/config) | 437,997,357 (weights) | `4b9d4a4b2d7ab06e3861861d5ee370974b31d17c0bb6af6f13d3dc79f33bbdcd` (weights) | `BAAI/bge-base-en-v1.5`, revision pinned in `runs/p18/freeze_manifest.json` (`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`). | `huggingface-cli download BAAI/bge-base-en-v1.5 --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a --local-dir models/bge-base-en-v1.5` (network + upstream model licence); or copy the local model directory. Verify the weight hash. |
| P17 memory fixture | `data/p14e2/benchmark/dev.agent.jsonl` | 27,908 | `a1bd72a755c77b129926f9b01d26ef4f123ced22cd26480aa9889dc3b55cc579` | P14-E2 Lite shopping-preference benchmark; manifest `data/p14e2/benchmark/manifest.json` (SHA `b15c5146…`) records 20 dev tasks, seed 20261115, catalogue 76,263 accepted / 655 quarantined. | Rebuild with `python experiments/p14e2/build_lite.py` (uses `ElectronicsCatalogue.load()`; writes `data/p14e2/benchmark/`), or copy the fixture. Verify SHA. |

**BGE index identity/provenance facts** (from `P14D2_REPORT.md`):
- identity set equals `ElectronicsCatalogue.load()` accepted ids: 76,263, 0 missing / 0 extra /
  0 duplicate; identity hash `673ff68e7e3618e3ecbc38d297a11a89ff94b1544a1e096a4f570ec188bde65c`;
- the index was **not** built from evaluator labels and no evaluator target is read by the server;
- `runs/p14d2/query_vectors.npz` is an evaluation artifact, not required for serving.

> Note on the catalogue digest: `data/p13r/electronics/manifest.json` records the P13-R frozen
> source catalogue digest `c307b1…`; the working `catalog_meta.jsonl` in this archive hashes to
> `14f27e…`. The archive's byte-level contract for the **demo asset actually used** is the table
> above. If you rebuild the catalogue, follow the P13-R/P14 pipeline and then re-verify the
> identity/index alignment before claiming parity.

## 3. Optional artifacts (not required by the Web Demo)

- **Legacy serving demo** (`/v1/recommend`, `/demo/`): SASRec checkpoint, Two-Tower checkpoint,
  Sports & Outdoors mappings/catalogue/sequences, P8 BGE index. Their canonical manifest is
  `config/demo_runtime_artifacts.json`; storage/recovery notes are in
  `docs/ARTIFACT_STORAGE.md`. These are **not** the Electronics Web Demo and are not needed to
  start `scripts/run_web_demo.ps1`.
- **P18 evidence / reports** (`runs/p18/…`, `runs/p17/…`, `runs/p14d2/…`): local generated
  evidence used by the reports; not needed for launch. They are intentionally git-ignored except
  for the ten small files already tracked under `runs/`.
- **Training adapters** (`training/agent_sft/output/*.tar.gz`, ~164 MB each): machine-local LoRA
  artifacts; ignored by `.gitignore`; not required for the Web Demo.

## 4. Reproduction steps on another machine

1. **Get the source**: checkout the archive branch/commit (see `GIT_ARCHIVE_REPORT.md`).
   Git alone contains source, tests, docs, scripts, locks, licences and screenshots.
2. **Get the external assets** in §2 (authorised Amazon-C4/E2E data or the author's accepted
   artifacts). Put them at the exact repo-relative paths; verify sizes/SHA-256.
3. **Python (WSL Ubuntu-22.04)**:
   ```bash
   python3.10 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt -r requirements-cpu.txt \
       -r requirements-dev.txt -r requirements-mcp.txt
   .venv/bin/python -c "import mcp, fastapi, uvicorn, pydantic; print('deps ok')"
   ```
4. **Frontend (Windows, Node 24 + pnpm)**:
   ```powershell
   cmd /c "pnpm --dir web install --frozen-lockfile"
   cmd /c "pnpm --dir e2e install --frozen-lockfile"
   ```
5. **Run the Web Demo** (from repo root):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1
   # frontend http://127.0.0.1:3000, BFF http://127.0.0.1:8787/api/health
   # stop: powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1
   # E2E: powershell -ExecutionPolicy Bypass -File scripts/run_web_demo_e2e.ps1
   ```
   `run_web_demo.ps1` currently assumes the WSL distro name `Ubuntu-22.04`; adapt that one
   parameter if your distro name differs.
6. **Verify**: `/api/health` reports two servers `ready` and catalogue 76,263. BGE profiles warm
   the encoder during startup; in the measured CPU run the first user query after ready was
   **0.54–0.56 s** (previously 15.6–17.1 s) and warm queries were ~0.28–0.30 s; E2E expects
   7/7 PASS (see `WEB_DEMO_REPORT.md`).

## 5. What is intentionally NOT in Git

| Excluded | Why |
|---|---|
| `data/` (all Amazon/C4 data, processed artefacts) | multi-GB, licensing/redistribution restrictions, regenerable |
| `models/` (BGE weights) | 438 MB model weights; restored from upstream HF model |
| `runs/` (4.5 GB local evidence/indices) | generated experiment output; ten small tracked evidence files remain |
| `training/**/output/*.tar.gz` adapters | 2 × ~164 MB machine-local training artifacts |
| `.venv*` / `node_modules/` / `.next/` / caches | rebuildable environments and build output |
| `.env`, `.env.*`, `.m4_credential`, SQLite DBs | secrets/credentials/local state |
| `webapp_accounts.json`, local logs | access control/local runtime state |

Exact exclusion patterns are in `.gitignore` (root, `web/.gitignore`, `e2e/.gitignore`).

## 6. Is this archive sufficient to reproduce the Web Demo elsewhere?

**Not from Git alone.** On a machine with (a) the Git archive, (b) a compatible WSL/Python/Node
environment, and (c) **legally obtained copies of the four external assets in §2**, the Web Demo
is reproducible: all code, scripts, tests, locks, licence notices and the run guide are in Git,
and the asset SHA-256 values are recorded. **Without those external assets, the real-data Web Demo
cannot start** (the source is there; the catalogue/index/model/memory fixture are not). The
archive documents how to obtain/rebuild and how to verify them; the P14-D2 BGE-index builder is a
known gap — obtain the accepted index or re-encode with the documented parameters and verify the
identity set/hash.

This is a deliberate, stated limitation, not a hidden success claim.
