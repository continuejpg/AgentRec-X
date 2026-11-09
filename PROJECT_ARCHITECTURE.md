# AgentRec-X — Project Architecture (P18)

> Frozen after P18: core recommendation/Agent evidence/MCP code and P14-D2 BGE index.
> The authoritative experiment history is `EXPERIMENT_SUMMARY.md`; the demo instructions are `DEMO_GUIDE.md`.

## 1. Runtime layers

```text
MCP Client (official mcp.Client, stdio)
        |
AgentRec-X MCP Server (recommendation/mcp/server.py)
        |
        +-- existing Catalogue / BGE or BM25 search
        +-- existing Rule Planner + ElectronicsPlannerExecutor
        +-- existing V0 Scorer / Product Type-Role / Eligibility / Decision Gate
        +-- P15 Evidence Bundle + Claim Validation
        +-- P16 shopping-comparison / preference-clarification
        +-- existing M5/M9 Memory + HistoryStateReader
```

The MCP layer is a **thin adapter**. It does not implement a second catalogue, search backend, memory store, scorer, gate or Agent runtime.

## 2. Two clearly separated modes

### CPU MCP Mode

- Local stdio MCP server;
- BGE (local CPU query encoder) or BM25 backend;
- Rule Planner + existing Gate;
- P15 evidence + P16 Skills;
- No Qwen call; no GPU required.

### Qwen Agent Mode

- Existing Qwen3.5-9B Base Planner (`/plan` service);
- Existing Agent Runtime and P15 explanation (`/explain`);
- Requires the external GPU/model service (currently not connected in P18);
- P18 uses **existing saved Qwen trajectories as Replay only**, never labelled live.

## 3. Authority boundaries

| Authority | Owner |
|---|---|
| Catalogue identity / namespace | existing ElectronicsCatalogue |
| Candidate retrieval | existing Search Backend |
| Candidate ledger / eligibility | existing AgentRec-X core |
| FINALIZE decision | existing Decision Gate |
| Product evidence/spans | P15 Evidence Bundle |
| User preferences/audit | existing M5/M9 Memory; M9 `process_turn` for user replies |
| Task workflow | existing PlannerAction + Executor |
| MCP tool exposure | P17 thin adapter |
| Public network/auth | **NOT IMPLEMENTED** |

## 4. Data/artifact layout

- `data/p14e2/benchmark/`: frozen Train/Dev/Reserved agent/evaluator fixtures;
- `runs/p14d2/`: 76,263-identity BGE index;
- `runs/p15/`: Evidence Bundle / claim validation E2E;
- `runs/p16/`: two reusable Skills E2E;
- `runs/p17/`: official SDK MCP client/server transcript;
- `runs/p18/`: freeze, demos, performance, reserved one-time result, review materials.

## 5. Known architecture boundaries

- No web frontend is required or built.
- Remote Streamable HTTP is intentionally absent; no unauthenticated Memory endpoint.
- The local stdio user is bound at server startup (`--user-key`); callers cannot choose another user.
- Memory writes are disabled by default and enabled only for the explicit local test client.
- Product text is untrusted data; no model-generated product fact is accepted without an evidence ref.
- `semantic_truth_verified=false` remains the honest status without human semantic review.
