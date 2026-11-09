# AgentRec-X — Demo Guide (P18)

## 1. Prerequisites

- Repository root: `D:\IT\CODE\PROJECTION\agentrec\X` (WSL path `/mnt/d/IT/CODE/PROJECTION/agentrec/X`);
- Python 3.10 WSL `.venv` with project dependencies and `mcp==2.3.0`;
- Local BGE index/model and Electronics catalogue already present;
- No GPU is required for CPU MCP Mode.

Install the isolated MCP dependency if needed:

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
.venv/bin/python -m pip install -r requirements-mcp.txt
```

## 2. Start CPU MCP Server

Read-only (recommended default):

```bash
.venv/bin/python -m recommendation.mcp.server \
  --user-key p14e2_lite_0049 \
  --memory-fixture data/p14e2/benchmark/dev.agent.jsonl \
  --search-mode bge
```

For the labelled local clarification test only:

```bash
.venv/bin/python -m recommendation.mcp.server \
  --user-key p14e2_lite_0057 \
  --memory-fixture data/p14e2/benchmark/dev.agent.jsonl \
  --search-mode bge \
  --allow-memory-writes \
  --clarification-ttl-seconds 1800
```

Logs/manifest go to **stderr**; stdout is the MCP stdio channel.

## 3. Connect an independent client

Use the official `mcp.Client` + `StdioServerParameters`. The P17/P18 clients do this directly; a minimal pattern:

```python
import anyio
from mcp import Client, StdioServerParameters

server = StdioServerParameters(
    command=".venv/bin/python",
    args=["-m", "recommendation.mcp.server", "--user-key", "p14e2_lite_0049",
          "--memory-fixture", "data/p14e2/benchmark/dev.agent.jsonl", "--search-mode", "bge"],
)
async def main():
    async with Client(server) as client:
        tools = await client.list_tools()
        print([tool.name for tool in tools.tools])
        print((await client.call_tool("search_products", {"query": "wireless earbuds", "top_k": 3})).structured_content)
anyio.run(main)
```

## 4. Run the three demos

```bash
.venv/bin/python -m experiments.p18.demos
```

Outputs:

- `runs/p18/demos/demo_transcript.json`
- `runs/p18/demos/search_and_recommend.json`
- `runs/p18/demos/multi_product_comparison.json`
- `runs/p18/demos/clarification_and_memory.json`

The script starts two independent MCP server processes: read-only user 0049 (BGE) and write-enabled user 0057 (BGE), with explicit test-user replies. No Qwen is called.

## 5. Qwen Agent Mode

Qwen Agent Mode is **not** started by P18. It requires the external Qwen3.5-9B Base service and GPU/tunnel. P18 uses saved real trajectories as **Replay only**:

```text
runs/p18/qwen_replay.json
```

Do not present Replay as a live Qwen call. To run live Qwen, restore the existing GPU/model service and use the existing P14-C1/P15 Agent entrypoints; do not build another runtime.

## 6. Memory write rules

- Writes are disabled unless the server is launched with `--allow-memory-writes`;
- the reply must be a real user-authored message and the test client marks `confirmation_source="explicit_test_user_reply"`;
- a model-generated sentence is not a user confirmation;
- the server binds one user at startup; there is no caller `user_id` parameter;
- cross-user handles, expired handles and unconfirmed writes are rejected.

## 7. Tests and logs

Targeted regression:

```bash
.venv/bin/python -m pytest tests/test_p18_final.py tests/test_p17_mcp.py tests/test_p16_skills.py \
  tests/test_p15_rag.py tests/test_p14c1.py tests/test_p14d2_dense.py \
  tests/test_p14c0r_product_role.py tests/test_p14b3_eval100.py -q --no-header
```

Inspect:

- MCP client transcript: `runs/p17/client_transcript.json`;
- P18 demos: `runs/p18/demos/`;
- performance: `runs/p18/performance.json`;
- human review materials: `runs/p18/human_review_materials.jsonl`;
- reserved one-time result: `runs/p18/reserved/`.

Failures are explicit MCP tool errors (`product_not_found`, `namespace mismatch`, `memory_writes_disabled`, `user_confirmation_required`, `clarification_id_expired`, unknown tool, etc.). Never bypass Gate/UNKNOWN/isolation to make a demo succeed.
