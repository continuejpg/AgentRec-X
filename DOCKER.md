# Running AgentRec-X in Docker

> **Status: the Docker path is provided but was not build-verified in the authoring
> environment.** The development host has no Docker daemon available inside its WSL
> distribution, so `docker build` and `docker compose up` could not be executed there. The
> native one-command path (`./scripts/run_demo.sh`) *was* executed end to end, and the CI
> workflow builds the image and smoke-tests the running container on every push. Treat the
> file as reviewed-but-unverified until that workflow has run once.

---

## One command

```bash
docker compose up
```

Then open:

| URL | What it is |
| --- | --- |
| <http://127.0.0.1:8000/demo/> | the browser demo (multi-turn sessions) |
| <http://127.0.0.1:8000/docs> | OpenAPI docs, including `POST /v1/demo/agent/recommend` |
| <http://127.0.0.1:8000/health> | readiness — `ok` only when the model actually loaded |
| <http://127.0.0.1:8000/v1/demo/health> | demo readiness: model, catalogue, profiles, sessions |

## What the image contains, and what it deliberately does not

The image ships the service, the demo web page, and the **small synthetic demo catalogue** in
`recommendation/demo/artifacts/demo_products.jsonl`. It generates its own tiny checkpoint and
mappings at build time.

It does **not** contain the accepted training artifacts. Those total roughly 832 MB — a
~349 MB checkpoint, a ~307 MB normalised Amazon catalogue, and a multi-GB raw corpus — none of
which belongs in a Git repository or a demo image. A reviewer should not have to download a
public dataset to watch the system run.

**The demo checkpoint is randomly initialised.** It makes the pipeline, the trust boundaries
and the HTTP contract runnable and inspectable. It says nothing about recommendation quality.
Every measured number in this repository comes from the accepted artifacts on the public
dataset and lives in [`EXPERIMENTS.md`](EXPERIMENTS.md) and
[`PHASE5_HANDOFF.md`](PHASE5_HANDOFF.md).

## Offline by default

The default configuration reads no API key, configures no provider and makes no network call.
The entrypoint is what guarantees that, not convention:

* it prints which mode it is in, so the container logs state the guarantee;
* in the default mode it **unsets** every `AGENTRECX_LLM_*` variable, so a credential inherited
  from the host cannot silently start billed calls;
* in `llm` mode it refuses to start unless a base URL and a model name are present.

## Real-provider (DeepSeek) mode — explicit opt-in

Copy the example and fill it in:

```bash
cp .env.example .env      # .env and .env.* are git-ignored
$EDITOR .env              # set AGENTRECX_AGENT_POLICY=llm and the three provider values
docker compose up
```

| Variable | Meaning |
| --- | --- |
| `AGENTRECX_AGENT_POLICY=llm` | switches the single-turn endpoint to the accepted model-driven policy |
| `AGENTRECX_LLM_BASE_URL` | provider base URL, e.g. `https://api.deepseek.com/v1` |
| `AGENTRECX_LLM_MODEL` | model identifier, e.g. `deepseek-flash` |
| `AGENTRECX_LLM_API_KEY` | your credential — **never committed, never logged, never baked into the image** |

Secrets reach the process only as environment variables at run time. They are not in the image:
`.dockerignore` excludes `.env`, `.env.*`, `*.key` and `*.pem` from the build context, so a
stray local file cannot be `COPY`ed in, and the entrypoint never prints a variable's value.

## Using the accepted artifacts instead of the demo catalogue

Point the container at a mounted artifact directory and disable the demo defaults:

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/accepted/artifacts:/artifacts:ro \
  -e AGENTRECX_CHECKPOINT_PATH=/artifacts/best.pt \
  -e AGENTRECX_MAPPINGS_PATH=/artifacts/mappings.json \
  -e AGENTRECX_CATALOG_METADATA_PATH=/artifacts/products.jsonl \
  -e AGENTRECX_MANIFEST_PATH=/artifacts/run.json \
  -e AGENTRECX_VERIFY_CHECKPOINT=1 \
  agentrecx:demo
```

With the accepted manifest in force, `AGENTRECX_VERIFY_CHECKPOINT=1` re-enables the digest
check against the accepted training run — the strong configuration. The demo image defaults
to `0` **and** `AGENTRECX_MANIFEST_PATH=none`, because the accepted manifest cross-checks a
checkpoint against a training run the demo checkpoint was never part of.

## Volumes and persistence

`docker compose up` mounts the `demo_artifacts` volume at `/app/runs`. It holds the generated
checkpoint, the mappings and the preference-memory SQLite database, so they survive a restart.
Delete the volume to force a clean regeneration:

```bash
docker compose down -v
```

## Health check

The image's `HEALTHCHECK` calls the service's own `/health`, so "the container is healthy"
means "the model and catalogue actually loaded" rather than "the process started". A container
whose checkpoint is missing reports `unhealthy` with the reason in `/health`'s `detail` field.

## Verifying the image

```bash
docker build -t agentrecx:ci .
docker run -d --name agentrecx-ci -p 8000:8000 agentrecx:ci
docker inspect --format '{{.State.Health.Status}}' agentrecx-ci    # -> healthy
curl -fsS http://127.0.0.1:8000/v1/demo/health
docker logs agentrecx-ci | head -5                                 # -> "agent policy : deterministic (offline"
docker rm -f agentrecx-ci
```

`.github/workflows/verify.yml` runs exactly these steps, plus an assertion that no
credential-shaped string appears in the container logs.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/health` returns `unavailable` | checkpoint or mappings missing or unreadable | check the container logs; the entrypoint regenerates demo artifacts when the volume is empty |
| `AGENTRECX_AGENT_POLICY=llm requires AGENTRECX_LLM_BASE_URL` | opt-in selected without provider configuration | set both provider values, or unset the policy switch to run offline |
| container exits immediately with a pydantic traceback | a `--reload`-style dev flag or a stale command | the image entrypoint takes no arguments; use environment variables |
| `python: command not found` from the entrypoint | `AGENTRECX_PYTHON` points somewhere without an interpreter | unset it (the image uses `python` from the virtualenv-free base image) |
| build fails resolving `torch` | a bare `pip install torch` was substituted for `requirements-cpu.txt` | use the pinned file; it selects the CPU wheel index |
