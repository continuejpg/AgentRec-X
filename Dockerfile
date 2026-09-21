# AgentRec-X - CPU-only demo image (packaging phase).
#
#   docker compose up
#
# What this image contains
# ------------------------
# The service, the demo web page and the **small synthetic demo catalogue** that lives in
# `recommendation/demo/artifacts/`.  It deliberately does NOT contain the accepted training
# artifacts: the real checkpoint is ~349 MB, the normalised Amazon catalogue ~307 MB, and the
# raw corpus is multi-GB.  None of them belong in a Git repository or a demo image, and a
# reviewer should not have to download a public dataset to see the system run.
#
# The checkpoint and mappings are generated at build time by
# `experiments/build_demo_catalog.py`.  That checkpoint is **randomly initialised**: it makes
# the pipeline, the trust boundaries and the HTTP contract runnable and inspectable, and it
# says nothing about recommendation quality.  Every measured number in this repository comes
# from the accepted artifacts on the public dataset (see docs/EXPERIMENTS.md,
# docs/PHASE5_HANDOFF.md).
#
# Offline by default
# ------------------
# No API key is read, no provider is configured and no network call is made in the default
# path.  Real-provider mode is an explicit opt-in (`AGENTRECX_AGENT_POLICY=llm` plus
# `AGENTRECX_LLM_*`); see DOCKER.md.  Secrets are never baked into this image - they would be
# readable in its layers - and are only ever passed as environment variables at run time.
#
# Build portability note
# ----------------------
# `python:3.10-slim` is Debian-based, which matches the accepted CPU wheel index used by
# requirements-cpu.txt.  Alpine (musl) is deliberately avoided: manylinux wheels do not load
# against musl without a glibc compatibility layer, and adding one would make the image much
# harder to trust than it makes it smaller.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    AGENTRECX_DEVICE=cpu \
    AGENTRECX_HOST=0.0.0.0 \
    AGENTRECX_PORT=8000 \
    AGENTRECX_VERIFY_CHECKPOINT=0 \
    AGENTRECX_MANIFEST_PATH=none \
    AGENTRECX_CHECKPOINT_PATH=/app/runs/demo_catalog/checkpoint.pt \
    AGENTRECX_MAPPINGS_PATH=/app/runs/demo_catalog/mappings.json \
    AGENTRECX_CATALOG_METADATA_PATH=/app/recommendation/demo/artifacts/demo_products.jsonl \
    AGENTRECX_MEMORY_DB=/app/runs/demo_catalog/preference_memory.sqlite3

WORKDIR /app

# Dependencies first, so a source change does not re-resolve the wheel closure.
# requirements-cpu.txt pins the CPU-only PyTorch wheel and its dedicated index; a bare
# `pip install torch` would pull the CUDA wheel and several GB of nvidia-* packages onto a
# CPU-only image, which requirements-cpu.txt's header documents at length.
COPY requirements.txt requirements-cpu.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt \
 && python -m pip install -r requirements-cpu.txt \
 && python -m pip check

# Runtime source.  `.dockerignore` keeps tests, docs, data/, runs/ and .git out of the build
# context; `data/` may hold multi-GB archives on a developer machine.
COPY recommendation ./recommendation
COPY experiments ./experiments
COPY config ./config
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY pytest.ini conftest.py requirements-dev.txt ./

# The synthetic demo artifacts.  Deterministic (`--out` fixed, seed fixed in the builder), so
# rebuilding the image reproduces the same catalogue and checkpoint.
RUN python -m experiments.build_demo_catalog --out /app/runs/demo_catalog \
 && chmod +x /usr/local/bin/entrypoint.sh \
 && test -s /app/runs/demo_catalog/checkpoint.pt \
 && test -s /app/recommendation/demo/artifacts/demo_products.jsonl

# Run as an unprivileged user.  The demo writes only its memory database and generated
# artifacts under /app/runs, so that directory is the only one that must be writable.
RUN useradd --create-home --uid 10001 agentrecx \
 && mkdir -p /app/runs \
 && chown -R agentrecx:agentrecx /app \
 && chmod +x /usr/local/bin/entrypoint.sh
USER agentrecx

EXPOSE 8000

# Health is the service's own readiness endpoint, so "the container is up" means "the model
# and catalogue actually loaded" rather than "the process started".
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import json,sys,urllib.request; \
body=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); \
sys.exit(0 if body.get('status')=='ok' else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
