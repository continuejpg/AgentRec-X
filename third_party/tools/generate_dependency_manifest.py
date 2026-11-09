#!/usr/bin/env python3
"""Generate third_party/DEPENDENCY_MANIFEST.json for the AgentRec-X Web Demo.

Sources of truth
----------------
- npm:     web/package.json (direct dependencies / devDependencies) enriched with
           the installed package.json under web/node_modules/<name> if present.
- Python:  the project virtual environment's *.dist-info/METADATA (name, version,
           license), against an explicit list of runtime packages used by the BFF
           and the P17 MCP server. Requirements files are recorded as the declared
           version ranges.
- Upstream: the five read-only reference repositories audited for task-5, with
           URL + commit SHA + license + reuse mode (see THIRD_PARTY_NOTICES.md).

The script is read-only; it only writes --out (default
third_party/DEPENDENCY_MANIFEST.json). Standard library only.

Usage
-----
    python third_party/tools/generate_dependency_manifest.py
    python third_party/tools/generate_dependency_manifest.py \\
        --venv .venv --web web --out third_party/DEPENDENCY_MANIFEST.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB = REPO_ROOT / "web"
DEFAULT_VENV = REPO_ROOT / ".venv"
DEFAULT_OUT = REPO_ROOT / "third_party" / "DEPENDENCY_MANIFEST.json"

# Well-known SPDX ids, used only when a package is declared but not installed.
KNOWN_NPM_LICENSES = {
    "next": "MIT",
    "react": "MIT",
    "react-dom": "MIT",
    "tailwindcss": "MIT",
    "@tailwindcss/postcss": "MIT",
    "typescript": "Apache-2.0",
    "@types/node": "MIT",
    "@types/react": "MIT",
    "@types/react-dom": "MIT",
    "eslint": "MIT",
    "eslint-config-next": "MIT",
    "@playwright/test": "Apache-2.0",
    "clsx": "MIT",
}

# Python packages that participate in the web-demo runtime. `role` is
# "direct" (declared in requirements*.txt and imported by demo code / MCP server)
# or "transitive" (pulled in by a direct dependency).
PYTHON_PACKAGES = {
    "mcp": {"role": "direct", "requirement": "requirements-mcp.txt", "used_by": ["web_demo (MCP client)", "recommendation/mcp (P17 server)"]},
    "fastapi": {"role": "direct", "requirement": "requirements.txt", "used_by": ["web_demo/app.py"]},
    "uvicorn": {"role": "direct", "requirement": "requirements.txt", "used_by": ["web_demo/__main__.py"]},
    "pydantic": {"role": "direct", "requirement": "requirements.txt", "used_by": ["web_demo request/view models"]},
    "numpy": {"role": "direct", "requirement": "requirements.txt", "used_by": ["recommendation/inference/ranking.py (served path)"]},
    "langgraph": {"role": "direct", "requirement": "requirements.txt", "used_by": ["recommendation/agent (P17 MCP server tool path)"]},
    "langsmith": {"role": "transitive", "requirement": "requirements.txt (pinned compatibility cap)", "used_by": ["langgraph -> langchain-core"]},
    "langchain-core": {"role": "transitive", "requirement": "", "used_by": ["langgraph"]},
    "starlette": {"role": "transitive", "requirement": "", "used_by": ["fastapi"]},
    "anyio": {"role": "transitive", "requirement": "", "used_by": ["fastapi", "mcp", "httpx"]},
    "httpx": {"role": "transitive", "requirement": "requirements-dev.txt", "used_by": ["mcp", "fastapi TestClient"]},
    "h11": {"role": "transitive", "requirement": "", "used_by": ["uvicorn", "httpcore"]},
    "sse-starlette": {"role": "transitive", "requirement": "", "used_by": ["optional SSE helper (not required by web_demo)"]},
    "websockets": {"role": "transitive", "requirement": "", "used_by": ["uvicorn"]},
}

UPSTREAM_REPOSITORIES = [
    {
        "id": "commerce-agents",
        "url": "https://github.com/anthropics/commerce-agents",
        "commit": "fd4d59224ab96b43c6dc6888207c67b3bd5a24cf",
        "license": "Apache-2.0",
        "license_file_at_root": True,
        "notice_file": False,
        "use_mode": "vendored-adapted",
        "reuse_policy": "Level-1 reuse: copy/adapt the examples/web-shared and examples/retail/storefront-web components with retained copyright headers and modification notes.",
        "license_copy": "third_party/licenses/commerce-agents-Apache-2.0.txt",
    },
    {
        "id": "shop-gym",
        "url": "https://github.com/agentic-foundation-modeling-research/shop-gym",
        "commit": "dbef277c37c9352c17c6df800caee5eb3591916f",
        "license": "MIT",
        "license_file_at_root": True,
        "license_copyright": "Copyright (c) 2026-present, Shopify Inc.",
        "notice_file": False,
        "use_mode": "design-reference-only",
        "reuse_policy": "Reference only (storefront layout/responsive); workspace not imported. If code is ever vendored, add the MIT text and notice first.",
        "license_copy": None,
    },
    {
        "id": "PAHF",
        "url": "https://github.com/facebookresearch/PAHF",
        "commit": "7a11213360a82d5f437a035e3a31c92d6307f8cf",
        "license": "MIT",
        "license_file_at_root": True,
        "license_copyright": "Copyright (c) Meta Platforms, Inc. and affiliates.",
        "notice_file": False,
        "use_mode": "design-reference-only",
        "reuse_policy": "Reference only (preference clarification / memory interaction design); NOT a Memory backend. If code is ever vendored, add the MIT text and notice first.",
        "license_copy": None,
    },
    {
        "id": "techjam-conversational-search-agent",
        "url": "https://github.com/sci-m-wang/techjam-conversational-search-agent",
        "commit": "b6310f0279e3e56a69ecbdb2a28f323400ce32c0",
        "license": "NO LICENSE FILE",
        "license_file_at_root": False,
        "notice_file": False,
        "use_mode": "design-reference-only",
        "reuse_policy": "No license grant: no code copying permitted. Design reference only (deterministic fast path vs optional LLM path, progress/usage telemetry).",
        "license_copy": None,
    },
    {
        "id": "coveo-commerce-agent-frontend",
        "url": "https://github.com/coveo-labs/commerce-agent-frontend-implementation",
        "commit": "a271546c9cfa1d055b48c48f634783f38d73dd3f",
        "license": "NO LICENSE FILE",
        "license_file_at_root": False,
        "notice_file": False,
        "use_mode": "design-reference-only",
        "reuse_policy": "No license grant: no code copying permitted. Design reference only (comparison table/summary, next-actions bar, tool activity model, loading-first surfaces, AG-UI/A2UI event separation).",
        "license_copy": None,
    },
]

LICENSE_TEXT_RE = re.compile(
    r"^(?:License-Expression|License|Classifier:\s*License)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def normalize_npm_license(license_value) -> str | None:
    """Return a short license id from a package.json `license` field."""
    if isinstance(license_value, str) and license_value.strip():
        value = license_value.strip()
        if value.upper().startswith("SEE LICENSE IN"):
            return value
        # Take the first identifier of an SPDX expression like "(MIT OR Apache-2.0)".
        match = re.search(r"[A-Za-z0-9.+-]+", value)
        return match.group(0) if match else value
    if isinstance(license_value, dict):
        return normalize_npm_license(license_value.get("type") or license_value.get("name"))
    if isinstance(license_value, list):
        parts = [normalize_npm_license(item) for item in license_value]
        return " OR ".join(part for part in parts if part) or None
    return None


def npm_dependencies(web_dir: Path) -> dict:
    package_json = read_json(web_dir / "package.json")
    result: dict = {
        "package_json": str((web_dir / "package.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "package_json_present": package_json is not None,
        "direct_dependencies": [],
        "dev_dependencies": [],
        "notes": [],
    }
    if package_json is None:
        result["notes"].append(
            "web/package.json not found yet (frontend milestone pending); rerun this generator after it exists."
        )
        return result

    node_modules = web_dir / "node_modules"
    for section, key in (("direct_dependencies", "dependencies"), ("dev_dependencies", "devDependencies")):
        declared = package_json.get(key) or {}
        for name in sorted(declared):
            entry = {
                "name": name,
                "version_range": declared[name],
                "installed_version": None,
                "license": None,
                "license_source": "unresolved",
                "source": "npm",
            }
            installed_pkg = read_json(node_modules / name / "package.json")
            if installed_pkg:
                entry["installed_version"] = installed_pkg.get("version")
                license_id = normalize_npm_license(installed_pkg.get("license"))
                if license_id:
                    entry["license"] = license_id
                    entry["license_source"] = "web/node_modules/<name>/package.json"
            if entry["license"] is None and name in KNOWN_NPM_LICENSES:
                entry["license"] = KNOWN_NPM_LICENSES[name]
                entry["license_source"] = "known-package-license-table (node_modules not installed)"
            result[section].append(entry)
    if not node_modules.is_dir():
        result["notes"].append(
            "web/node_modules not installed; versions fall back to package.json ranges and licenses to the known table. Rerun after `pnpm install` for installed metadata."
        )
    return result


def python_metadata(venv_dir: Path) -> dict:
    site_packages = None
    for candidate in sorted(venv_dir.glob("lib/python*/site-packages")):
        site_packages = candidate
        break
    if site_packages is None:
        site_packages = venv_dir / "Lib" / "site-packages"
    dist_infos = {}
    if site_packages.is_dir():
        for info in site_packages.glob("*.dist-info"):
            dist_infos[info.name.split("-")[0].lower().replace("_", "-")] = info
    entries = []
    for name, meta in PYTHON_PACKAGES.items():
        key = name.lower().replace("_", "-")
        entry = {
            "name": name,
            "role": meta["role"],
            "declared_requirement_file": meta["requirement"] or None,
            "used_by": meta["used_by"],
            "installed_version": None,
            "license": None,
            "license_source": "unresolved",
            "source": "PyPI",
        }
        info = dist_infos.get(key)
        if info is not None:
            stem = info.name[: -len(".dist-info")] if info.name.endswith(".dist-info") else info.name
            entry["installed_version"] = stem.rsplit("-", 1)[-1]
            metadata_path = info / "METADATA"
            raw_lines: list[str] = []
            if metadata_path.is_file():
                metadata_text = metadata_path.read_text(encoding="utf-8", errors="replace")
                raw_lines = [match.group(1).strip() for match in LICENSE_TEXT_RE.finditer(metadata_text)]
            entry["license_evidence"] = raw_lines or None
            joined = " ".join(raw_lines).lower()
            if "apache" in joined:
                entry["license"] = "Apache-2.0"
            elif "mit" in joined:
                entry["license"] = "MIT"
            elif "bsd" in joined:
                entry["license"] = "BSD-3-Clause"
            if entry["license"]:
                entry["license_source"] = f"{info.name}/METADATA"
        entries.append(entry)
    return {
        "venv_dir": str(venv_dir),
        "site_packages_present": site_packages.is_dir(),
        "runtime_dependencies": entries,
        "notes": [
            "Runtime set = packages imported by web_demo/ (BFF: fastapi, uvicorn, pydantic, mcp) and by the P17 MCP stdio server path (mcp, numpy, langgraph). Transitive entries are included because their licenses travel with the served environment.",
            "Versions are the ones actually installed in the project virtual environment at generation time; declared ranges live in requirements*.txt.",
            "PyTorch is an environment-specific dependency of the recommendation stack and is intentionally excluded here (requirements-cpu.txt; AGENTS.md §10/§11).",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--web", default=str(DEFAULT_WEB))
    parser.add_argument("--venv", default=str(DEFAULT_VENV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    web_dir = Path(args.web)
    manifest = {
        "$comment": "Machine-readable dependency + upstream-license manifest for the AgentRec-X Web Demo. Generated by third_party/tools/generate_dependency_manifest.py; human-readable attribution is THIRD_PARTY_NOTICES.md.",
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": "AgentRec-X",
        "task": "task-5 (upstream reuse & license compliance)",
        "upstream_repositories": UPSTREAM_REPOSITORIES,
        "npm": npm_dependencies(web_dir),
        "python": python_metadata(Path(args.venv)),
        "license_summary": {},
    }
    for section in ("direct_dependencies", "dev_dependencies"):
        counts: dict[str, int] = {}
        for dep in manifest["npm"][section]:
            counts[dep["license"] or "UNRESOLVED"] = counts.get(dep["license"] or "UNRESOLVED", 0) + 1
        manifest["license_summary"][f"npm_{section}"] = counts
    py_counts: dict[str, int] = {}
    for dep in manifest["python"]["runtime_dependencies"]:
        py_counts[dep["license"] or "UNRESOLVED"] = py_counts.get(dep["license"] or "UNRESOLVED", 0) + 1
    manifest["license_summary"]["python_runtime"] = py_counts

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[generate_dependency_manifest] wrote {out}")
    print(f"[generate_dependency_manifest] npm direct={len(manifest['npm']['direct_dependencies'])} "
          f"dev={len(manifest['npm']['dev_dependencies'])} "
          f"python={len(manifest['python']['runtime_dependencies'])}")
    unresolved = [d["name"] for d in manifest["npm"]["direct_dependencies"] + manifest["npm"]["dev_dependencies"] if not d["license"]]
    unresolved += [d["name"] for d in manifest["python"]["runtime_dependencies"] if not d["license"]]
    if unresolved:
        print("[generate_dependency_manifest] unresolved licenses: " + ", ".join(sorted(set(unresolved))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
