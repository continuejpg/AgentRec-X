# third_party — upstream reuse & license compliance assets

These files are the compliance record for the AgentRec-X Web Demo's reuse of open-source
upstream code. They do not contain vendored upstream source; adapted source lives in `web/`
under the ownership of the frontend milestone.

| Path | Purpose |
| --- | --- |
| [`licenses/`](licenses/) | Verbatim license texts. `commerce-agents-Apache-2.0.txt` is the required Apache-2.0 copy for the adapted files; [`licenses/README.md`](licenses/README.md) records the audit for all five upstream repositories (including the ones not copied). |
| [`UPSTREAM_REUSE_GUIDE.md`](UPSTREAM_REUSE_GUIDE.md) | Per-repository reuse analysis: verified commit SHA, license, NOTICE status, stack compatibility, component-by-component mapping onto `web/`, and the copy/reference boundary (Apache-2.0 = copy allowed; MIT = reference only here; no-license = reference only, copying prohibited). |
| [`DEPENDENCY_MANIFEST.json`](DEPENDENCY_MANIFEST.json) | Machine-readable dependency manifest: upstream repositories + `web/package.json` direct npm dependencies + Python BFF/MCP runtime dependencies, each with name/version/license/source. |
| [`tools/generate_dependency_manifest.py`](tools/generate_dependency_manifest.py) | Regenerates `DEPENDENCY_MANIFEST.json` from `web/package.json`, installed `web/node_modules`, and the project virtual environment (`*.dist-info/METADATA`). |
| [`tools/check_copied_headers.py`](tools/check_copied_headers.py) | Audits `web/`: every file carrying the upstream copyright must keep the Apache-2.0 SPDX header + a modification note and appear in `THIRD_PARTY_NOTICES.md`; verifies the license copy hash; flags copied upstream imagery. Exit code 0 = pass, 1 = compliance issue, 2 = tree missing. |

The human-readable attribution record is the repository-root
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Quick check:

```powershell
python third_party/tools/check_copied_headers.py --root web --notices THIRD_PARTY_NOTICES.md
python third_party/tools/generate_dependency_manifest.py
```

Upstream reference clones are **not** part of this repository; they live read-only under
`D:\IT\CODE\PROJECTION\agentrec\_refs\` (outside the repo) and are never modified or committed.
