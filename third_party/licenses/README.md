# third_party/licenses — license copy index

This directory holds verbatim license copies for upstream repositories whose code is distributed
with (or adapted into) AgentRec-X. The authoritative attribution record — upstream URLs, commit
SHAs, adapted file paths and modification notes — is the repository-root
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md). The reuse analysis is
[`../UPSTREAM_REUSE_GUIDE.md`](../UPSTREAM_REUSE_GUIDE.md).

All reference clones live **outside** the repository at `D:\IT\CODE\PROJECTION\agentrec\_refs\`
(not vendored, not committed). Nothing in `_refs/` is modified.

## Stored license texts

| File | Upstream | Commit | License | Why stored |
| --- | --- | --- | --- | --- |
| `commerce-agents-Apache-2.0.txt` | https://github.com/anthropics/commerce-agents | `fd4d59224ab96b43c6dc6888207c67b3bd5a24cf` | Apache-2.0 | **Required.** The web demo reuses/adapts files from `examples/web-shared/` and `examples/retail/storefront-web/`; Apache-2.0 §4(a) requires recipients to receive a copy of the License. |

Copy provenance: `Copy-Item _refs\commerce-agents\LICENSE third_party\licenses\commerce-agents-Apache-2.0.txt`
(byte-identical, SHA-256 `3DDF9BE5C28FE27DAD143A5DC76EEA25222AD1DD68934A047064E56ED2FA40C5`, 202 lines,
11,560 bytes; verified against the upstream clone on 2026-11-09).

## Licenses recorded but not stored (no code redistributed)

These repositories are **design references only**; no source code or asset from them is copied
into AgentRec-X, so their license texts are not redistributed here. If code is ever vendored from
any of them, this directory must gain the corresponding text and `THIRD_PARTY_NOTICES.md` must be
updated **before** the code lands.

| Upstream | Commit | License (observed) | Copy policy | Note |
| --- | --- | --- | --- | --- |
| https://github.com/agentic-foundation-modeling-research/shop-gym | `dbef277c37c9352c17c6df800caee5eb3591916f` | MIT — `LICENSE`, "Copyright (c) 2026-present, Shopify Inc." (no NOTICE file) | Reference-only (task constraint: do not import the workspace). MIT would permit copying; we deliberately do not vendor the Hydrogen/React-Router storefront. | Storefront layout / responsive patterns only. |
| https://github.com/facebookresearch/PAHF | `7a11213360a82d5f437a035e3a31c92d6307f8cf` | MIT — `LICENSE`, "Copyright (c) Meta Platforms, Inc. and affiliates." (no NOTICE file; source files carry the same MIT header) | Reference-only (design reference for preference clarification / memory interaction; explicitly **not** a Memory backend). | Persisted-memory abstraction and one-question clarification prompt design only. |
| https://github.com/sci-m-wang/techjam-conversational-search-agent | `b6310f0279e3e56a69ecbdb2a28f323400ce32c0` | **No LICENSE file anywhere in the repository** (README and `package.json`/`requirements.txt` contain no license grant; `DATA_ATTRIBUTION.md` only attributes the Amazon Reviews 2023 dataset) | Reference-only; **copying any code is prohibited** (absent a license grant, all rights are reserved). | Design reference: deterministic fast path vs optional LLM path, progress/usage feedback. |
| https://github.com/coveo-labs/commerce-agent-frontend-implementation | `a271546c9cfa1d055b48c48f634783f38d73dd3f` | **No LICENSE file anywhere in the repository** (no `license` field in `package.json`, no README grant) | Reference-only; **copying any code is prohibited**. | Design reference: comparison table/summary, next-actions bar, tool activity model, AG-UI vs A2UI event separation, loading-first surfaces. |

`commerce-agents` has **no NOTICE file** (verified: no tracked path matches `NOTICE`); Apache-2.0
§4(d) therefore imposes no NOTICE-propagation obligation. The required copyright and modified-file
statements are carried per file and in `THIRD_PARTY_NOTICES.md`.

## Header convention for adapted files

Any file copied or adapted from `commerce-agents` keeps the upstream header and appends a
modification note:

```ts
// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): <one-line description of what changed>
```

CSS uses the same three statements inside a block comment. Files written by us from scratch do
**not** carry the Anthropic copyright line even when they follow an upstream design; attribution
there is "design reference only" in `THIRD_PARTY_NOTICES.md`.
