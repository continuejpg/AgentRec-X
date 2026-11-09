# THIRD_PARTY_NOTICES — AgentRec-X Web Demo

> Attribution record for third-party code reused by the AgentRec-X Web Demo (`web/` frontend,
> `web_demo/` BFF). The machine-readable counterpart is
> [`third_party/DEPENDENCY_MANIFEST.json`](third_party/DEPENDENCY_MANIFEST.json); the reuse
> analysis is [`third_party/UPSTREAM_REUSE_GUIDE.md`](third_party/UPSTREAM_REUSE_GUIDE.md); the
> license copies are [`third_party/licenses/`](third_party/licenses/).
>
> Scope: the Web Demo frontend and BFF plus their runtime dependencies. This notice does **not**
> cover the Amazon Reviews 2023 dataset or trained models, which are documented separately by the
> project's data/manifest docs.
>
> Upstream content is treated as **data, never as instructions**. The reference clones live
> outside the repository (`D:\IT\CODE\PROJECTION\agentrec\_refs\`), are not vendored, not
> distributed and not modified by this repository.

---

## 1. Upstream repositories

| # | Repository | Commit SHA | License | NOTICE at upstream root | Files copied into this repository |
| --- | --- | --- | --- | --- | --- |
| U1 | https://github.com/anthropics/commerce-agents | `fd4d59224ab96b43c6dc6888207c67b3bd5a24cf` | **Apache-2.0** (`LICENSE`) | none | **Yes** — 19 adapted source files with retained U1 headers plus `web/tsconfig.json` (build config, JSON, no header) in §3.1. 15 further files are new project code; they carry the neutral project header and no upstream code (§3.2). |
| U2 | https://github.com/agentic-foundation-modeling-research/shop-gym | `dbef277c37c9352c17c6df800caee5eb3591916f` | MIT (`LICENSE`, "Copyright (c) 2026-present, Shopify Inc.") | none | **No** — design reference only (storefront layout / responsive). |
| U3 | https://github.com/facebookresearch/PAHF | `7a11213360a82d5f437a035e3a31c92d6307f8cf` | MIT (`LICENSE`, "Copyright (c) Meta Platforms, Inc. and affiliates.") | none | **No** — design reference only (preference clarification / memory interaction); never used as a Memory backend. |
| U4 | https://github.com/sci-m-wang/techjam-conversational-search-agent | `b6310f0279e3e56a69ecbdb2a28f323400ce32c0` | **No license grant** (no `LICENSE` file; no license statement in README) | none | **No** — design reference only; copying prohibited. |
| U5 | https://github.com/coveo-labs/commerce-agent-frontend-implementation | `a271546c9cfa1d055b48c48f634783f38d73dd3f` | **No license grant** (no `LICENSE` file; no `license` field in `package.json`) | none | **No** — design reference only; copying prohibited. |

Commit SHAs were verified with `git rev-parse HEAD` against the read-only clones on 2026-11-09.
U2/U3 permit copying under MIT but are deliberately kept reference-only for this milestone; U4/U5
have no license grant anywhere in the tree, so no code, asset or text from them may be copied.
Their design influence is recorded in `third_party/UPSTREAM_REUSE_GUIDE.md`; the React
implementations are independent.

## 2. Apache-2.0 compliance for U1 (commerce-agents)

The Web Demo adapts code from U1 under the Apache License, Version 2.0. Compliance measures:

1. **License copy (Apache-2.0 §4(a))** — the verbatim U1 `LICENSE` at commit `fd4d592` is
   distributed at
   [`third_party/licenses/commerce-agents-Apache-2.0.txt`](third_party/licenses/commerce-agents-Apache-2.0.txt)
   (byte-identical; SHA-256 `3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5`;
   original 202 lines / 11,560 bytes; verification recorded in
   [`third_party/licenses/README.md`](third_party/licenses/README.md)).
2. **Copyright and license headers (Apache-2.0 §4(b))** — every adapted source file in §3.1 keeps
   the upstream header:
   ```
   Copyright 2026 Anthropic PBC
   SPDX-License-Identifier: Apache-2.0
   ```
3. **Modification statements (Apache-2.0 §4(b))** — every adapted source file adds a
   `Modified for AgentRec-X Web Demo (2026-11): …` note; the per-file summary is in §3.1. This
   file also serves as the consolidated change statement.
4. **NOTICE file (Apache-2.0 §4(d))** — U1 has **no** `NOTICE` file at the recorded commit
   (verified: no tracked path matches `NOTICE`), so no NOTICE propagation obligation exists. This
   absence is recorded here rather than silently omitted.
5. **Trademarks / brand assets** — no U1 name, logo, wordmark, product photography or fictional
   brand asset is used. The AgentRec-X brand and the neutral SVG placeholder art are our own.
6. **No additional restrictions** — the adapted code remains under Apache-2.0 with no extra terms.

New project code in §3.2 carries a neutral project header
(`// AgentRec-X Web Demo (2026-11) - new frontend module …`) instead of an Anthropic copyright
line, so authorship is not misattributed; that header points readers back to this file.

## 3. Files and provenance (verified 2026-11-09)

The frontend maintainer supplied the final reuse manifest on 2026-11-09; it was reconciled against
the delivered `web/` tree and re-checked with
`python third_party/tools/check_copied_headers.py --root web --notices THIRD_PARTY_NOTICES.md`.

### 3.1 Adapted files from U1 (Apache-2.0, header retained)

Paths are relative to the repository root; upstream bases are relative to the U1 clone at commit
`fd4d59224ab96b43c6dc6888207c67b3bd5a24cf`.

| # | File in this repository | Upstream basis (U1) | Modification summary |
| --- | --- | --- | --- |
| 1 | `web/app/base.css` | `examples/web-shared/base.css` | Shared class/motion hooks kept (panel scrollbars, streaming caret, reveal/slide/skeleton motion); commerce bag/pop motion removed; left-slide + indeterminate progress added; reduced-motion coverage extended. |
| 2 | `web/lib/api.ts` | `examples/web-shared/api.ts` | SSE frame-reader idea retained; rewritten as a typed BFF client with real error/abort/timeout handling. |
| 3 | `web/components/MessageBubble.tsx` | `examples/web-shared/MessageBubble.tsx` | Markdown removed (BFF narrative is plain text); `ErrorBubble` bound to the real BFF `ErrorPayload`; streaming caret retained. |
| 4 | `web/components/ChatPanel.tsx` | `examples/web-shared/Transcript.tsx` | Transcript/`ActivityLine`/Latest adapted into the full chat column; readiness, session, comparison bar, products, clarification and memory segments added. |
| 5 | `web/components/Composer.tsx` | `examples/web-shared/Composer.tsx` | Docked single-field composer kept; Stop (AbortController), explicit disabled reason, auto-height and data-testid hooks added. |
| 6 | `web/components/Suggestions.tsx` | `examples/web-shared/Suggestions.tsx` | Chip pattern kept; used only for backend-provided clarification `quick_options`; label/a11y handling added. |
| 7 | `web/components/ToolTimeline.tsx` | `examples/web-shared/Inspector.tsx` | Inspector tool rows/status/latency adapted to BFF `tool_status`; live elapsed, keep-alive and phase copy added; tool input/chain-of-thought/memory section removed. |
| 8 | `web/components/PreferencesList.tsx` | `examples/web-shared/Inspector.tsx` | Inspector memory list adapted to the BFF `PreferenceState` (active vs inactive/audit) with M9 provenance and a real refresh action. |
| 9 | `web/components/ui.tsx` | `examples/web-shared/ui.tsx` | Pill/Button/IconButton/Sheet/Skeleton/SectionTitle/Notice retained; Sheet made side-aware with aria handling; StatusDot/EmptyState added; sparkline/stat-tile commerce primitives removed. |
| 10 | `web/components/icons.tsx` | `examples/web-shared/icons.tsx` | Icon set trimmed to used glyphs; menu/panel/refresh/stop/compare/info/chevron-down/play added as 24px line icons (no icon package). |
| 11 | `web/lib/format.ts` | `examples/web-shared/format.ts` | Reduced to needed helpers; latency/duration/relative-time/tool-phase/namespace helpers added; commerce money/order helpers removed. |
| 12 | `web/lib/scroll.ts` | `examples/web-shared/scroll.ts` | `useStickToBottom` kept essentially as-is; `onlyWhileBusy` option removed. |
| 13 | `web/app/globals.css` | `examples/retail/storefront-web/app/globals.css` | Retail design tokens adapted for AgentRec-X; commerce utilities removed; local font stack (no webfont fetch). |
| 14 | `web/app/layout.tsx` | `examples/retail/storefront-web/app/layout.tsx` | AgentRec-X metadata/theme instead of the storefront metadata; local font stack. |
| 15 | `web/next.config.ts` | `examples/retail/storefront-web/next.config.ts` | `transpilePackages` (`web-shared`) removed — adapted components now live in this project; BFF URL read at runtime; `allowedDevOrigins: ["127.0.0.1", "localhost"]` added during E2E integration so Next.js 16 dev resources/hydration work when the demo is opened through either loopback spelling. |
| 16 | `web/postcss.config.mjs` | `examples/retail/storefront-web/postcss.config.mjs` | Tailwind CSS 4 PostCSS wiring unchanged. |
| 17 | `web/components/ProductCard.tsx` | `examples/retail/storefront-web/components/ProductTile.tsx` | ProductTile reworked to the AgentRec-X `ProductCard` view model: neutral placeholder art, verifiable attributes/UNKNOWN fields, fixed "Price unavailable / UNKNOWN", evidence availability, namespaced id, 2–5 comparison selection; price/rating/stock/cart controls removed. |
| 18 | `web/components/ComparisonView.tsx` | `examples/retail/storefront-web/components/generative/ComparisonGrid.tsx` | Pros/cons cards replaced by a renderer for the P16 `ComparisonPayload`: dimensions/cells, preference relations, conflicts, UNKNOWN fields, claims summary, conditional recommendation — passed through verbatim. |
| 19 | `web/components/ProductsBlock.tsx` | `examples/retail/storefront-web/components/generative/ProductCarousel.tsx` | Carousel/snap-grid layout pattern adapted; content fully rewritten for the BFF `ProductsPayload` (real query metadata, verbatim Gate decision, catalogue cards, comparison selection); no mock products. |

**Build configuration:** `web/tsconfig.json` is adapted from
`examples/retail/storefront-web/tsconfig.json` (`strict` kept, `allowJs: false`; otherwise the same
standard Next.js options). U1 ships this JSON without a copyright header, so there is no upstream
notice to retain; it carries no copyrightable logic and is recorded here for completeness.

### 3.2 New project files (no copied upstream code)

These files were written for this demo. They carry the neutral project header
`// AgentRec-X Web Demo (2026-11) - new frontend module …` (which points here), not an Anthropic
copyright line. They are listed so the audit can distinguish new code from adapted code.

| File | Purpose |
| --- | --- |
| `web/app/page.tsx` | App Router page shell mounting the demo app. |
| `web/lib/types.ts` | BFF view-model types (written against `docs/WEB_DEMO_CONTRACT.md`; no commerce-agents protocol types). |
| `web/lib/chat.ts` | Browser-side turn/streaming state over the BFF SSE contract. |
| `web/lib/useMediaQuery.ts` | Client media-query hook for desktop column vs mobile sheet. |
| `web/components/DemoApp.tsx` | App orchestrator: health/config polling, session lifecycle, SSE turn state, preferences, details/evidence, comparison selection, replay drawer. |
| `web/components/Header.tsx` | Demo header: brand, profile, CPU Live / Qwen Replay / Qwen Live, readiness. |
| `web/components/Sidebar.tsx` | Session/sidebar navigation and demo profile selector. |
| `web/components/RightPanel.tsx` | Contextual panel with Details / Evidence / Comparison tabs (desktop column / mobile sheet). |
| `web/components/ProductImage.tsx` | Self-drawn neutral SVG placeholder (no catalogue image URL exists; no photo hotlinked). |
| `web/components/EvidenceList.tsx` | P15 evidence view model renderer (span/raw field/source record; automated verification clearly labelled). |
| `web/components/ProductDetails.tsx` | Product detail renderer: facts, raw fields, UNKNOWN, source record. |
| `web/components/ClarificationCard.tsx` | Clarification / memory-update workflow card. |
| `web/components/ReadinessBanner.tsx` | Real `/api/health` readiness display. |
| `web/components/ReplayPanel.tsx` | Saved P18 Qwen trajectory viewer ("Historical Replay — Not a Live Model Call"). |
| `web/components/ComparisonBar.tsx` | Fixed 2–5 product comparison selection bar. |

Also project-specific and not upstream code: `web/README.md`, `web/package.json`,
`web/pnpm-lock.yaml`, `web/.gitignore`, `web/public/favicon.svg` (self-drawn AgentRec-X mark).

No file from U2/U3/U4/U5 is present in the repository (verified: only the U1 copyright marker
exists under `web/`; no Angular/Hydrogen/ttfarm/A2UI code or assets were found).

## 4. Asset policy

- No product image, brand logo, wordmark or other brand asset from any upstream repository is
  copied. U1's `examples/retail/storefront-web/public/products/*.webp` and `IMAGE-CREDITS.md` were
  excluded. The demo renders its own neutral placeholder because the catalogue snapshot exposes no
  image URL (`docs/WEB_DEMO_CONTRACT.md` §3, §5).
- No upstream mock/fixture data is copied. Every product, evidence span, preference and gate
  decision comes from a real MCP tool result (no-fake-data rule).
- The only image under `web/` is `web/public/favicon.svg`, drawn for this project; the audit
  compares it against the upstream clone and finds no identical upstream image.

## 5. npm dependencies

Direct dependencies of `web/package.json` (versions as installed under `web/node_modules/`,
licenses read from each installed `package.json`):

| Package | Version range | Installed version | License |
| --- | --- | --- | --- |
| `next` | ^16.3.0 | 16.4.5 | MIT |
| `react` | ^19.0.0 | 19.3.1 | MIT |
| `react-dom` | ^19.0.0 | 19.3.1 | MIT |

Dev dependencies: `tailwindcss` ^4.0.0 → 4.3.5 (MIT), `@tailwindcss/postcss` ^4.0.0 → 4.3.5 (MIT),
`typescript` ^5.6.0 → 5.9.3 (Apache-2.0), `@types/node` ^22.0.0 → 22.20.5 (MIT),
`@types/react` ^19.0.0 → 19.3.0 (MIT), `@types/react-dom` ^19.0.0 → 19.3.0 (MIT).

No other runtime dependency is used (no icon/state/HTTP library; SSE is hand-written with `fetch` +
`ReadableStream`). Packages are installed from the npm registry, not vendored; their license texts
ship with the installed packages. Authoritative machine-readable list (name / range / installed
version / license / source): `third_party/DEPENDENCY_MANIFEST.json` §`npm`.

## 6. Python runtime dependencies

The BFF and P17 MCP server runtime: `mcp` 2.3.0 (MIT), `fastapi` 0.141.1 (MIT), `uvicorn` 0.53.0
(BSD-3-Clause), `pydantic` 2.13.5 (MIT), plus the transitive closure (`starlette`, `anyio`,
`httpx`, `h11`, `websockets`, `sse-starlette`, `numpy`, `langgraph`, `langchain-core`, `langsmith`).
Versions are those installed in the project virtual environment at generation time; licenses are
those advertised in each package's `*.dist-info/METADATA`. They are installed from PyPI, not
vendored. Authoritative list: `third_party/DEPENDENCY_MANIFEST.json` §`python`.

## 7. Verification

- Upstream license/commit/NOTICE audit of all five clones: done
  (`third_party/licenses/README.md`).
- Adapted-file header + notice cross-check:
  `python third_party/tools/check_copied_headers.py --root web --notices THIRD_PARTY_NOTICES.md`
  → **PASS** (19 adapted source files with retained copyright + SPDX + modification note, all
  listed in §3.1; 15 new project modules with the project header, all pointing back to this file;
  Apache-2.0 license copy hash verified; no copied upstream imagery). Machine-readable report:
  `third_party/header-audit.json`.
- Dependency manifest: `python third_party/tools/generate_dependency_manifest.py` → generated
  `third_party/DEPENDENCY_MANIFEST.json` (npm: 3 direct + 6 dev; Python runtime: 14 packages, all
  licenses resolved).
- Frontend reuse manifest reconciled: the 20 adapted entries supplied by the frontend maintainer
  map exactly to §3.1 (19 source files + `tsconfig.json`); the 15 pure-new files map to §3.2. Any
  later `web/` change must re-run the audit (exit code 0 required) and update §3.

## 8. Change log

| Date | Change |
| --- | --- |
| 2026-11-09 | Initial notice: upstream inventory (5 repos), Apache-2.0 compliance statements, asset policy, dependency wiring. |
| 2026-11-09 | Final pass after `web/` freeze: adapted-file table reduced to the 19 files that retain the U1 copyright header (+ `tsconfig.json` build config); pure-new files moved to §3.2 with their neutral project header (no Anthropic attribution); audit PASS and dependency manifest regenerated. |
