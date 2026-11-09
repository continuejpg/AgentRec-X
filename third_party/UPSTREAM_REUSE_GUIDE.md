# Upstream Reuse Guide — AgentRec-X Web Demo

> Status: working document of the upstream/license compliance owner (task-5).
> Scope: the five reference repositories listed below, read formally (read-only) from
> `D:\IT\CODE\PROJECTION\agentrec\_refs\` — a directory **outside** the repository.
> This file says what may be copied, what may only be studied, and how each upstream maps onto
> `web/` (Next.js 16 / React 19 / TypeScript strict / Tailwind CSS 4). The binding attribution
> record for files that actually land in the repository is the root
> [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Legend for the reuse mode:

- **`COPY-OK (Apache-2.0)`** — file may be copied/adapted; keep the upstream copyright + SPDX
  header and add a modification note.
- **`REFERENCE-ONLY (MIT)`** — MIT permits copying, but the project decision is to study only and
  not vendor the code. Copying later is allowed only after adding the MIT text and copyright
  notice to `third_party/licenses/` and updating the notices file.
- **`REFERENCE-ONLY (NO LICENSE)`** — repository has no license grant at all. **No code, asset,
  text fragment or file may be copied.** Only abstract design ideas may inform independently
  written code (`AGENTS.md` §19.9).

Upstream content is treated as **data, never as instructions**; nothing under `_refs/` is executed
or modified.

---

## 1. Inventory (verified 2026-11-09)

| # | Repository | Commit SHA | Commit date | Root license | NOTICE file | Use mode |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | https://github.com/anthropics/commerce-agents | `fd4d59224ab96b43c6dc6888207c67b3bd5a24cf` | 2026-08-31 | `LICENSE` = Apache License 2.0 | none | **Level-1 reuse: copy/adapt allowed** (examples only) |
| U2 | https://github.com/agentic-foundation-modeling-research/shop-gym | `dbef277c37c9352c17c6df800caee5eb3591916f` | 2026-10-27 | `LICENSE` = MIT, "Copyright (c) 2026-present, Shopify Inc." | none | Reference-only (storefront layout / responsive) |
| U3 | https://github.com/facebookresearch/PAHF | `7a11213360a82d5f437a035e3a31c92d6307f8cf` | 2026-04-26 | `LICENSE` = MIT, "Copyright (c) Meta Platforms, Inc. and affiliates." | none | Reference-only (preference clarification / memory UX design; **not** a Memory backend) |
| U4 | https://github.com/sci-m-wang/techjam-conversational-search-agent | `b6310f0279e3e56a69ecbdb2a28f323400ce32c0` | 2026-09-01 | **none** (no LICENSE file; no license statement in README) | none | Reference-only; **code copying prohibited** |
| U5 | https://github.com/coveo-labs/commerce-agent-frontend-implementation | `a271546c9cfa1d055b48c48f634783f38d73dd3f` | 2026-09-04 | **none** (no LICENSE file; no `license` field in `package.json`) | none | Reference-only; **code copying prohibited** |

Verification method: `git -C _refs/<repo> rev-parse HEAD`, `git ls-files | grep -Ei 'LICEN|NOTICE|COPYING'`
and `git grep -n -E 'Copyright|SPDX-License-Identifier|Licensed under'`. U1's example files all
carry `// Copyright 2026 Anthropic PBC` + `// SPDX-License-Identifier: Apache-2.0`.
U4 additionally contains `DATA_ATTRIBUTION.md`, which attributes its competition **data** to
Amazon Reviews 2023 — it is not a code license.

## 2. Ground rules

1. **Copy only from U1.** U2/U3 are MIT but are deliberately kept reference-only here; U4/U5 have
   no license grant and are never copied. This is stricter than the licenses require and matches
   task-5 and `AGENTS.md` §19.9.
2. **Attribution is per file, not per repository.** A copied/adapted U1 file keeps:
   ```ts
   // Copyright 2026 Anthropic PBC
   // SPDX-License-Identifier: Apache-2.0
   // Modified for AgentRec-X Web Demo (2026-11): <what changed>
   ```
   CSS: the same three statements in a `/* ... */` block. Files written from scratch get no
   Anthropic copyright line; they are listed as design references only.
3. **No product images or brand assets.** U1 ships `examples/retail/storefront-web/public/products/*.webp`
   plus `IMAGE-CREDITS.md`, and its demo brand ("ACME", "ACME Assistant") is fictional. None of
   this is copied: AgentRec-X renders its own neutral SVG placeholder because the P13-R catalogue
   snapshot has no image URL (`docs/WEB_DEMO_CONTRACT.md` §5, §3 `image_url: null`).
4. **No invented commerce data.** U1 components embed prices, ratings, review counts, stock,
   delivery promises and a cart/checkout flow. Those parts must be removed during adaptation: the
   AgentRec-X contract fixes price to `Price unavailable / UNKNOWN` and forbids cart/checkout,
   ratings, stock and brand claims (§0, §5).
5. **License copy location.** Apache-2.0 text: `third_party/licenses/commerce-agents-Apache-2.0.txt`
   (verbatim copy of U1 `LICENSE`, SHA-256 verified). MIT texts for U2/U3 are not stored because
   no code is redistributed from them (see `third_party/licenses/README.md`).
6. **No NOTICE obligation for U1** (no NOTICE file upstream); Apache-2.0 §4(d) does not apply.
   Apache-2.0 §4(b)/(c) are satisfied by the retained per-file headers and the modification notes.

## 3. U1 — commerce-agents (Apache-2.0) — primary reuse source

### 3.1 Stack compatibility with `web/`

| Aspect | U1 `storefront-web` | AgentRec-X `web/` (contract §5) | Verdict |
| --- | --- | --- | --- |
| Framework | Next.js `^16.3.0`, App Router, `"use client"` islands | Next.js 16, App Router | compatible |
| React | `^19.0.0` | React 19 | compatible |
| Language | TypeScript strict, `@/*` path alias | TypeScript strict | compatible |
| Styling | Tailwind CSS 4 (`@import "tailwindcss"`, `@tailwindcss/postcss`), CSS custom-property tokens, arbitrary values like `text-(--ink)` | Tailwind CSS 4, tokenized CSS variables, light theme | compatible |
| Shared code | separate private package `examples/web-shared` linked as `"web-shared": "file:../../web-shared"`, enabled via `transpilePackages: ["web-shared"]` | single `web/` app | **copy the files into `web/`** (e.g. `web/components/upstream/` + `web/lib/upstream/`), add `@source` for the new directory, or mirror the package layout; do **not** keep a `file:` link to `_refs/` |
| Data / protocol | `protocol.ts` mirrors `commerce_common/streaming.py` events (`text_delta`, `tool_call`, `tool_result`, `ui`, `progress`, `cart_update`, …); `turn.ts` reducer; `api.ts` client | BFF SSE events: `tool_status`, `text`, `products`, `evidence`, `comparison`, `clarification`, `memory_update`, `done`, `error` (contract §2) | **incompatible — rewrite types/reducer**; only the rendering layer is reused |
| Commerce semantics | cart, checkout, orders, price intelligence, ratings, stock, delivery | no cart/checkout; price UNKNOWN; catalogue facts only; no re-ranking | **strip on adaptation** |
| Fonts | `next/font/google` Instrument Sans (network at build) | offline Windows dev | prefer system font stack; do not add a build-time network dependency |
| Assets | `public/products/*.webp`, image credits | no product images; own SVG placeholder | do not copy |

### 3.2 `examples/web-shared/` components

| Upstream file | What it provides | Reuse mode | Mapping / adaptation notes |
| --- | --- | --- | --- |
| `MessageBubble.tsx` (32 lines) | `UserBubble`, `AssistantText` (streaming caret), `ErrorBubble` | **COPY-OK** | → `web/components/chat/MessageBubble.tsx`. Keep classes `.user-bubble` / `.streaming-caret`; error bubble maps to the `error` SSE event. |
| `Transcript.tsx` (101 lines) | chat transcript: maps `ChatItem[]`/`UISegment`s, busy state, `ActivityLine`, suggestions slot | **COPY-OK (adapt)** | → `web/components/chat/Transcript.tsx`. Replace `ChatItem`/`UISegment` protocol with our `TurnView` + `tool_activity` + typed blocks (`products`/`comparison`/`clarification`); keep the activity-line + text + block composition. |
| `Composer.tsx` (99 lines) | one field with send arrow inside, `Prefill` + `nonce` (same text twice), IME/Enter handling, disabled/streaming state | **COPY-OK (adapt)** | → `web/components/chat/Composer.tsx`; fixed bottom on mobile. Add our selected-product context (`selected_product_ids`) as chips if the design keeps them; never send `user_id`/`confirmed_by_user` (contract §0). |
| `Suggestions.tsx` (32 lines) | chip row of offered prompts/quick actions | **COPY-OK (adapt)** | → suggested **real** actions only (e.g. "View evidence", "Compare selected"), derived from the last real payload; contract §5 "suggested real actions". |
| `Inspector.tsx` (274 lines) | right-panel tool trace: `buildToolRows` from `TraceEntry[]`, run/ok/error status rows, input/output excerpts, memory-fact section, `Panel` shell | **COPY-OK (adapt)** | → right contextual panel "Tool activity / Evidence" tab. Feed only real `tool_status` events (tool, label, status, latency, detail); never synthesize results. The memory-facts section informs the separate "Your active preferences" block, whose data must come from `get_active_preferences`. |
| `Markdown.tsx` (131 lines) | *"Renders React text nodes only, so model- and catalog-derived text stays inert"* — bold/italic/code/bullets/quotes/tables/HR through structured React nodes (no HTML injection), uses `QuotedAsData` | **COPY-OK** | → `web/components/chat/Markdown.tsx`. Strong fit for catalogue-derived titles/evidence; contract only requires plain narrative text, so use it as a safe renderer, not a requirement. |
| `QuotedAsData.tsx` (22 lines) | renders externally-quoted text as visibly quoted data | **COPY-OK** | → evidence `span_text`, `raw_excerpt`, clarification source text. |
| `tool-copy.ts` (52 lines) | tool-name → human progress copy (`"Searching the catalog…"`, query interpolation for query tools) | **COPY-OK (adapt)** | → map the 7 P17 tools to the contract's labels: Search / Recommend / Product Info / Compare / Clarification / Memory / Evidence. Keep the "fallback = humanize tool name" safety branch. |
| `base.css` (131 lines) | shared rules: `.panel-scroll` scrollbars, `.streaming-caret` (`ac-blink`), motion classes `ac-reveal` / `ac-slide-in-right` / `ac-pop` / `ac-collapse`, `.ac-skeleton` shimmer, `prefers-reduced-motion` handling | **COPY-OK** | → `web/app/upstream/base.css` (or merge into `web/app/globals.css` after Tailwind). Our globals must define the tokens it reads (`--ground`, `--ink`, `--line-strong`, `--well`, `--card`). Motion/reduced-motion behavior is a genuine accessibility win. |
| `ui.tsx` (451 lines) | primitives: `Pill`, `KindIcon`, `Thumb`, `Panel`, `Segmented`, **`Sheet` (mobile drawer)**, `Skeleton`, `Notice`, `IconButton`, `Button`, `Facts`, `SectionTitle` | **COPY-OK (adapt selectively)** | → `Sheet` is the natural basis for the contract's mobile Sheets/Drawers; `Pill` for MATCH/VIOLATION/UNKNOWN & mode badges; `Panel` for the right contextual panel; `Segmented` for CPU Live / Qwen Replay tabs. Remove merchant-portal-specific tones/icons not used. |
| `icons.tsx` (152 lines) | inline SVG icon set (no external assets, no brand logos) | **COPY-OK (selective)** | → `web/components/icons.tsx`. Keep only used icons to reduce audit surface. |
| `format.ts` (218 lines) | money/rating/plural/option formatting helpers | **REFERENCE / partial** | Our price display is a fixed `Price unavailable / UNKNOWN` string; do **not** import money/rating helpers into the UNKNOWN path. Keep generic helpers (plural, truncation) if needed. |
| `protocol.ts` (110 lines), `turn.ts` (495 lines), `api.ts`, `session.ts`, `resource.ts`, `catalog.ts` | U1 event model + client reducer + commerce API bindings | **DO NOT COPY — rewrite** | Mirror `docs/WEB_DEMO_CONTRACT.md` §2 event names and view-model shapes; the BFF owns the truth. Keep `turn.ts` only as a *reference* for reducer shape (pending/running/complete per tool). |
| `generative.tsx`, `scroll.ts`, `ActivityButton.tsx` | generative-block fallback, scroll helper, activity button | **REFERENCE** (small) | pattern-level; the unknown-block fallback idea is useful for our block union. |
| `portal/*` (`AssistantPanel`, `AssistantRail`, `Shell`, `cards`, `home`, `merchant.ts`, `portal.css`), `storefront/bag.tsx`, `storefront/frame.ts`, `storefront/orders.tsx`, `storefront/home.tsx`, `storefront/Shell.tsx` | merchant console / cart / orders / store shell | **DO NOT REUSE** | cart/checkout/order/policy semantics are out of scope; contract forbids them. |
| `storefront/Chat.tsx` (50 lines) | thin wrapper: while a search runs, shows `ActivityLine` + a row of 4 skeleton cards where the carousel will land | **COPY-OK (pattern)** | → pending-search skeleton in our chat; must disappear when the real `products` event arrives and must not fabricate product data. |
| `package.json` | private `web-shared` package (peer deps react/react-dom ^19) | informational | No workspace dependency should be added to `web/`; copy files instead. |

### 3.3 `examples/retail/storefront-web/` components

| Upstream file | What it provides | Reuse mode | Mapping / adaptation notes |
| --- | --- | --- | --- |
| `app/globals.css` (97 lines) | ACME retail design system: full token set (`--ground`, `--surface`, `--card`, `--chrome`, `--well`, `--ink*`, `--line*`, `--accent*`, `--ok`, `--warn`, `--danger`, `--info`, `--violet`, `--star`, shadows, `--radius`) + `.user-bubble`, `.chip`, `.btn-primary`, body font | **COPY-OK (adapt colors)** | → `web/app/globals.css` token block, re-themed for AgentRec-X (light, calm, no ACME terracotta identity required). Keep the token *names/roles* — they make `base.css` + copied components work unchanged. `@source`/`@import` paths must point at wherever the copied components live. |
| `app/layout.tsx` (25 lines) | App Router metadata + `next/font/google` Instrument Sans | **COPY-OK (adapt)** | → `web/app/layout.tsx` metadata shell; replace the Google font with a system stack (offline build). |
| `components/ProductTile.tsx` (280 lines) | `ProductTitle` (keeps trailing "(48-Pack)" unbreakable), `ProductImage` (+ fallback), `Rating`, `OptionLine`, `AddButton`, `DeliveryPromise`, `LowStockChip`, `ProductTile`, `ProductRow` | **COPY-OK (adapt, heavily)** | → `web/components/products/ProductTile.tsx`. Keep card shell/layout/hover/animation and the neutral image fallback; **remove** cart add-button, price/rating/review/stock/delivery/options logic; add our fields (product_id + namespaced_id, product_type/form_factor/product_role, retrieval_source, match_status, verifiable attributes, UNKNOWN list, `Price unavailable / UNKNOWN`) and buttons View details / View evidence / Select for comparison. |
| `components/generative/ComparisonGrid.tsx` (112 lines) | comparison **cards** per product: recommended highlight, pros/cons rows aligned with CSS subgrid (sign column so text shares a left edge), partial-load skeleton, price delta, "Compared on: …" dimension footer | **COPY-OK (adapt)** | → basis for `web/components/comparison/ComparisonGrid.tsx`. Our contract needs a dimension × product **table** with MATCH/VIOLATION/UNKNOWN (Coveo's table layout is the closer design reference), so reuse the card shell, recommended flag, skeleton-partial pattern and dimension footer; replace pros/cons with P16 dimension cells and add `conflicts`, `preference_relations`, `evidence_sources`, `final_answer` blocks. |
| `components/generative/ProductCarousel.tsx` (373 lines) | horizontal scroller with overflow detection, prev/next buttons, `aria-label`s, keyboard-friendly, product detail expansion; also price-intelligence/review-aspects/variant blocks | **COPY-OK (adapt)** | → `web/components/products/ProductCarousel.tsx` (or vertical grid on mobile). Keep scroller mechanics + a11y; strip price intelligence, review synthesis, variants, add-to-cart; items are our real `ProductCard`s. |
| `components/generative/index.tsx` (46 lines) | maps a block/component name to a component, with fallback for unknown blocks | **REFERENCE** | pattern for our SSE block union + unknown-block safety. |
| `components/generative/{CheckoutSummary,OrderStatusCard,PlanChecklist,GuideCard}.tsx`, `components/CartPanel.tsx`, `lib/flight.ts`, `lib/orders.tsx`, `lib/showcase-fixtures.ts`, `lib/storePolicy.ts` | checkout/order/plan/guide/cart + **mock fixtures** | **DO NOT REUSE** | out of contract scope; `showcase-fixtures` is fabricated data and must never enter the demo (no fake data rule). |
| `lib/types.ts` (125 lines), `lib/format.ts` (80 lines) | UI types/format helpers | **REFERENCE only** | our view models are defined by the BFF contract; do not import commerce shapes. |
| `next.config.ts` (8 lines) | `reactStrictMode`, `transpilePackages: ["web-shared"]` | informational | only needed if the copied `web-shared` files stay as a separate local package. |
| `app/page.tsx` (92 lines), `components/Chat.tsx` (50 lines), `components/views/HomeView.tsx`, `app/showcase/page.tsx`, `lib/api.ts` | storefront page wiring / cart panel / showcase | **REFERENCE** | our page wiring is driven by the BFF contract (sessions, SSE, config); do not copy cart/home logic. |

### 3.4 What is explicitly *not* carried over from U1

- Any brand/logo/wordmark (ACME, ACME Assistant) and the `public/products/*.webp` photography.
- Cart, checkout, orders, delivery promises, stock, rating stars, review counts, variant options,
  price-intelligence rows, policy/store-locator surfaces.
- `showcase-fixtures.ts` and every other fabricated catalogue/answer.
- The commerce streaming protocol and client reducer (ours is the BFF contract).

## 4. U2 — shop-gym (MIT) — storefront layout / responsive reference only

Read-only study targets:

- `packages/shop_arena/src/shop_arena/gen/templates/hydrogen/app/components/PageLayout.tsx`,
  `HeaderShell.tsx`, `MobileNavDrawer.tsx`, `Aside.tsx`, `SearchForm.tsx`, `ProductItem.tsx`.
- `…/hydrogen/app/styles/app.css` — responsive token set: `--aside-width: 400px`,
  `--grid-item-width: 355px`, `--header-height: 64px`, layered `--z-overlay`/`--z-aside`/
  `--z-header`, plus breakpoint rules; `…/styles/reset.css`.

What we take (design only): desktop three-zone shell (header / main / aside), a mobile nav drawer
pattern, fixed header height variables, z-index layering for overlays and sheets, and card-grid
sizing variables. This complements the contract's desktop-sidebar + center chat + right panel and
mobile Sheet/Drawer layout.

What makes the code non-portable: Hydrogen `2026.1.1` / React Router `7.12` / React 18 /
GraphQL Storefront API / Oxygen worker scaffolding — a different framework generation from
Next.js 16 + React 19, and a Shopify storefront data model we do not have. **No code copied**
(task constraint: do not import the workspace); therefore no MIT text is redistributed. If this
ever changes, add `LICENSE` (Shopify Inc., MIT) to `third_party/licenses/` and list each copied
file first.

## 5. U3 — PAHF (MIT) — preference clarification / memory interaction reference only

Read-only study targets:

- `memory/banks.py` — memory-bank abstraction (`FAISSMemoryBank`, `SQLiteMemoryBank`) with
  store/update/retrieve semantics; `memory/utils.py` — retrieval + formatting of memory entries.
- `prompts/shopping_prompts.py` — the clarifying-question policy: *"WHEN TO ASK FOR
  CLARIFICATION"* (subjective preference, missing feature information, no relevant memory, ambiguous
  task), "Ask human" action, and **"you can ask the human AT MOST ONE QUESTION per task"**.
- `agents/shopping_agent.py` — ReAct loop that searches memory before asking/deciding.

What we take (design only): the "ask exactly one focused question, then decide" interaction rule,
the separation between memory **storage**, **retrieval**, and **prompt formatting**, and the idea
of recording *why* clarification was needed (our `ClarificationPayload.reason` =
`missing_key_need`). This informs the contract's clarification card and the "active vs inactive /
audit history" preference presentation.

What we do **not** adopt: PAHF as a Memory backend or code dependency (faiss/torch/transformers,
LLM-driven ReAct, research datasets). Contract §0: M9/P16 are the only memory sources, no new
runtime/model. U3 files carry `Copyright (c) 2017-present, Facebook, Inc.` + MIT header; since no
code is copied, no MIT text is redistributed. If copied later: add MIT text + headers + notices.

## 6. U4 — techjam-conversational-search-agent (NO LICENSE) — Fast/LLM path & progress design only

Read-only study targets:

- `ttfarm/agent.py` docstring — three tiers: Tier 0 deterministic pipeline (default, no model key),
  Tier 1 optional model refinement of the parse, Tier 2 abnormal-session handover to an LLM agent;
  *without a key the behavior is byte-identical to the pure deterministic agent*.
- `ttfarm/policy.py` — `choose_ask` (when to ask a clarifying attribute question), `choose_k`,
  `compose_message`; `ttfarm/nlu.py` — observation/layer classification; `ttfarm/escalation.py` —
  abnormal-session handover with a case file (profile, transcript, proven-wrong products).
- `docs/agent_api_contract.json` — turn contract: `ask_attribute` enum (category/material/color/
  size/style/brand/budget/feature/use_case/other), `recommendations[]`, `usage{prompt_tokens,
  completion_tokens}`; `docs/baseline_results.json`, `docs/submission_report.md`.

What we take (design only): the deterministic-first / LLM-never-required tiering (our demo
controller is deterministic, contract §2 "no LLM"), explicit "is a clarification question needed
before recommending?" policy as a separate decision, and per-turn progress/usage telemetry in the
response. The `ask_attribute` enum is a useful checklist when deciding whether a real P16 question
supports fixed choices — contract: `quick_options` stays `[]` unless the real question has fixed
choices.

**No code may be copied** (no license = all rights reserved). Ideas are reimplemented
independently; this file records them. Its data is Amazon Reviews 2023-derived
(`DATA_ATTRIBUTION.md`) and is not used at all.

## 7. U5 — coveo-commerce-agent-frontend (NO LICENSE) — commerce surface / event-separation design only

Read-only study targets (Angular 21 + RxJS + `@ag-ui/client`):

- `src/app/components/product-carousel.component.ts` — carousel surface with controls, loading
  skeletons, staggered reveal animations, explicit aria labels.
- `src/app/components/comparison-table.component.ts` — dimension × product comparison grid with a
  **loading-first skeleton** version of the same grid.
- `src/app/components/comparison-summary.component.ts` — "Assistant recommendation" summary
  surface for the comparison result.
- `src/app/components/next-actions-bar.component.ts` — suggested next steps as buttons (+ loading
  placeholders).
- `src/app/components/skeleton.component.ts`, `product-research-card.component.ts`,
  `transcript-panel.component.ts`, `surface-outlet.component.ts`.
- `src/app/models.ts` — AG-UI event union + A2UI operations + surface shapes with per-surface
  `isLoading`; `src/app/a2ui-parser.ts` — maps streamed A2UI ops to ordered renderable surfaces;
  `src/app/services/ag-ui-client-transport.service.ts` — live SDK transport kept separate from the
  mock transport; `src/app/conversation.interfaces.ts` — `ToolActivity = {id, name, status:
  'running'|'completed', argsPreview, resultPreview}`.

What we take (design only): (a) **tool-activity model** — id/name/status/args-preview/results-
preview is essentially our `tool_status` event shape; (b) **loading-first surfaces** — render the
skeleton of the surface that is expected, then swap in real content only when the event arrives
(our `products`/`comparison` blocks, pending tool states); (c) `ComparisonTable` +
`ComparisonSummary` + `NextActionsBar` as the three-part comparison result; (d) **event separation**
— protocol/transport vs surface-rendering adapter vs component layer (our SSE reducer should stay
separate from the components); (e) unknown-surface fallback.

**No code may be copied** (no license). Angular syntax/templates and `@ag-ui/client` are not
dependencies of `web/`; the React implementation must be written independently.

## 8. Frontend mapping recommendation (target: `web/`)

`web/` is owned by the frontend teammate; actual paths will be recorded in
`THIRD_PARTY_NOTICES.md` once the final file list arrives. Suggested mapping:

| `web/` responsibility | Upstream basis | Mode |
| --- | --- | --- |
| design tokens + base styles (`app/globals.css`, `app/upstream/base.css`) | U1 `storefront-web/app/globals.css` + `web-shared/base.css` | COPY-OK |
| chat bubbles, transcript, composer, suggestions | U1 `web-shared/MessageBubble.tsx`, `Transcript.tsx`, `Composer.tsx`, `Suggestions.tsx` | COPY-OK (adapted to BFF view models) |
| markdown/quoted external text | U1 `web-shared/Markdown.tsx`, `QuotedAsData.tsx` | COPY-OK |
| tool activity timeline + right inspector panel | U1 `web-shared/Inspector.tsx`, `tool-copy.ts` + U5 `ToolActivity` model | COPY-OK (U1) / REFERENCE (U5) |
| product card / placeholder art / carousel | U1 `ProductTile.tsx`, `ProductCarousel.tsx` | COPY-OK (strip commerce data) |
| comparison table + summary + next actions | U1 `ComparisonGrid.tsx` shell + U5 table/summary/actions design | COPY-OK (U1) / REFERENCE (U5) |
| mobile sheets / drawers / buttons / pills | U1 `web-shared/ui.tsx` (`Sheet`, `Pill`, `Panel`, `Segmented`, `Skeleton`) | COPY-OK |
| layout: header/sidebar/aside/mobile nav | U2 Hydrogen `PageLayout` / `HeaderShell` / `MobileNavDrawer` / `app.css` | REFERENCE (MIT) |
| clarification card + "one question" flow | U3 clarification prompt policy + U4 `ask_attribute`/`choose_ask` | REFERENCE (MIT / no-license) |
| skeleton-first loading surfaces, event/transport separation | U5 `a2ui-parser`, comparison table skeleton, transport split | REFERENCE (no license) |
| streaming demo controller / tiering / progress telemetry | U4 `ttfarm/agent.py` tiers + `docs/agent_api_contract.json` | REFERENCE (no license) |

## 9. Attribution obligations checklist

For every copied/adapted U1 file (checked by `third_party/tools/check_copied_headers.py`):

- [ ] `Copyright 2026 Anthropic PBC` retained;
- [ ] `SPDX-License-Identifier: Apache-2.0` retained;
- [ ] `Modified for AgentRec-X Web Demo (2026-11): …` note added;
- [ ] file listed in `THIRD_PARTY_NOTICES.md` with upstream path and modification summary;
- [ ] no product images / brand assets / mock fixtures copied;
- [ ] Apache-2.0 text present at `third_party/licenses/commerce-agents-Apache-2.0.txt`.

For U2/U3 (MIT) and U4/U5 (no license): no file may appear in the vendored section; they are
recorded only as design references. If the policy for U2/U3 changes, the MIT text + copyright
notice must be added before the code lands.

## 10. Verification (recorded 2026-11-09)

- License copies: copy compared against `_refs/commerce-agents/LICENSE` (byte-identical, SHA-256
  `3DDF9BE5C28FE27DAD143A5DC76EEA25222AD1DD68934A047064E56ED2FA40C5`, 202 lines / 11,560 bytes).
- Upstream repos: `git rev-parse HEAD`, tracked `LICENS*`/`NOTICE*`/`COPYING*` scan, and
  `git grep -n -E 'Copyright|SPDX-License-Identifier|Licensed under'` — all five SHAs match the
  task brief; U1 has no NOTICE; U4/U5 have no license file.
- Vendor-tree headers: `python third_party/tools/check_copied_headers.py --root web --notices
  THIRD_PARTY_NOTICES.md` → **PASS**: 19 adapted source files carry the retained copyright + SPDX
  + modification note and are listed in §3.1 of the notices; 15 new project modules carry the
  neutral project header (no Anthropic attribution) and point back to the notices; the Apache-2.0
  license copy hash matches; no copied upstream imagery. Machine-readable report:
  `third_party/header-audit.json`.
- Frontend reuse manifest (delivered 2026-11-09 at `web/` freeze): 20 adapted entries (19 source
  files + `tsconfig.json`) + 15 new files, reconciled one-to-one against the tree.
- Dependency manifest: `python third_party/tools/generate_dependency_manifest.py` →
  `third_party/DEPENDENCY_MANIFEST.json`; npm 3 direct + 6 dev dependencies and 14 Python runtime
  packages, all with resolved licenses.
- Re-run the audit after any `web/` change; exit code 0 is the acceptance condition.
