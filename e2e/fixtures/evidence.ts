/**
 * Evidence recorder + artifact helpers for the AgentRec-X Web Demo E2E suite.
 *
 * The acceptance run must prove *what was really asserted*, not just that a screenshot exists.
 * Every test appends structured records (checks, timings, payload cross-checks, screenshot
 * dimensions) to `runs/web_demo/e2e/evidence.jsonl`; `generate-report.mjs` merges those records
 * with the Playwright summary into `summary.json` + `REPORT.md`.
 *
 * No mocks: records only ever describe real DOM/network observations made against the local demo.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import type { Page, TestInfo } from "@playwright/test";

export const E2E_DIR = path.resolve(__dirname, "..");
export const REPO_ROOT = path.resolve(E2E_DIR, "..");
export const RUNS_DIR = path.join(REPO_ROOT, "runs", "web_demo");
export const SCREENSHOTS_DIR = path.join(RUNS_DIR, "screenshots");
export const TRACES_DIR = path.join(RUNS_DIR, "traces");
export const E2E_OUT_DIR = path.join(RUNS_DIR, "e2e");
export const EVIDENCE_FILE = path.join(E2E_OUT_DIR, "evidence.jsonl");

/** Real local BFF (FastAPI in WSL). Overridable for a non-default local port. */
export const BFF_BASE = (process.env.E2E_BFF_URL ?? "http://127.0.0.1:8787").replace(/\/+$/, "");
/** Real Next.js frontend (Playwright `baseURL` is the same URL). */
export const FRONTEND_BASE = (process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000").replace(/\/+$/, "");

export type EvidenceKind = "check" | "timing" | "payload" | "screenshot" | "note";

export interface EvidenceRecord {
  ts: string;
  test: string;
  kind: EvidenceKind;
  name: string;
  pass?: boolean | null;
  detail?: string | null;
  duration_ms?: number | null;
  path?: string | null;
  width?: number | null;
  height?: number | null;
}

export function ensureArtifactDirs(): void {
  for (const dir of [SCREENSHOTS_DIR, TRACES_DIR, E2E_OUT_DIR]) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/** Stable label used to join `evidence.jsonl` records with the Playwright summary. */
export function testLabel(testInfo: TestInfo): string {
  return testInfo.titlePath.filter(Boolean).join(" > ");
}

function appendEvidence(record: EvidenceRecord): void {
  ensureArtifactDirs();
  fs.appendFileSync(EVIDENCE_FILE, `${JSON.stringify(record)}\n`, "utf8");
}

function base(testInfo: TestInfo, kind: EvidenceKind, name: string, rest: Partial<EvidenceRecord>): void {
  appendEvidence({
    ts: new Date().toISOString(),
    test: testLabel(testInfo),
    kind,
    name,
    ...rest,
  });
}

/** Record a real assertion outcome (DOM or network) with a human-readable observation. */
export function recordCheck(
  testInfo: TestInfo,
  name: string,
  detail: string,
  pass: boolean,
): void {
  base(testInfo, "check", name, { pass, detail });
}

/** Record a measured duration (e.g. chat turn latency, compare round-trip). */
export function recordTiming(
  testInfo: TestInfo,
  name: string,
  durationMs: number,
  detail?: string,
): void {
  base(testInfo, "timing", name, {
    pass: null,
    duration_ms: Math.round(durationMs),
    detail: detail ?? null,
  });
}

/** Record a cross-checked real payload (BFF JSON) so the report shows network evidence. */
export function recordPayload(
  testInfo: TestInfo,
  name: string,
  payload: unknown,
  detail?: string,
): void {
  let serialised: string;
  try {
    serialised = JSON.stringify(payload);
  } catch {
    serialised = String(payload);
  }
  const truncated = serialised.length > 4000 ? `${serialised.slice(0, 4000)}…[truncated]` : serialised;
  base(testInfo, "payload", name, { pass: null, detail: detail ?? null, path: null, width: null, height: null });
  // The payload body goes to a dedicated file next to evidence.jsonl to keep JSONL lines small.
  ensureArtifactDirs();
  const payloadDir = path.join(E2E_OUT_DIR, "payloads");
  fs.mkdirSync(payloadDir, { recursive: true });
  const safe = `${Date.now()}-${name.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80)}.json`;
  fs.writeFileSync(path.join(payloadDir, safe), truncated, "utf8");
  appendEvidence({
    ts: new Date().toISOString(),
    test: testLabel(testInfo),
    kind: "note",
    name: `${name}.payload_file`,
    path: path.join("payloads", safe),
  });
}

/** Read PNG width/height from the IHDR header without any image library. */
export function pngSize(file: string): { width: number; height: number } | null {
  try {
    const buf = fs.readFileSync(file);
    if (buf.length < 24) return null;
    if (buf.toString("ascii", 1, 4) !== "PNG") return null;
    return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
  } catch {
    return null;
  }
}

/**
 * Save a final acceptance screenshot explicitly (not only-on-failure) and record its real path
 * and pixel dimensions in the evidence stream.
 */
export async function saveScreenshot(
  page: Page,
  testInfo: TestInfo,
  fileName: string,
  expected?: { width: number; height: number },
): Promise<{ file: string; width: number; height: number }> {
  ensureArtifactDirs();
  const file = path.join(SCREENSHOTS_DIR, fileName);
  await page.screenshot({ path: file, animations: "disabled" });
  const size = pngSize(file) ?? { width: 0, height: 0 };
  const matches = expected ? size.width === expected.width && size.height === expected.height : true;
  base(testInfo, "screenshot", fileName, {
    pass: matches,
    path: file,
    width: size.width,
    height: size.height,
    detail: expected ? `expected ${expected.width}x${expected.height}; actual ${size.width}x${size.height}` : null,
  });
  return { file, ...size };
}

/** Force the page to a deterministic size for 1440x900 desktop or 390x844 mobile captures. */
export async function setViewport(
  page: Page,
  width: number,
  height: number,
): Promise<void> {
  await page.setViewportSize({ width, height });
  await page.waitForTimeout(150);
}
