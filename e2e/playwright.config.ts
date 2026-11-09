/**
 * Playwright configuration for the AgentRec-X Web Demo acceptance suite.
 *
 * Real stack only: system Chrome (channel "chrome"), the real Next.js frontend on :3000 and the
 * real FastAPI/MCP/BGE/M9 BFF on :8787. No webServer block: `scripts/run_web_demo_e2e.ps1`
 * starts/stops the demo and verifies real readiness before the suite runs.
 */
import * as path from "node:path";
import { defineConfig } from "@playwright/test";

const E2E_DIR = __dirname;
const ARTIFACT_ROOT = path.resolve(E2E_DIR, "..", "runs", "web_demo");

export default defineConfig({
  testDir: path.join(E2E_DIR, "tests"),
  outputDir: path.join(ARTIFACT_ROOT, "traces"),
  globalSetup: path.join(E2E_DIR, "global-setup.ts"),
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 300_000,
  expect: { timeout: 25_000 },
  reporter: [
    ["list"],
    [path.join(E2E_DIR, "reporters", "artifact-reporter.ts")],
  ],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000",
    channel: "chrome",
    headless: true,
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
    actionTimeout: 25_000,
    navigationTimeout: 120_000,
    ignoreHTTPSErrors: false,
  },
  metadata: {
    stack: "real local AgentRec-X demo (Next.js + FastAPI BFF + P17 MCP + BGE + M9); no mocks",
    desktopViewport: "1440x900",
    mobileViewport: "390x844",
  },
});
