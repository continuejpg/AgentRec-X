/**
 * Global setup: fail fast and loudly if the real demo stack is not actually ready.
 *
 * The Playwright suite is only allowed to run against a live local demo whose /api/health reports
 * `ready=true` (both MCP servers loaded catalogue + BGE) and whose Next.js frontend answers. This
 * prevents a "green" run against a dead/mocked/half-started backend.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { request } from "@playwright/test";

const E2E_DIR = path.resolve(__dirname);
const E2E_OUT_DIR = path.resolve(E2E_DIR, "..", "runs", "web_demo", "e2e");

export default async function globalSetup(): Promise<void> {
  const bff = (process.env.E2E_BFF_URL ?? "http://127.0.0.1:8787").replace(/\/+$/, "");
  const frontend = (process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000").replace(/\/+$/, "");
  fs.mkdirSync(E2E_OUT_DIR, { recursive: true });

  const context = await request.newContext();
  const preflight: Record<string, unknown> = {
    started_at: new Date().toISOString(),
    bff,
    frontend,
  };
  try {
    let health: any = null;
    let lastError = "";
    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      try {
        const response = await context.get(`${bff}/api/health`, { timeout: 10_000 });
        health = await response.json();
        if (health?.ready === true) break;
      } catch (error) {
        lastError = String(error);
      }
      await new Promise((resolve) => setTimeout(resolve, 2_000));
    }
    if (!health || health.ready !== true) {
      throw new Error(
        `Real BFF is not ready at ${bff}/api/health (ready=${health?.ready}). ` +
          `Start the demo with scripts/run_web_demo_e2e.ps1. Last error: ${lastError}`,
      );
    }
    preflight.health = health;
    if (health.catalogue_size !== 76263) {
      throw new Error(
        `Unexpected real catalogue_size=${health.catalogue_size}; the verified demo catalogue is 76263.`,
      );
    }
    const servers = Array.isArray(health.servers) ? health.servers : [];
    const notReady = servers.filter((server: any) => server.status !== "ready");
    const expectedProfiles = Number(process.env.E2E_EXPECTED_PROFILES ?? "2");
    if (servers.length < expectedProfiles || notReady.length > 0) {
      throw new Error(
        `Expected at least ${expectedProfiles} ready MCP profiles, got ${JSON.stringify(servers.map((s: any) => ({ id: s.profile_id, status: s.status })))}`,
      );
    }

    const frontendResponse = await context.get(frontend, { timeout: 60_000 });
    if (frontendResponse.status() !== 200) {
      throw new Error(`Real frontend at ${frontend} returned HTTP ${frontendResponse.status()}`);
    }
    preflight.frontend_status = frontendResponse.status();
    preflight.completed_at = new Date().toISOString();
  } finally {
    await context.dispose();
    fs.writeFileSync(path.join(E2E_OUT_DIR, "preflight.json"), JSON.stringify(preflight, null, 2), "utf8");
  }
}
