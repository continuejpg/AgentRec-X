/**
 * Custom Playwright reporter: writes a machine-readable summary of the real acceptance run to
 * `runs/web_demo/e2e/summary.json`. `generate-report.mjs` later merges this with evidence.jsonl,
 * launch.json, cleanup.json and the saved screenshots to produce the human REPORT.md.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import type {
  FullConfig,
  FullResult,
  Reporter,
  Suite,
  TestCase,
  TestResult,
} from "@playwright/test/reporter";

function relative(file: string, root: string): string {
  const relativePath = path.relative(root, file);
  return relativePath.split(path.sep).join("/");
}

function firstLines(value: string | undefined, count: number): string {
  return String(value ?? "")
    .split(/\r?\n/)
    .slice(0, count)
    .join("\n");
}

class ArtifactReporter implements Reporter {
  private config: FullConfig | null = null;
  private startedAt = new Date();
  private records: any[] = [];
  private expectedTests = 0;

  onBegin(config: FullConfig, suite: Suite): void {
    this.config = config;
    this.startedAt = new Date();
    this.expectedTests = suite.allTests().length;
  }

  onTestEnd(test: TestCase, result: TestResult): void {
    const root = this.config?.rootDir ?? process.cwd();
    const attachments = (result.attachments ?? []).map((attachment) => {
      let bytes: number | null = null;
      if (attachment.path && fs.existsSync(attachment.path)) {
        try {
          bytes = fs.statSync(attachment.path).size;
        } catch {
          bytes = null;
        }
      }
      return {
        name: attachment.name,
        path: attachment.path ? relative(attachment.path, path.resolve(root, "..")) : null,
        content_type: attachment.contentType ?? null,
        bytes,
      };
    });
    const errors = (result.errors ?? []).map((error) => ({
      message: firstLines(error.message, 10),
      stack: firstLines(error.stack, 16),
    }));
    this.records.push({
      title: test.title,
      title_path: test.titlePath().filter(Boolean),
      file: relative(test.location.file, root),
      line: test.location.line,
      status: result.status,
      expected_status: test.expectedStatus,
      duration_ms: Math.round(result.duration),
      retry: result.retry,
      worker: result.workerIndex,
      start_time: new Date(result.startTime).toISOString(),
      errors,
      attachments,
    });
  }

  async onEnd(result: FullResult): Promise<void> {
    const totals = {
      tests: this.records.length,
      passed: this.records.filter((record) => record.status === "passed").length,
      failed: this.records.filter(
        (record) => record.status === "failed" || record.status === "timedOut" || record.status === "interrupted",
      ).length,
      skipped: this.records.filter((record) => record.status === "skipped").length,
      expected_tests: this.expectedTests,
      duration_ms: Math.round(result.duration),
    };
    const finish = new Date();
    const summary = {
      schema: "agentrecx.web_demo.e2e.summary.v1",
      generated_at: finish.toISOString(),
      run: {
        started_at: this.startedAt.toISOString(),
        finished_at: finish.toISOString(),
        duration_ms: Math.round(finish.getTime() - this.startedAt.getTime()),
        status: result.status,
        expected_tests: this.expectedTests,
      },
      environment: {
        base_url: process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000",
        bff_url: process.env.E2E_BFF_URL ?? "http://127.0.0.1:8787",
        browser: "system Chrome via Playwright channel=chrome (headless)",
        desktop_viewport: "1440x900",
        mobile_viewport: "390x844",
        trace: "retain-on-failure",
        screenshot: "only-on-failure (final acceptance screenshots saved explicitly)",
        mocks: "none — real local BFF + real P17 MCP + BGE + M9",
      },
      totals,
      tests: this.records,
    };

    const defaultPath = path.resolve(
      __dirname,
      "..",
      "..",
      "runs",
      "web_demo",
      "e2e",
      "summary.json",
    );
    const outputPath = process.env.E2E_SUMMARY_PATH ?? defaultPath;
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  }
}

export default ArtifactReporter;
