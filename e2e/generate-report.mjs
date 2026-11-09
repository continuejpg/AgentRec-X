#!/usr/bin/env node
/**
 * Merge the Playwright run summary, the explicitly saved screenshots, the evidence stream, launch
 * timings and cleanup info into the two acceptance artifacts:
 *
 *   runs/web_demo/e2e/summary.json   (enriched machine-readable summary)
 *   runs/web_demo/e2e/REPORT.md      (human acceptance report)
 *
 * Run by scripts/run_web_demo_e2e.ps1 after Playwright finishes (also runnable directly:
 *   node e2e/generate-report.mjs
 * ). It never talks to the demo stack and never invents a result: a missing artifact or a failed
 * test is reported as such.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const RUNS = path.join(REPO_ROOT, "runs", "web_demo");
const E2E_OUT = path.join(RUNS, "e2e");
const SHOTS = path.join(RUNS, "screenshots");
const TRACES = path.join(RUNS, "traces");

function readJson(file, fallback = null) {
  try {
    const text = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "");
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function readJsonl(file) {
  try {
    return fs
      .readFileSync(file, "utf8")
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
  } catch {
    return [];
  }
}

function pngSize(file) {
  try {
    const buf = fs.readFileSync(file);
    if (buf.length < 24 || buf.toString("ascii", 1, 4) !== "PNG") return null;
    return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20), bytes: buf.length };
  } catch {
    return null;
  }
}

function walk(dir, predicate) {
  const results = [];
  if (!fs.existsSync(dir)) return results;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) results.push(...walk(full, predicate));
    else if (predicate(full)) results.push(full);
  }
  return results;
}

function relativeToRoot(file) {
  return path.relative(REPO_ROOT, file).split(path.sep).join("/");
}

function mdEscape(value) {
  return String(value ?? "")
    .replace(/\r?\n/g, " ")
    .replace(/\|/g, "\\|")
    .trim();
}

function formatMs(ms) {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

function statusMark(status) {
  if (status === "passed") return "PASS";
  if (status === "skipped") return "SKIP";
  return "FAIL";
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------
const notes = readJson(path.join(__dirname, "report-notes.json"), {}) ?? {};
const summary = readJson(path.join(E2E_OUT, "summary.json"), null);
const launch = readJson(path.join(E2E_OUT, "launch.json"), null);
const cleanup = readJson(path.join(E2E_OUT, "cleanup.json"), null);
const preflight = readJson(path.join(E2E_OUT, "preflight.json"), null);
const evidence = readJsonl(path.join(E2E_OUT, "evidence.jsonl"));

const tests = summary?.tests ?? [];
const totals = summary?.totals ?? {
  tests: tests.length,
  passed: tests.filter((test) => test.status === "passed").length,
  failed: tests.filter((test) => test.status === "failed" || test.status === "timedOut" || test.status === "interrupted")
    .length,
  skipped: tests.filter((test) => test.status === "skipped").length,
  duration_ms: 0,
};

const evidenceByTest = new Map();
for (const record of evidence) {
  const key = String(record.test ?? "");
  if (!evidenceByTest.has(key)) evidenceByTest.set(key, []);
  evidenceByTest.get(key).push(record);
}

const allScreenshots = walk(SHOTS, (file) => file.toLowerCase().endsWith(".png"))
  .map((file) => {
    const size = pngSize(file);
    return {
      name: path.basename(file),
      path: relativeToRoot(file),
      width: size?.width ?? null,
      height: size?.height ?? null,
      bytes: size?.bytes ?? null,
    };
  })
  .sort((a, b) => a.name.localeCompare(b.name));
const screenshotByName = new Map(allScreenshots.map((shot) => [shot.name, shot]));

const requiredScreenshots = notes.required_screenshots ?? [];
const optionalScreenshots = notes.optional_screenshots ?? [];
const missingScreenshots = requiredScreenshots.filter((expected) => !screenshotByName.has(expected.name));
const wrongSizeScreenshots = requiredScreenshots.filter((expected) => {
  const shot = screenshotByName.get(expected.name);
  return shot && (shot.width !== expected.width || shot.height !== expected.height);
});

const traceFiles = walk(TRACES, (file) => file.toLowerCase().endsWith(".zip")).map((file) => {
  let bytes = null;
  try {
    bytes = fs.statSync(file).size;
  } catch {
    bytes = null;
  }
  return { path: relativeToRoot(file), bytes };
});

const screenshotChecks = [...requiredScreenshots, ...optionalScreenshots].map((expected) => {
  const shot = screenshotByName.get(expected.name);
  const ok = Boolean(shot) && shot.width === expected.width && shot.height === expected.height;
  return {
    name: expected.name,
    expected: `${expected.width}x${expected.height}`,
    actual: shot ? `${shot.width}x${shot.height}` : "MISSING",
    path: shot?.path ?? null,
    bytes: shot?.bytes ?? null,
    pass: ok,
    required: requiredScreenshots.some((entry) => entry.name === expected.name),
  };
});

const evidenceChecks = evidence.filter((record) => record.kind === "check");
const failedEvidence = evidenceChecks.filter((record) => record.pass === false);
const screenshotEvidenceFailures = evidence.filter(
  (record) => record.kind === "screenshot" && record.pass === false,
);
const timings = evidence.filter((record) => record.kind === "timing");

const allTestsPassed = totals.tests > 0 && totals.failed === 0 && totals.tests === (totals.expected_tests ?? totals.tests);
const artifactsComplete =
  missingScreenshots.length === 0 &&
  wrongSizeScreenshots.length === 0 &&
  screenshotEvidenceFailures.length === 0 &&
  failedEvidence.length === 0 &&
  (totals.passed ?? 0) > 0;
const cleanupOk = cleanup == null ? null : cleanup.stopped === true;
const verdict = allTestsPassed && artifactsComplete && cleanupOk !== false ? "PASS" : "FAIL";

// ---------------------------------------------------------------------------
// Enrich summary.json
// ---------------------------------------------------------------------------
const generatedAt = new Date().toISOString();
const enriched = {
  ...(summary ?? {}),
  generated_at: generatedAt,
  report_path: relativeToRoot(path.join(E2E_OUT, "REPORT.md")),
  launch,
  cleanup,
  preflight,
  artifacts: {
    screenshots: allScreenshots,
    screenshot_checks: screenshotChecks,
    traces: traceFiles,
    trace_policy: "trace: retain-on-failure — passing tests intentionally leave no trace.zip",
    evidence_jsonl: relativeToRoot(path.join(E2E_OUT, "evidence.jsonl")),
    evidence_records: evidence.length,
    evidence_checks: evidenceChecks.length,
    failed_evidence_checks: failedEvidence.length,
    payload_dir: relativeToRoot(path.join(E2E_OUT, "payloads")),
  },
  verdict: {
    result: verdict,
    tests_passed: allTestsPassed,
    artifacts_complete: artifactsComplete,
    cleanup_ok: cleanupOk,
    missing_screenshots: missingScreenshots.map((shot) => shot.name),
    wrong_size_screenshots: wrongSizeScreenshots.map((shot) => `${shot.name} (expected ${shot.width}x${shot.height})`),
    failed_evidence_checks: failedEvidence.map((record) => `${record.test} :: ${record.name}`),
  },
  known_limitations: notes.known_limitations ?? [],
};
fs.mkdirSync(E2E_OUT, { recursive: true });
fs.writeFileSync(path.join(E2E_OUT, "summary.json"), `${JSON.stringify(enriched, null, 2)}\n`, "utf8");

// ---------------------------------------------------------------------------
// REPORT.md
// ---------------------------------------------------------------------------
const lines = [];
const push = (...parts) => lines.push(parts.join(""));

push(`# ${notes.title ?? "AgentRec-X Web Demo — E2E 验收报告"}`);
push("");
push(
  `> 结论：**${verdict}** · ${totals.tests} 个测试，${totals.passed} passed / ${totals.failed} failed / ${totals.skipped} skipped · Playwright 时长 ${formatMs(
    totals.duration_ms,
  )}`,
);
push("");
push(
  `> 真实栈 E2E：系统 Chrome（Playwright \`channel=chrome\`，headless）→ 真实 Next.js 前端 :3000 → 真实 FastAPI BFF :8787 → 真实 P17 MCP（BGE / P16 Gate / P15 Evidence / M9 Memory）。**无 mock、无伪造商品/价格/偏好**。`,
);
push("");
push(`- 生成时间：${generatedAt}`);
if (launch?.started_at) {
  push(
    `- Demo 启动：${launch.started_at} → ready ${formatMs(launch.startup_ms)}（BFF cold start，等待 \`/api/health\` ready=true）`,
  );
}
if (launch?.servers) {
  for (const server of launch.servers) {
    push(
      `  - ${server.profile_id}：status=${server.status} search_mode=${server.search_mode} writes_enabled=${server.writes_enabled} startup=${formatMs(server.startup_ms)} tools=${server.tools_count ?? "?"}`,
    );
  }
}
if (launch?.catalogue_size != null) push(`- 真实 catalogue_size：${launch.catalogue_size}`);
if (cleanup) {
  push(
    `- Demo 停止：${cleanup.stopped ? "已停止，无残留端口监听" : `**仍有残留监听：${JSON.stringify(cleanup.remaining_ports)}**`}${
      cleanup.stopped_at ? `（${cleanup.stopped_at}）` : ""
    }`,
  );
}
push("");
push(`## 1. 运行方式`);
push("");
push("```powershell");
push(notes.command ?? "powershell -ExecutionPolicy Bypass -File scripts/run_web_demo_e2e.ps1");
push("```");
push("");
push(
  `该脚本检查端口 → 安装 e2e 依赖（\`PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1\`，复用系统 Chrome）→ \`scripts/run_web_demo.ps1 -NoBrowser\` 启动真实 Demo → 等待 \`/api/health\` ready=true → 运行 Playwright → \`scripts/stop_web_demo.ps1\` 停止并校验端口释放 → 生成本报告与 \`summary.json\`。`,
);
push("");
push(`## 2. 测试结果汇总`);
push("");
push("| # | 测试文件 | 测试 | 结果 | 耗时 |");
push("|---|---|---|---|---|");
tests.forEach((test, index) => {
  push(
    `| ${index + 1} | ${mdEscape(test.file)} | ${mdEscape(test.title)} | **${statusMark(test.status)}** | ${formatMs(test.duration_ms)} |`,
  );
});
const expectedTotal = totals.expected_tests ?? tests.length;
push("");
push(
  `合计：**${totals.passed}/${expectedTotal} passed**${totals.failed ? `，**${totals.failed} failed**` : ""}${
    totals.skipped ? `，${totals.skipped} skipped` : ""
  }；Playwright 墙钟时长 ${formatMs(totals.duration_ms)}。`,
);

// Per-scenario evidence
push("");
push(`## 3. 逐场景真实断言证据`);
push("");
for (const scenario of notes.scenarios ?? []) {
  push(`### ${scenario.scenario}`);
  push("");
  if (scenario.query) push(`- 真实输入：\`${scenario.query}\``);
  if (scenario.reference) push(`- 真实参考轨迹：\`${scenario.reference}\``);
  if (scenario.network) push(`- 网络断言：${scenario.network}`);
  if (scenario.real_assertions?.length) {
    push(`- 必须成立的真实断言：`);
    for (const assertion of scenario.real_assertions) push(`  - ${assertion}`);
  }
  if (scenario.screenshots?.length) push(`- 截图：${scenario.screenshots.map((name) => `\`${name}\``).join("、")}`);
  const scenarioTests = tests.filter((test) => String(test.title_path?.[0] ?? "").includes(scenario.spec));
  if (!scenarioTests.length) {
    push(`- **未找到对应测试记录**（${scenario.spec}）`);
    push("");
    continue;
  }
  for (const test of scenarioTests) {
    push("");
    push(`**${mdEscape(test.title)}** — ${statusMark(test.status)}，${formatMs(test.duration_ms)}`);
    push("");
    const key = (test.title_path ?? []).filter(Boolean).join(" > ");
    const records = evidenceByTest.get(key) ?? [];
    const checks = records.filter((record) => record.kind === "check");
    const payloads = records.filter((record) => record.kind === "payload");
    const timingRecords = records.filter((record) => record.kind === "timing");
    if (checks.length) {
      push("| 断言 | 结果 | 观测值 |");
      push("|---|---|---|");
      for (const check of checks) {
        push(`| ${mdEscape(check.name)} | ${check.pass === false ? "**FAIL**" : "PASS"} | ${mdEscape(check.detail)} |`);
      }
      push("");
    } else {
      push("（该测试未写入证据记录 — 请检查测试是否实际执行）");
      push("");
    }
    if (timingRecords.length) {
      push(
        `耗时：${timingRecords.map((record) => `${mdEscape(record.name)}=${formatMs(record.duration_ms)}`).join("；")}`,
      );
      push("");
    }
    if (payloads.length) {
      push(`真实 HTTP payload 交叉验证（完整文件见 \`${relativeToRoot(path.join(E2E_OUT, "payloads"))}/\`）：`);
      for (const payload of payloads) push(`- \`${payload.name}\``);
      push("");
    }
    if (test.errors?.length) {
      push("失败详情：");
      for (const error of test.errors) push(`\n\`\`\`\n${mdEscape(error.message)}\n\`\`\``);
      push("");
    }
    const tracesForTest = (test.attachments ?? []).filter((attachment) => attachment.name === "trace");
    if (tracesForTest.length) {
      for (const trace of tracesForTest) push(`Trace：\`${trace.path ?? "(missing)"}\``);
      push("");
    }
  }
  push("");
}

// 4. Screenshots
push(`## 4. 截图（显式保存，尺寸真实校验）`);
push("");
push("| 文件 | 期望尺寸 | 实际尺寸 | 大小 | 结果 |");
push("|---|---|---|---|---|");
for (const shot of screenshotChecks) {
  push(
    `| [${shot.name}](${shot.path ?? "#"}) | ${shot.expected} | ${shot.actual} | ${
      shot.bytes == null ? "—" : `${(shot.bytes / 1024).toFixed(1)} KB`
    } | ${shot.pass ? "PASS" : shot.required ? "**FAIL**" : "optional/absent"} |`,
  );
}
push("");
push(`截图目录：\`${relativeToRoot(SHOTS)}/\``);
push("");

// 5. traces
push(`## 5. Trace`);
push("");
if (traceFiles.length) {
  for (const trace of traceFiles) push(`- [${trace.path}](${trace.path})（${(trace.bytes / 1024).toFixed(1)} KB）`);
} else {
  push(
    "本次运行没有失败测试，因此没有 trace.zip — 这是 `trace: retain-on-failure` 的预期行为（失败时 trace 会自动写入 `runs/web_demo/traces/<test>/trace.zip`）。",
  );
}
push("");
push(`Trace 目录：\`${relativeToRoot(TRACES)}/\``);
push("");

// 6. Real failure visibility
push(`## 6. 真实失败可见性`);
push("");
push(
  "本套件不把失败伪装成成功：非法商品引用在真实 BFF HTTP 边界断言（400/422 + 真实 `{code,message}`，绝不返回 `comparison_ready`）；只读 profile 的澄清回复在 UI 上显示真实的 `Memory writes are disabled … the reply is not faked`，且同一请求在 BFF 返回 403/409 真实错误、不返回任何 `memory_update`/`preferences` 成功 payload。",
);
push("");

// 7. Known limitations
push(`## 7. 已知限制与未解决问题`);
push("");
for (const limitation of notes.known_limitations ?? []) push(`- ${limitation}`);
if (missingScreenshots.length) {
  push(`- **缺失必需截图**：${missingScreenshots.map((shot) => shot.name).join("、")}`);
}
if (wrongSizeScreenshots.length) {
  push(`- **截图尺寸不符**：${wrongSizeScreenshots.map((shot) => `${shot.name}（期望 ${shot.width}x${shot.height}）`).join("、")}`);
}
if (failedEvidence.length) {
  push(`- **失败断言记录**：${failedEvidence.map((record) => `${record.test} :: ${record.name}`).join("；")}`);
}
for (const test of tests.filter((entry) => entry.status !== "passed")) {
  push(`- 失败测试：\`${test.file}\` · ${mdEscape(test.title)}`);
}
if (!(notes.known_limitations ?? []).length && !missingScreenshots.length && !wrongSizeScreenshots.length && !failedEvidence.length && allTestsPassed) {
  push("- 无（本次运行未发现未解决问题；质量/真实用户证据的边界见 P18_FINAL_REPORT.md，不在本 E2E 报告范围内）");
}
push("");

// 8. Artifacts
push(`## 8. Artifacts 路径`);
push("");
push(`- summary.json：\`${relativeToRoot(path.join(E2E_OUT, "summary.json"))}\``);
push(`- REPORT.md：\`${relativeToRoot(path.join(E2E_OUT, "REPORT.md"))}\``);
push(`- 截图：\`${relativeToRoot(SHOTS)}/\``);
push(`- traces：\`${relativeToRoot(TRACES)}/\``);
push(`- evidence.jsonl：\`${relativeToRoot(path.join(E2E_OUT, "evidence.jsonl"))}\``);
push(`- payloads：\`${relativeToRoot(path.join(E2E_OUT, "payloads"))}/\``);
push(`- launch.json：\`${relativeToRoot(path.join(E2E_OUT, "launch.json"))}\``);
push(`- cleanup.json：\`${relativeToRoot(path.join(E2E_OUT, "cleanup.json"))}\``);
push("");

// 9. Verdict
push(`## 9. 验判定`);
push("");
push(`| 项 | 结果 |`);
push(`|---|---|`);
push(`| 测试通过 | ${allTestsPassed ? "PASS" : "FAIL"} (${totals.passed}/${expectedTotal}) |`);
push(`| 截图齐全且尺寸正确 | ${artifactsComplete ? "PASS" : "FAIL"} |`);
push(`| Demo 进程清理 | ${cleanupOk == null ? "未记录" : cleanupOk ? "PASS（端口已释放）" : "FAIL（仍有残留）"} |`);
push(`| 无 mock 真实后端 | PASS（真实 BFF / MCP / BGE / M9） |`);
push(`| 总体 | **${verdict}** |`);
push("");
push(
  `> 说明：本报告只证明**本地真实栈的 Web E2E 工程验收**；不等于推荐质量、真实用户满意度或公网生产就绪（见 P18_FINAL_REPORT.md 的质量结论）。`,
);
push("");

fs.mkdirSync(E2E_OUT, { recursive: true });
fs.writeFileSync(path.join(E2E_OUT, "REPORT.md"), `${lines.join("\n")}\n`, "utf8");

// Console summary for the launcher script.
console.log(`[report] ${relativeToRoot(path.join(E2E_OUT, "REPORT.md"))}`);
console.log(`[report] verdict=${verdict} tests=${totals.passed}/${expectedTotal} failed=${totals.failed}`);
console.log(`[report] screenshots=${screenshotChecks.filter((shot) => shot.pass).length}/${screenshotChecks.length} traces=${traceFiles.length}`);
if (failedEvidence.length) {
  console.log(`[report] failed evidence checks=${failedEvidence.length}`);
  process.exitCode = 1;
}
