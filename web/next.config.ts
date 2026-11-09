// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): removed the web-shared transpile entry (the
// adapted components now live in this project); the BFF URL is read at runtime.

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Next.js 16 blocks dev-only resources (HMR/dev overlay, and therefore the hydrated app in a
  // Playwright/Chrome session) when the page is opened through a different host spelling than
  // the dev server's canonical origin. The demo launcher and E2E open http://127.0.0.1:3000
  // while Next prints http://localhost:3000; explicitly allow both loopback spellings.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
