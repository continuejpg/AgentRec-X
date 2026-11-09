// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): AgentRec-X metadata/theme and a local font stack
// instead of a downloaded webfont, so the build works offline.

import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentRec-X · Shopping Assistant",
  description:
    "Local AgentRec-X shopping assistant demo: real catalogue search, P15/P16 evidence, Gate decisions, M9 preferences and Qwen replay in one web UI.",
  icons: {
    icon: "/favicon.svg",
  },
};

export const viewport: Viewport = {
  themeColor: "#f4f6fa",
  colorScheme: "light",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
