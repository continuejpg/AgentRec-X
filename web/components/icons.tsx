// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): icon set trimmed to the glyphs this demo uses and
// extended with panel / menu / refresh / stop / compare / info / chevron-down / play, all drawn
// as 24px line icons so no icon package is needed.

import type { SVGProps } from "react";

const STROKE = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

const PATHS = {
  tag: (
    <>
      <path d="M3.5 12.2V4.5a1 1 0 0 1 1-1h7.7l8.3 8.3-8.7 8.7z" />
      <circle cx="8" cy="8" r="1.4" />
    </>
  ),
  box: (
    <>
      <path d="M3.5 7.5 12 3.5l8.5 4v9L12 20.5l-8.5-4z" />
      <path d="M3.5 7.5 12 11.5l8.5-4M12 11.5v9" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6" />
      <path d="m16 16 4 4" />
    </>
  ),
  message: <path d="M4.5 5.5h15v10h-8l-4.5 3.5v-3.5h-2.5z" />,
  clock: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 7.5V12l3 2" />
    </>
  ),
  alert: (
    <>
      <path d="M12 4.5 20.5 19h-17z" />
      <path d="M12 10v4M12 16.5v.5" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 11v5.5M12 7.6v.4" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8.5" r="3.5" />
      <path d="M5 19.5c1-3.5 3.8-5 7-5s6 1.5 7 5" />
    </>
  ),
  check: <path d="m5 12.5 4.5 4.5L19 7.5" strokeWidth={2} />,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  "panel-right": (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
      <path d="M14.5 4.5v15" />
    </>
  ),
  "chevron-right": <path d="m9 6 6 6-6 6" />,
  "chevron-down": <path d="m6 9 6 6 6-6" />,
  "arrow-up": <path d="M12 19V5M6 11l6-6 6 6" strokeWidth={2} />,
  "arrow-right": <path d="M5 12h14M13 6l6 6-6 6" />,
  refresh: (
    <>
      <path d="M19.5 12a7.5 7.5 0 0 1-12.8 5.3L4 14.5" />
      <path d="M4 19v-4.5h4.5M4.5 12a7.5 7.5 0 0 1 12.8-5.3L20 9.5" />
      <path d="M20 5v4.5h-4.5" />
    </>
  ),
  stop: <rect x="6.5" y="6.5" width="11" height="11" rx="1.5" />,
  compare: (
    <>
      <rect x="3.5" y="4.5" width="7" height="15" rx="1.5" />
      <rect x="13.5" y="4.5" width="7" height="15" rx="1.5" />
    </>
  ),
  play: <path d="M8 5.5 18.5 12 8 18.5z" />,
  photo: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="m5 17 4.5-4 3.5 3 2.5-2 4 3.5" />
    </>
  ),
} as const;

export type IconName = keyof typeof PATHS | "spark";

/** `spark` is the assistant's mark and the only filled glyph. */
export function Icon({
  name,
  size = 18,
  className = "",
  ...rest
}: { name: IconName; size?: number } & Omit<SVGProps<SVGSVGElement>, "name">) {
  if (name === "spark") {
    return (
      <svg
        viewBox="0 0 24 24"
        width={size}
        height={size}
        fill="currentColor"
        aria-hidden
        className={`shrink-0 ${className}`}
        {...rest}
      >
        <path d="M12 2.5c.4 4.6 2.4 7.4 7.5 8-5.1.6-7.1 3.4-7.5 8-.4-4.6-2.4-7.4-7.5-8 5.1-.6 7.1-3.4 7.5-8z" />
        <path d="M19 15c.2 2 .9 3 3 3.2-2.1.3-2.8 1.3-3 3.3-.2-2-.9-3-3-3.3 2.1-.2 2.8-1.2 3-3.2z" opacity=".7" />
      </svg>
    );
  }
  const path = PATHS[name];
  if (path === undefined) return null;
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden className={`shrink-0 ${className}`} {...STROKE} {...rest}>
      {path}
    </svg>
  );
}
