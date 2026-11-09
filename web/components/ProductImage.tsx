// AgentRec-X Web Demo (2026-11) - new frontend module for the local shopping-assistant demo.
// No upstream code is copied in this file; adapted Apache-2.0 components are listed in
// THIRD_PARTY_NOTICES.md and keep their Copyright 2026 Anthropic PBC + SPDX headers.

"use client";

import { useId } from "react";

/** Small deterministic hash so each placeholder differs subtly without implying a product photo. */
function hashSeed(seed: string): number {
  let value = 0;
  for (let index = 0; index < seed.length; index += 1) {
    value = (value * 31 + seed.charCodeAt(index)) % 997;
  }
  return value;
}

/**
 * A neutral, self-drawn catalogue placeholder: layered soft hills, a sun disc and a photo frame,
 * tinted by a deterministic seed. It never depicts a real product and is labelled as absent.
 */
export function ProductPlaceholder({
  seed,
  className = "",
  label = "No catalogue image",
  compact = false,
}: {
  seed: string;
  className?: string;
  label?: string | null;
  compact?: boolean;
}) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const seedValue = hashSeed(seed);
  const hue = 205 + (seedValue % 30); // cool navy range only: neutral, never a brand colour
  const gradientId = `ph-grad-${uid}`;
  const patternId = `ph-dots-${uid}`;
  const tilt = ((seedValue % 5) - 2) * 0.9;
  return (
    <div className={`relative flex h-full w-full items-center justify-center overflow-hidden bg-(--well) ${className}`}>
      <svg
        viewBox="0 0 160 110"
        preserveAspectRatio="xMidYMid slice"
        aria-hidden
        className="h-full w-full"
        style={{ transform: `rotate(${tilt}deg) scale(1.04)` }}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={`hsl(${hue}, 26%, 93%)`} />
            <stop offset="100%" stopColor={`hsl(${hue}, 22%, 84%)`} />
          </linearGradient>
          <pattern id={patternId} width="9" height="9" patternUnits="userSpaceOnUse">
            <circle cx="1.2" cy="1.2" r="0.8" fill="rgba(30,44,79,0.14)" />
          </pattern>
        </defs>
        <rect width="160" height="110" fill={`url(#${gradientId})`} />
        <rect width="160" height="110" fill={`url(#${patternId})`} opacity=".55" />
        <circle cx="118" cy="32" r="12" fill="rgba(30,44,79,0.10)" />
        <path
          d="M-6 84c16-14 30-16 44-8s28 13 44 5 28-18 46-13 26 8 38 6v40H-6z"
          fill={`hsl(${hue}, 24%, 78%)`}
          opacity=".85"
        />
        <path
          d="M-6 94c18-10 32-11 48-4s26 9 42 3 30-12 44-9 24 9 38 7v22H-6z"
          fill={`hsl(${hue}, 28%, 70%)`}
          opacity=".7"
        />
        <g fill="none" stroke="rgba(30,44,79,0.34)" strokeWidth="1.4" strokeLinejoin="round">
          <rect x="63" y="44" width="34" height="26" rx="3.5" />
          <circle cx="73" cy="53" r="3" />
          <path d="m65 67 8-7 6 5 5-4 6 6" />
        </g>
      </svg>
      {!compact && label ? (
        <span className="pointer-events-none absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/20 to-transparent px-2 pb-0.5 pt-3 text-[9.5px] font-medium uppercase tracking-[0.08em] text-(--ink-2)">
          {label}
        </span>
      ) : null}
    </div>
  );
}

/**
 * Renders the real image only when the payload carries one (the contract says it does not);
 * otherwise the neutral placeholder.
 */
export function ProductImage({
  productId,
  imageUrl,
  alt,
  className = "",
  compact = false,
}: {
  productId: string;
  imageUrl: string | null;
  alt: string;
  className?: string;
  compact?: boolean;
}) {
  if (imageUrl) {
    return (
      // eslint-disable-next-line @next/next/no-img-element -- catalogue URLs are not guaranteed
      // to be allow-listed at build time; this demo never reaches a real image today.
      <img src={imageUrl} alt={alt} loading="lazy" className={`h-full w-full object-cover ${className}`} />
    );
  }
  return <ProductPlaceholder seed={productId} className={className} compact={compact} />;
}
