// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): kept Pill / Button / IconButton / Sheet / Skeleton
// / SectionTitle / Notice, added a side-aware Sheet (drawers + detail panels) and a StatusDot for
// real BFF readiness; commerce-specific primitives (sparkline, stat tiles, carts) removed.

"use client";

import { type ButtonHTMLAttributes, type ReactNode, useEffect, useId, useState } from "react";
import { createPortal } from "react-dom";
import { Icon, type IconName } from "./icons";

export type Tone = "ok" | "warn" | "danger" | "info" | "violet" | "accent" | "muted";

export const TONE_SOFT: Record<Tone, string> = {
  ok: "bg-(--ok-soft) text-(--ok)",
  warn: "bg-(--warn-soft) text-(--warn)",
  danger: "bg-(--danger-soft) text-(--danger)",
  info: "bg-(--info-soft) text-(--info)",
  violet: "bg-(--violet-soft) text-(--violet)",
  accent: "bg-(--accent-soft) text-(--accent-ink)",
  muted: "bg-(--well) text-(--ink-soft)",
};

/** A status or label chip; `dot` marks a state rather than a category. */
export function Pill({
  tone = "muted",
  dot = false,
  children,
  title,
  className = "",
}: {
  tone?: Tone;
  dot?: boolean;
  children: ReactNode;
  title?: string;
  className?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-0.5 text-[11.5px] font-semibold leading-[1.35] ${TONE_SOFT[tone]} ${className}`}
    >
      {dot ? <i aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" /> : null}
      {children}
    </span>
  );
}

/** A readiness / stream state dot with an accessible label. */
export function StatusDot({ tone, label }: { tone: Tone; label: string }) {
  const fill: Record<Tone, string> = {
    ok: "bg-(--ok)",
    warn: "bg-(--warn)",
    danger: "bg-(--danger)",
    info: "bg-(--info)",
    violet: "bg-(--violet)",
    accent: "bg-(--accent)",
    muted: "bg-(--ink-faint)",
  };
  return (
    <span role="img" aria-label={label} title={label} className={`inline-block h-2 w-2 shrink-0 rounded-full ${fill[tone]}`} />
  );
}

export function IconButton({
  icon,
  label,
  onClick,
  className = "",
  size = 30,
  disabled = false,
}: {
  icon: IconName;
  label: string;
  onClick: () => void;
  className?: string;
  size?: number;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      disabled={disabled}
      className={`grid shrink-0 place-items-center rounded-lg text-(--ink-soft) transition-colors hover:bg-(--ground) hover:text-(--ink) disabled:pointer-events-none disabled:opacity-40 ${className}`}
      style={{ width: size, height: size }}
    >
      <Icon name={icon} size={Math.round(size * 0.57)} />
    </button>
  );
}

/** `primary` is the ink button; `accent` is the terracotta action. */
export function Button({
  variant = "secondary",
  size = "md",
  icon,
  children,
  className = "",
  ...rest
}: {
  variant?: "primary" | "secondary" | "accent" | "ghost";
  size?: "sm" | "md";
  icon?: IconName;
  children: ReactNode;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  const variants = {
    primary: "bg-(--ink) text-(--surface) shadow-(--shadow-sm) hover:brightness-110",
    secondary: "border border-(--line-strong) bg-(--card) text-(--ink) hover:bg-(--ground)",
    accent: "bg-(--accent-strong) text-(--on-accent) shadow-(--shadow-sm) hover:brightness-95",
    ghost: "text-(--ink-soft) hover:bg-(--ground) hover:text-(--ink)",
  };
  const sizes = {
    sm: "px-2.5 py-[5px] text-[12.5px] rounded-[8px]",
    md: "px-3.5 py-2 text-[13.5px] rounded-[10px]",
  };
  return (
    <button
      type="button"
      className={`inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap font-semibold transition disabled:cursor-not-allowed disabled:opacity-45 ${variants[variant]} ${sizes[size]} ${className}`}
      {...rest}
    >
      {icon ? <Icon name={icon} size={size === "sm" ? 14 : 16} /> : null}
      {children}
    </button>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`ac-skeleton rounded-xl ${className}`} />;
}

/** A section heading inside a panel or sheet, with an optional note on the right. */
export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <h3 className="mb-2 flex items-baseline gap-2 text-[13px] font-semibold text-(--ink)">
      {children}
      {aside ? <span className="ml-auto text-[11.5px] font-normal text-(--ink-soft)">{aside}</span> : null}
    </h3>
  );
}

/** The empty or unreachable state inside a view. */
export function Notice({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-(--line) bg-(--card) p-4 text-[13px] leading-relaxed text-(--ink-soft) ${className}`}>
      {children}
    </div>
  );
}

/** A compact empty state with an icon; `children` can carry a real action. */
export function EmptyState({
  icon = "box",
  title,
  hint,
  children,
  compact = false,
}: {
  icon?: IconName;
  title: string;
  hint?: ReactNode;
  children?: ReactNode;
  compact?: boolean;
}) {
  return (
    <div className={`flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-(--line-strong) bg-(--card)/60 text-center ${compact ? "px-4 py-6" : "px-6 py-10"}`}>
      <span className="grid h-9 w-9 place-items-center rounded-full bg-(--well) text-(--ink-faint)">
        <Icon name={icon} size={18} />
      </span>
      <div className="text-[13.5px] font-semibold text-(--ink)">{title}</div>
      {hint ? <p className="max-w-sm text-[12.5px] leading-relaxed text-(--ink-soft)">{hint}</p> : null}
      {children ? <div className="mt-1 flex flex-wrap items-center justify-center gap-2">{children}</div> : null}
    </div>
  );
}

/** A right- or left-hand sheet over a scrim, portalled to the body; Escape and the scrim close it. */
export function Sheet({
  title,
  detail,
  onClose,
  footer,
  children,
  side = "right",
  closeLabel = "Close",
  labelledBy,
}: {
  title: ReactNode;
  detail?: ReactNode;
  onClose: () => void;
  footer?: ReactNode;
  children: ReactNode;
  side?: "left" | "right";
  closeLabel?: string;
  labelledBy?: string;
}) {
  const [host, setHost] = useState<HTMLElement | null>(null);
  const fallbackId = useId();
  const titleId = labelledBy ?? fallbackId;
  useEffect(() => {
    setHost(document.body);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  if (!host) return null;
  return createPortal(
    <>
      <div onClick={onClose} aria-hidden className="fixed inset-0 z-40 bg-black/30" />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`fixed inset-y-0 z-50 flex w-[min(96vw,468px)] flex-col overflow-hidden bg-(--card) shadow-(--shadow-lg) ${
          side === "right"
            ? "ac-slide-in-right right-0 sm:inset-y-2.5 sm:right-2.5 sm:rounded-[18px]"
            : "ac-slide-in-left left-0 sm:inset-y-2.5 sm:left-2.5 sm:rounded-[18px]"
        }`}
      >
        <div className="flex items-center gap-2 border-b border-(--line) py-3 pl-[18px] pr-3">
          <div id={titleId} className="min-w-0 flex-1 truncate text-[14px] font-semibold text-(--ink)">
            {title}
            {detail ? <span className="ml-2 font-normal tabular-nums text-(--ink-soft)">{detail}</span> : null}
          </div>
          <IconButton icon="x" label={closeLabel} onClick={onClose} />
        </div>
        <div className="panel-scroll flex flex-1 flex-col gap-4 overflow-y-auto p-[18px]">{children}</div>
        {footer ? <div className="ac-safe-bottom flex items-center gap-2 border-t border-(--line) px-[18px] py-3">{footer}</div> : null}
      </aside>
    </>,
    host,
  );
}
