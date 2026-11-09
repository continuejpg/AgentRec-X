// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Modified for AgentRec-X Web Demo (2026-11): unchanged suggestion-chip pattern, used here for
// clarification quick options only — and only when the backend actually provides them.

"use client";

export default function Suggestions({
  suggestions,
  onPick,
  disabled,
  label = "Quick options",
}: {
  suggestions: string[];
  onPick: (text: string) => void;
  disabled?: boolean;
  label?: string;
}) {
  if (!suggestions.length) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-1.5" role="group" aria-label={label}>
      {suggestions.map((suggestion, index) => (
        <button
          key={suggestion}
          type="button"
          disabled={disabled}
          onClick={() => onPick(suggestion)}
          className="chip"
          style={{ animationDelay: `${index * 70}ms` }}
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}
