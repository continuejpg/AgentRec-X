"""Minimal local-Qwen operation proposal server for Agent bridge experiments.

Run this on the AutoDL GPU host; the Agent benchmark/service runs locally and calls it over
an SSH tunnel.  It never writes M5 and never sees gold/task expectations.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .format import prompt_text
from .modeling import MODEL_ID, load_text_model, load_tokenizer
from .planner_prompt import planner_prompt
from .schema import parse_and_validate


class State:
    tokenizer = None
    model = None
    max_new_tokens = 256


def _row(payload: dict) -> dict:
    messages = payload.get("messages") or []
    current = str(payload.get("current") or "")
    return {
        "state": {"previous_user_messages": list(messages)[-12:]},
        "catalogue_context": {},
        "user_message": current,
    }


def _parse_planner_text(text: str) -> tuple[dict | None, list[str]]:
    """Parse the independent Planner Action schema; never the Preference schema."""
    import re

    errors: list[str] = []
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None, ["invalid planner JSON"]
    if not isinstance(payload, dict):
        return None, ["planner output is not an object"]
    if "operations" in payload:
        return None, ["planner returned Preference Operation schema"]
    action = str(payload.get("action") or "").strip().upper()
    if action not in {"ASK", "RETRIEVE", "RERANK", "FINALIZE"}:
        errors.append("invalid planner action")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        errors.append("planner arguments must be an object")
    elif action in {"ASK", "RETRIEVE"}:
        key = "question" if action == "ASK" else "query"
        if not str(arguments.get(key) or "").strip():
            errors.append(f"{action} requires arguments.{key}")
    if errors:
        return None, errors
    return {"action": action, "arguments": arguments, "rationale": str(payload.get("rationale") or "")}, []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/propose", "/plan", "/explain"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/plan":
            prompt = planner_prompt(State.tokenizer, request)
        elif self.path == "/explain":
            messages = request.get("messages") or []
            if not isinstance(messages, list) or not messages:
                self.send_error(400)
                return
            prompt = str(State.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        else:
            row = _row(request)
            prompt = prompt_text(State.tokenizer, row)
        inputs = State.tokenizer(prompt, return_tensors="pt").to(State.model.device)
        started = time.perf_counter()
        import torch

        with torch.no_grad():
            generated = State.model.generate(
                **inputs,
                max_new_tokens=State.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        elapsed = time.perf_counter() - started
        text = State.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if self.path == "/plan":
            payload, errors = _parse_planner_text(text)
        elif self.path == "/explain":
            payload, errors = None, []
        else:
            row = _row(request)
            payload, errors = parse_and_validate(text, row["user_message"])
        response = {
            "text": text,
            "payload": payload,
            "errors": errors,
            "latency_ms": elapsed * 1000.0,
            "input_tokens": int(inputs["input_ids"].numel()),
            "output_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
        }
        body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    import torch

    State.max_new_tokens = args.max_new_tokens
    State.tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    State.model = load_text_model(args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel

        State.model = PeftModel.from_pretrained(State.model, args.adapter)
    State.model.eval()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"proposal server on 127.0.0.1:{args.port} adapter={args.adapter}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
