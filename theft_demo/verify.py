"""Qwen check of one suspicious episode: the 5 person crops -> CONFIRMED / UNCERTAIN / NORMAL + reason.

Copied from fire/verify.py and adapted: the prompt is the theft "paza" prompt (prompts/theft_verify.txt, from
src/vlm.py), there is no separate [Crop] image (the frames already are crops), and AsyncVerifier caps the
number of calls running at once (several people can be checked at the same time).

Never raises: a timeout gives verdict TIMEOUT, any other failure ERROR.
"""
from __future__ import annotations

import base64
import os
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from openai import APITimeoutError, OpenAI

DEMO = Path(__file__).resolve().parent


@dataclass
class Verdict:
    verdict: str            # CONFIRMED / UNCERTAIN / NORMAL / TIMEOUT / ERROR / UNVERIFIED
    confidence: int
    reason: str
    raw: str
    latency_s: float
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    queue_s: float = 0.0    # AsyncVerifier: waited for a free slot before the call


def parse_response(text: str) -> tuple[str, int, str]:
    vm = re.search(r"VERDICT\W{0,5}(CONFIRMED|UNCERTAIN|NORMAL)", text, re.IGNORECASE)
    if vm:
        cm = re.search(r"CONFIDENCE\W{0,5}(\d{1,3})", text, re.IGNORECASE)
        rm = re.search(r"(?:REASON|DESCRIPTION)\W{0,5}(.+)", text, re.IGNORECASE | re.DOTALL)
        return (vm.group(1).upper(), max(0, min(100, int(cm.group(1)))) if cm else 0,
                " ".join(rm.group(1).split()) if rm else "")
    vm = re.search(r"\b(CONFIRMED|UNCERTAIN|NORMAL)\b", text.upper())
    if vm:
        return vm.group(1), 0, "(parsed from free text)"
    return "ERROR", 0, "unparseable response"


class Verifier:
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 timeout_s: float = 180, max_tokens: int = 16000, temperature: float = 0.0,
                 prompt_path: Path = DEMO / "prompts" / "theft_verify.txt"):
        base_url = base_url or os.getenv("VLM_API_URL", "https://openrouter.ai/api/v1")
        api_key = api_key or os.getenv("VLM_API_KEY", "")
        if not api_key or not model:
            raise ValueError("VLM_API_KEY (theft_demo/.env) and verify.model (theft_demo/config.yaml) must be set")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def classify(self, frames: list[bytes]) -> Verdict:
        k = len(frames)
        content: list[dict] = [{"type": "text", "text": self.prompt.format(k=k)}]
        for i, jpg in enumerate(frames, 1):
            content += [{"type": "text", "text": f"[Frame {i}/{k}]"}, _image(jpg)]
        messages = [{"role": "user", "content": content}]

        t0 = time.monotonic()
        last_err = ""
        while True:
            left = self.timeout_s - (time.monotonic() - t0)
            if left <= 1:
                return Verdict("TIMEOUT", 0, f"no answer within {self.timeout_s:.0f} s", last_err,
                               time.monotonic() - t0, self.model)
            try:
                resp = self.client.with_options(timeout=left).chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
                    extra_body={"usage": {"include": True}},   # OpenRouter: return cost
                )
            except APITimeoutError:
                continue                                        # loop ends as TIMEOUT
            except Exception as e:                              # network / rate limit / server error: retry while time is left
                last_err = f"{type(e).__name__}: {e}"[:300]
                if self.timeout_s - (time.monotonic() - t0) < 10:
                    return Verdict("ERROR", 0, last_err, "", time.monotonic() - t0, self.model)
                time.sleep(3)
                continue
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            verdict, conf, reason = parse_response(text)
            u = resp.usage
            details = getattr(u, "completion_tokens_details", None)
            return Verdict(verdict, conf, reason, text, time.monotonic() - t0, self.model,
                           getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0,
                           getattr(details, "reasoning_tokens", 0) or 0, float(getattr(u, "cost", 0) or 0))


def _image(jpg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}}


class AsyncVerifier:
    """Each call in its own daemon thread (detection never waits, Ctrl+C never waits for a call), at most
    max_parallel at once; the rest wait their turn. The future's result has queue_s set: time spent waiting."""

    def __init__(self, verifier: Verifier, max_parallel: int = 4):
        self.verifier = verifier
        self.slots = threading.Semaphore(max(1, max_parallel))

    def submit(self, frames: list[bytes]) -> Future:
        f: Future = Future()
        t0 = time.monotonic()

        def work():
            with self.slots:
                waited = time.monotonic() - t0
                try:
                    v = self.verifier.classify(frames)
                    v.queue_s = waited
                    f.set_result(v)
                except BaseException as e:       # classify never raises; belt and braces
                    f.set_exception(e)
        threading.Thread(target=work, daemon=True, name="qwen").start()
        return f

    def close(self) -> None:
        pass


class NoVerifier:
    """No Qwen (--no-qwen / THEFT_QWEN=off): every check is UNVERIFIED and treated as a possible theft."""

    def submit(self, frames: list[bytes]) -> Future:
        f: Future = Future()
        f.set_result(Verdict("UNVERIFIED", 0, "AI check off", "", 0.0))
        return f

    def close(self) -> None:
        pass
