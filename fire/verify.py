"""Qwen verification of a fire/smoke alarm. Client and parser copied from src/vlm.py and adapted (plan rule 2).

Never raises: a timeout gives verdict TIMEOUT, any other failure ERROR. Both are treated as UNCERTAIN
by the event logic ("Possible fire, not verified").

Live use: AsyncVerifier (calls run in worker threads). Offline tests: SyncVerifier (the call blocks and its
measured latency is replayed in video time by the pipeline).
"""
from __future__ import annotations

import base64
import os
import re
import time
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from openai import APITimeoutError, OpenAI

FIRE = Path(__file__).resolve().parent
VERDICTS = ("CONFIRMED", "UNCERTAIN", "NORMAL")


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


def parse_response(text: str) -> tuple[str, int, str]:
    vm = re.search(r"VERDICT\W{0,5}(CONFIRMED|UNCERTAIN|NORMAL)", text, re.IGNORECASE)
    if vm:
        cm = re.search(r"CONFIDENCE\W{0,5}(\d{1,3})", text, re.IGNORECASE)
        rm = re.search(r"(?:REASON|DESCRIPTION)\W{0,5}(.+)", text, re.IGNORECASE | re.DOTALL)
        return (vm.group(1).upper(), max(0, min(100, int(cm.group(1)))) if cm else 0,
                rm.group(1).strip().splitlines()[0].strip() if rm else "")
    vm = re.search(r"\b(CONFIRMED|UNCERTAIN|NORMAL)\b", text.upper())
    if vm:
        return vm.group(1), 0, "(parsed from free text)"
    return "ERROR", 0, "unparseable response"


class Verifier:
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 timeout_s: float = 180, max_tokens: int = 8000, temperature: float = 0.0,
                 window_s: float = 3, prompt_path: Path = FIRE / "prompts" / "fire_verify.txt"):
        base_url = base_url or os.getenv("VLM_API_URL", "https://openrouter.ai/api/v1")
        api_key = api_key or os.getenv("VLM_API_KEY", "")
        if not api_key or not model:
            raise ValueError("VLM_API_KEY (fire/.env) and verify.model (fire/config.yaml) must be set")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.window_s = window_s
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def classify(self, frames: list[bytes], crop: bytes | None, cls: str) -> Verdict:
        k = len(frames)
        content: list[dict] = [{"type": "text", "text": self.prompt.format(k=k, window=self.window_s, cls=cls)}]
        for i, jpg in enumerate(frames, 1):
            content += [{"type": "text", "text": f"[Frame {i}/{k}]"}, _image(jpg)]
        if crop:
            content += [{"type": "text", "text": "[Crop]"}, _image(crop)]
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
    """Live: each call runs in its own daemon thread so detection never waits for Qwen, and Ctrl+C
    never waits for a call still in progress (a ThreadPoolExecutor is joined at exit: up to timeout_s)."""

    def __init__(self, verifier: Verifier):
        self.verifier = verifier

    def submit(self, frames, crop, cls) -> Future:
        f: Future = Future()

        def work():
            try:
                f.set_result(self.verifier.classify(frames, crop, cls))
            except BaseException as e:           # classify never raises; belt and braces
                f.set_exception(e)
        threading.Thread(target=work, daemon=True, name="qwen").start()
        return f

    def close(self) -> None:
        pass


class SyncVerifier:
    """Offline tests: the call blocks; the pipeline applies the verdict latency_s later in video time."""

    def __init__(self, verifier: Verifier):
        self.verifier = verifier

    def submit(self, frames, crop, cls) -> Future:
        f: Future = Future()
        f.set_result(self.verifier.classify(frames, crop, cls))
        return f

    def close(self) -> None:
        pass


class NoVerifier:
    """No Qwen: every event is UNVERIFIED and treated as an alert (step 2 behaviour)."""

    def submit(self, frames, crop, cls) -> Future:
        f: Future = Future()
        f.set_result(Verdict("UNVERIFIED", 0, "Qwen not used", "", 0.0))
        return f

    def close(self) -> None:
        pass
