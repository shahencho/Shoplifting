"""OpenAI-compatible VLM client, concealment prompt and verdict parser."""
from __future__ import annotations

import base64
import json
import re
import time
from collections import deque
from dataclasses import dataclass

from openai import OpenAI

VERDICTS = ("CONFIRMED", "UNCERTAIN", "NORMAL")

SYSTEM_PROMPT = (
    "You are a retail loss-prevention analyst reviewing CCTV stills. "
    "You judge body movements and handling of goods only. "
    "You never try to identify who a person is."
)

USER_PROMPT = """You are given {k} frames from one CCTV clip in a shop, in chronological order, each labelled [Frame i/{k}].

Task: decide whether a person CONCEALS merchandise, i.e. puts a product into a bag, backpack, pocket, or under clothing, instead of keeping it visible in hand or in a basket/cart.

Instructions:
1. Compare the frames in order. Describe what each visible person's hands do and where items go between frames.
2. List any specific concealment actions you see (which frame, which hand, where the item went). If none, say "none".
3. Normal behaviour to NOT flag: browsing, picking up and putting back, holding items openly, placing items in a basket or cart, using a phone, adjusting own clothing or bag without a product.
4. Give your verdict:
   - CONFIRMED: concealment is clearly visible.
   - UNCERTAIN: suspicious movement toward bag/pocket/clothing with a product, but not fully visible.
   - NORMAL: no concealment.

Reply with ONLY a JSON object, no markdown fences:
{{"observations": "<frame-by-frame comparison>", "concealment_actions": ["<action>", ...], "verdict": "CONFIRMED|UNCERTAIN|NORMAL", "confidence": <0-100>, "explanation": "<one line>"}}"""


@dataclass
class Verdict:
    verdict: str            # CONFIRMED / UNCERTAIN / NORMAL / ERROR
    confidence: int
    explanation: str
    actions: list[str]
    raw: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    reasoning_tokens: int = 0


def parse_response(text: str) -> tuple[str, int, str, list[str]]:
    """Parse the model reply. Falls back to regex if the JSON is malformed."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).strip().upper()
            if v in VERDICTS:
                conf = int(float(d.get("confidence", 0)))
                acts = d.get("concealment_actions") or []
                if isinstance(acts, str):
                    acts = [acts]
                return v, max(0, min(100, conf)), str(d.get("explanation", "")).strip(), [str(a) for a in acts]
        except (ValueError, TypeError):
            pass
    # Fallback: first verdict keyword and first number after "confidence"
    vm = re.search(r"\b(CONFIRMED|UNCERTAIN|NORMAL)\b", text.upper())
    cm = re.search(r"confidence\D{0,5}(\d{1,3})", text, re.IGNORECASE)
    if vm:
        return vm.group(1), int(cm.group(1)) if cm else 0, "(parsed from free text)", []
    return "ERROR", 0, "unparseable response", []


class RateLimiter:
    """Sliding-window cap on calls per minute."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        if self.per_min <= 0:
            return
        now = time.monotonic()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        if len(self.calls) >= self.per_min:
            time.sleep(60 - (now - self.calls[0]) + 0.1)
        self.calls.append(time.monotonic())


class VLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, *, temperature: float = 0.0,
                 max_tokens: int = 600, timeout_s: float = 90, max_retries: int = 3,
                 rate_limit_per_min: int = 10):
        if not api_key or not model:
            raise ValueError("VLM_API_KEY and VLM_MODEL_NAME must be set in .env")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.limiter = RateLimiter(rate_limit_per_min)

    def classify(self, jpegs: list[bytes]) -> Verdict:
        k = len(jpegs)
        content: list[dict] = [{"type": "text", "text": USER_PROMPT.format(k=k)}]
        for i, jpg in enumerate(jpegs, 1):
            content.append({"type": "text", "text": f"[Frame {i}/{k}]"})
            b64 = base64.b64encode(jpg).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

        last_err = ""
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    extra_body={"usage": {"include": True}},  # OpenRouter: return cost
                )
            except Exception as e:  # network / rate-limit / server errors
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt * 2)
                continue
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            verdict, conf, expl, acts = parse_response(text)
            u = resp.usage
            details = getattr(u, "completion_tokens_details", None)
            return Verdict(
                verdict, conf, expl, acts, text, time.monotonic() - t0,
                getattr(u, "prompt_tokens", 0) or 0,
                getattr(u, "completion_tokens", 0) or 0,
                float(getattr(u, "cost", 0) or 0),
                getattr(details, "reasoning_tokens", 0) or 0,
            )
        return Verdict("ERROR", 0, last_err[:300], [], "", 0.0)
