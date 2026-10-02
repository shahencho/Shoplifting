from types import SimpleNamespace

from theft_demo.verify import Verifier


class FakeClient:
    """Stands in for the OpenAI client: records the request, answers CONFIRMED."""

    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kw):
        return self

    def create(self, **kw):
        self.requests.append(kw)
        msg = SimpleNamespace(content="VERDICT: CONFIRMED\nCONFIDENCE: 85\nDESCRIPTION: item into the pocket")
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50, completion_tokens_details=None, cost=0.0005)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _classify(**kw):
    v = Verifier("qwen/qwen3.6-plus", api_key="test", **kw)
    v.client = FakeClient()
    verdict = v.classify([b"jpg"] * 5)
    return verdict, v.client.requests[0]["extra_body"]


def test_thinking_off_is_sent_as_reasoning_effort_none():
    verdict, body = _classify(reasoning="none")
    assert body["reasoning"] == {"effort": "none"} and body["usage"] == {"include": True}
    assert (verdict.verdict, verdict.confidence, verdict.cost_usd) == ("CONFIRMED", 85, 0.0005)


def test_no_reasoning_setting_leaves_the_model_default():
    _, body = _classify()
    assert "reasoning" not in body
