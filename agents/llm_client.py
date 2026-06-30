from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    raw_text: str
    parsed: dict | None

class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse:
        ...


class MockLLMClient(LLMClient):
    def __init__(self, seed: int | None = None, noise: float = 0.15) -> None:
        self._rng = random.Random(seed)
        self.noise = noise

    def complete(self, system: str, user: str) -> LLMResponse:
        text_blob = (system + user).lower()

        coop_lean = 0.5
        if "trust=very high" in text_blob or "trust=high" in text_blob:
            coop_lean += 0.25
        if "trust=low" in text_blob or "trust=very low" in text_blob:
            coop_lean -= 0.25
        if "resentment=high" in text_blob or "resentment=very high" in text_blob:
            coop_lean -= 0.3
        if "fear=high" in text_blob or "fear=very high" in text_blob:
            coop_lean -= 0.15
        if "predicted next action: defect" in text_blob:
            coop_lean -= 0.2
        if "predicted next action: cooperate" in text_blob:
            coop_lean += 0.15
        if "scarcity" in text_blob:
            coop_lean -= 0.2

        coop_lean += self._rng.uniform(-self.noise, self.noise)
        coop_lean = max(0.0, min(1.0, coop_lean))

        action = "cooperate" if self._rng.random() < coop_lean else "defect"
        rationale = (
            f"Estimated cooperation propensity {coop_lean:.2f} given current"
            f"trust/reputation signals; chose {action}."
        )
        parsed = {"action": action, "rationale": rationale}
        return LLMResponse(raw_text=json.dumps(parsed), parsed=parsed)

class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int=400) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        parsed = _safe_json_extract(text)
        return LLMResponse(raw_text=text, parsed=parsed)
    
def _safe_json_extract(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    
def get_client(kind: str = "mock", **kwargs) -> LLMClient:
    if kind == "mock":
        return MockLLMClient(**kwargs)
    if kind == "anthropic":
        return AnthropicClient(**kwargs)
    raise ValueError(f"Unknown LLM clinet kind: {kind!r}")