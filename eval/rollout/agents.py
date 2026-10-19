"""LLM and heuristic agents for eval rollouts."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

import config
from config import OLLAMA_BASE_URL

# torch/transformers are imported lazily so the game engine, opponents and
# metrics can be imported (and unit-tested) without a GPU stack.


class HeuristicAgent:
    """Fast baseline without GPU inference."""

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        legal = obs.get("legal_actions") or []
        game = obs.get("game", "")
        if not legal:
            return "heuristic fallback", None
        if game in ("pd-classic", "pd-tight", "pd-high-temptation", "ipd-stage"):
            action = "C" if "C" in legal else legal[0]
        elif game == "stag-hunt":
            action = "Stag" if "Stag" in legal else legal[0]
        elif game == "bos":
            action = "Opera" if "Opera" in legal else legal[0]
        elif game == "matching-pennies":
            action = legal[0]
        elif game == "negotiation":
            action = (1, 1, 1)
        elif game == "tic-tac-toe":
            action = legal[0]
        elif game == "auction":
            action = obs.get("private_value", 100)
        elif game == "divide-dollar":
            action = 0.5
        elif game == "p-beauty":
            action = 33
        else:
            action = legal[0]
        return f"heuristic: choose {action}", action


class LoRALLMAgent:
    """Qwen + LoRA agent; output format matches DPO training data."""

    def __init__(
        self, model, tokenizer, max_new_tokens: int = config.EVAL_MAX_TOKENS, *,
        strict_output: bool = False, do_sample: bool = config.EVAL_DO_SAMPLE,
        temperature: float = config.EVAL_TEMPERATURE, top_p: float = config.EVAL_TOP_P,
        reasoning_format: str = config.EVAL_REASONING_FORMAT,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.strict_output = strict_output
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        if reasoning_format not in _REASONING_FORMATS:
            raise ValueError(f"reasoning_format must be one of {sorted(_REASONING_FORMATS)}")
        self.reasoning_format = reasoning_format
        self.last_meta: dict[str, Any] = {}

    def _generate(self, obs: dict[str, Any]) -> str:
        n_legal = len(obs.get("legal_actions") or [])
        text = _format_llm_prompt(self.tokenizer, obs["prompt"], self.reasoning_format,
                                  n_legal=n_legal)
        import torch
        from transformers import GenerationConfig

        # The chat template already inserts BOS; tokenizing with special tokens
        # again would double it (Gemma 3 degrades noticeably with two BOS).
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            pad_token_id=pad_id,
            eos_token_id=stop_token_ids(self.model, self.tokenizer),
        )
        if self.do_sample and obs.get("seed") is not None:
            torch.manual_seed(int(obs["seed"]))
        self.model.eval()
        # Kwarg do_sample overrides model.generation_config (Llama base defaults to sample).
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=gen_cfg, do_sample=self.do_sample)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        vocab_size = len(self.tokenizer)
        token_ids = [int(token_id) for token_id in new_tokens.detach().cpu().tolist()]
        invalid_ids = [token_id for token_id in token_ids if not 0 <= token_id < vocab_size]
        if invalid_ids:
            raise RuntimeError(
                "model.generate() returned token IDs outside the tokenizer vocabulary: "
                f"vocab_size={vocab_size}, invalid_ids={invalid_ids[:10]}. "
                "Do not filter these IDs: that silently corrupts rollout reasoning."
            )
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        # One malformed sample (e.g. <p>…</p> instead of <think>…</think>) must
        # not kill a multi-hour shard: regenerate once before giving up.
        attempts = int(__import__("os").environ.get("BLIND_STRICT_ATTEMPTS", "4")) if self.strict_output else 1
        last_response = ""
        for attempt in range(attempts):
            response = self._generate(obs)
            last_response = response
            if not response.strip():
                if attempt + 1 < attempts:
                    continue
                raise RuntimeError("model.generate() produced an empty blind-rollout response")
            reasoning, action, parsed_ok = parse_response(response, obs)
            if not self.strict_output or (reasoning and parsed_ok):
                break
        else:
            raise RuntimeError(
                "blind model did not return non-empty reasoning and a valid <action> "
                f"after {attempts} attempts; raw response={last_response[:500]!r}"
            )

        if action is None:
            action = _fallback_action(obs)
            reasoning = reasoning or f"fallback: {action}"
            parsed_ok = False
        self.last_meta = {"parsed_ok": parsed_ok, "raw_response": last_response}
        return reasoning, action


class OllamaAgent:
    """Frontier eval via local Ollama HTTP API (no HF / bitsandbytes)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = OLLAMA_BASE_URL,
        max_new_tokens: int = config.EVAL_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.last_meta: dict[str, Any] = {}

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        prompt = obs["prompt"]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": 0,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed ({self.base_url}, model={self.model}). "
                f"Is the server running? "
                f"Try: /scratch/workspace/eunbiyoon_umass_edu-paper/ollama-local/scripts/start.sh"
            ) from exc

        response = body.get("message", {}).get("content", "") or ""
        reasoning, action, parsed_ok = parse_response(response, obs)
        if action is None:
            action = _fallback_action(obs)
            reasoning = reasoning or f"fallback: {action}"
            parsed_ok = False
        self.last_meta = {"parsed_ok": parsed_ok, "raw_response": response}
        return reasoning, action


class LiteLLMAgent:
    """API frontier reference (Haiku 4.5 row) via LiteLLM.

    Uses the exact prompt and output contract the local models get, so the
    reference rows are comparable. Requires ``pip install litellm`` and the
    provider key (e.g. ANTHROPIC_API_KEY) in the environment.
    """

    def __init__(
        self, model: str = config.LITELLM_HAIKU_MODEL, *,
        max_new_tokens: int = config.EVAL_MAX_TOKENS,
        temperature: float = config.EVAL_TEMPERATURE,
        do_sample: bool = config.EVAL_DO_SAMPLE,
        reasoning_format: str = config.EVAL_REASONING_FORMAT,
        max_retries: int = 5,
    ) -> None:
        import litellm  # noqa: F401  (fail fast if missing)

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature if do_sample else 0.0
        self.reasoning_format = reasoning_format
        self.max_retries = max_retries
        self.last_meta: dict[str, Any] = {}

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        import litellm

        n_legal = len(obs.get("legal_actions") or [])
        content = obs["prompt"] + output_contract(self.reasoning_format, n_legal)
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            num_retries=self.max_retries,
        )
        text = response.choices[0].message.content or ""
        reasoning, action, parsed_ok = parse_response(text, obs)
        if action is None:
            action = _fallback_action(obs)
            reasoning = reasoning or f"fallback: {action}"
            parsed_ok = False
        self.last_meta = {"parsed_ok": parsed_ok, "raw_response": text}
        return reasoning, action


_REASONING_FORMATS = ("concise", "slots")

_CONCISE_CONTRACT = (
    "\n\nThe opponent's current-round action is not observable. Reason only from "
    "the game description and completed-round history shown above. "
    "Keep the reasoning concise: at most 80 words. "
    "Respond exactly as <think>your non-empty reasoning</think>"
    "<action>one legal action</action>."
)

# Slot form elicits the [Prior][Update][EV][Decision] structure the coupling
# metric parses (eval/metric/coupling.py). A+beta ignores the blind reasoning
# format (it is the rejected side; the chosen side is re-written by the teacher);
# Hypothesis B's coupling filter needs the [EV] slot to exist.
_SLOTS_CONTRACT_HEAD = (
    "\n\nThe opponent's current-round action is not observable. Reason only from "
    "the game description and completed-round history above — never state the "
    "opponent's move this round as observed.\n"
    "Write one <think> block with exactly these four sections, in order:\n"
    "[Prior] your belief about the opponent's action this round, as rough probabilities.\n"
    "[Update] how the completed-round history (if any) shifts that belief.\n"
)
_SLOTS_EV_ALL = (
    "[EV] for EACH legal action, the probability-weighted expected value as "
    "arithmetic, e.g. EV(C) = 0.7*3 + 0.3*0 = 2.1\n"
)
# Above a few dozen legal actions (negotiation ~125, auction ~201) the model
# spends its whole budget enumerating EV(...) for every one and then closes
# </think> WITHOUT a [Decision] line — ~20% of negotiation rounds came back
# malformed this way. A 3-5 action shortlist keeps [EV] parseable and leaves
# room for [Decision]. Small games (<= threshold) keep the exhaustive form the
# coupling filter prefers.
_SLOTS_EV_SHORTLIST = (
    "[EV] name the 3-5 most promising legal actions and, for each, the "
    "probability-weighted expected value as arithmetic, e.g. "
    "EV([0,4,2]) = 0.5*8 + 0.5*10 = 9\n"
)
_SLOTS_CONTRACT_TAIL = (
    "[Decision] the legal action with the highest EV.\n"
    "Keep each section to one short sentence. Then one <action> block with that action.\n"
    "Respond exactly as <think>[Prior] ... [Update] ... [EV] ... [Decision] ...</think>"
    "<action>one legal action</action>."
)
_EV_SHORTLIST_THRESHOLD = 12


def _slots_contract(n_legal: int | None) -> str:
    ev = _SLOTS_EV_SHORTLIST if (n_legal or 0) > _EV_SHORTLIST_THRESHOLD else _SLOTS_EV_ALL
    return _SLOTS_CONTRACT_HEAD + ev + _SLOTS_CONTRACT_TAIL


def output_contract(reasoning_format: str, n_legal: int | None) -> str:
    return _slots_contract(n_legal) if reasoning_format == "slots" else _CONCISE_CONTRACT


def _format_llm_prompt(tokenizer, prompt: str, reasoning_format: str = "concise",
                       *, n_legal: int | None = None) -> str:
    """Use chat template when present (Instruct); else raw prompt (pretrained base).

    This is the ONLY place a game prompt becomes model input. DPO pairs are
    rendered through ``format_training_prompt`` -> here, so training and
    evaluation see byte-identical inputs.
    """
    contract = output_contract(reasoning_format, n_legal)
    messages = [{"role": "user", "content": prompt + contract}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt + contract + "\n"


def legal_count_from_prompt(prompt: str) -> int | None:
    """Recover |legal actions| from a stored game prompt (for the EV shortlist)."""
    m = re.search(r"Your legal actions: \[(.*?)\]", prompt)
    if m:
        return len([x for x in m.group(1).split(",") if x.strip()])
    m = re.search(r"Legal moves: \[(.*?)\]", prompt)
    if m:
        return len([x for x in m.group(1).split(",") if x.strip()])
    if "Legal proposals (" in prompt:
        m = re.search(r"Legal proposals \((\d+) total\)", prompt)
        return int(m.group(1)) if m else 125
    if "integer bid in [0, 200]" in prompt:
        return 201
    if "share as a float" in prompt:
        return 101
    if "integer guess in [0, 100]" in prompt:
        return 101
    return None


def format_training_prompt(tokenizer, prompt: str,
                           reasoning_format: str = config.EVAL_REASONING_FORMAT) -> str:
    """Render a stored DPO-pair prompt exactly as eval renders a live prompt."""
    return _format_llm_prompt(tokenizer, prompt, reasoning_format,
                              n_legal=legal_count_from_prompt(prompt))


def stop_token_ids(model, tokenizer) -> list[int] | int:
    """EOS ids that end an assistant turn.

    ``tokenizer.eos_token_id`` alone is wrong for Gemma 3 (``<eos>``) whose
    turns end with ``<end_of_turn>``; overriding the generation config with it
    makes Gemma run on into fabricated user turns until max_new_tokens.
    """
    ids: list[int] = []
    gen = getattr(model, "generation_config", None)
    for value in (getattr(gen, "eos_token_id", None), tokenizer.eos_token_id):
        if value is None:
            continue
        ids.extend(value if isinstance(value, (list, tuple)) else [value])
    for token in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(token) if hasattr(tokenizer, "convert_tokens_to_ids") else None
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            ids.append(tid)
    ids = list(dict.fromkeys(ids))
    return ids[0] if len(ids) == 1 else ids


_REASONING_TAGS = r"think|p|reasoning|thinking|thought|analysis"


def _parse_response(text: str, obs: dict[str, Any]) -> tuple[str, Any]:
    thinking, action, _ = parse_response(text, obs)
    return thinking, action


def parse_response(text: str, obs: dict[str, Any]) -> tuple[str, Any, bool]:
    """Return (reasoning, action, parsed_ok).

    ``parsed_ok`` is True only when an ``<action>`` tag resolves unambiguously
    to a legal action. Every other path (missing tag, last-line guess, numeric
    default, ambiguous substring) still returns an action so the episode can
    continue, but is reported as a parse failure.
    """
    thinking = ""
    m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m_think:
        thinking = m_think.group(1).strip()
    else:
        # Sampling sometimes wraps the reasoning in <p>...</p> (or mismatches the
        # open/close tag). Recover the reasoning as the text before <action>,
        # tolerating any single wrapper tag on either side.
        m_alt = re.search(
            rf"(?:<(?:{_REASONING_TAGS})>)?\s*(.+?)\s*(?:</(?:{_REASONING_TAGS})>)?\s*<action>",
            text,
            re.DOTALL,
        )
        if m_alt:
            thinking = m_alt.group(1).strip()
        else:
            pre = text.split("<action>", 1)[0]
            thinking = re.sub(r"^\s*<[a-z]+>|</[a-z]+>\s*$", "", pre.strip(), flags=re.I).strip()

    tagged = True
    action_raw = None
    m_act = re.findall(r"<action>(.*?)</action>", text, re.DOTALL)
    if m_act:
        action_raw = m_act[-1].strip()
    else:
        tagged = False
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line and not line.startswith("<"):
                action_raw = line
                break

    if not action_raw:
        return thinking, None, False
    action, exact = _coerce_action(action_raw, obs)
    return thinking, action, bool(tagged and exact)


def _coerce_action(raw: str, obs: dict[str, Any]) -> tuple[Any, bool]:
    """Map raw action text to a legal action; the flag says whether it was exact."""
    legal = obs.get("legal_actions") or []
    game = obs.get("game", "")
    raw = raw.strip()

    if game == "negotiation":
        nums = [int(x) for x in re.findall(r"-?\d+", raw)]
        cap = config.BARGAIN_CAPACITY
        if len(nums) == 3 and all(0 <= n <= cap for n in nums):
            return tuple(nums), True
        return (1, 1, 1), False

    if game == "tic-tac-toe":
        nums = re.findall(r"\d+", raw)
        if len(nums) == 1 and int(nums[0]) in legal:
            return int(nums[0]), True
        return (legal[0] if legal else 0), False

    if game in ("auction", "p-beauty"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
        if len(nums) == 1 and float(nums[0]).is_integer() and int(float(nums[0])) in legal:
            return int(float(nums[0])), True
        fallback = obs.get("private_value", 100) if game == "auction" else 33
        return fallback, False

    if game == "divide-dollar":
        nums = re.findall(r"\d*\.?\d+", raw)
        if len(nums) == 1:
            share = round(float(nums[0]), 2)
            if share in legal:
                return share, True
        return 0.5, False

    for candidate in legal:
        if str(candidate).lower() == raw.lower():
            return candidate, True
    hits = [c for c in legal if re.search(rf"\b{re.escape(str(c))}\b", raw, re.I)]
    if len(hits) == 1:
        return hits[0], True
    return (legal[0] if legal else raw), False


def _fallback_action(obs: dict[str, Any]) -> Any:
    legal = obs.get("legal_actions") or []
    return legal[0] if legal else None
