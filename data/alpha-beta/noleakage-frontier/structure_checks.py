#!/usr/bin/env python3
"""Structural validation of a frontier paraphrase.

Two checks, both on the teacher's <think>/<action> output:
  * validate_pinned_action       — <action> and [DECISION] agree with the solver
  * validate_reasoning_structure — ordered [Prior]/[Update]/[EV]/[DECISION] with
                                   posterior-weighted EV arithmetic (waived on
                                   the terminal round, paper Appendix F Ex. 1)
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

# _action_text lives with the prompt it formats; path-load it like the rest of
# this tree so the module also works when imported on its own.
_PROMPT_MOD = "noleak_frontier_prompt"
if _PROMPT_MOD in sys.modules:
    _prompt = sys.modules[_PROMPT_MOD]
else:
    _spec = importlib.util.spec_from_file_location(
        _PROMPT_MOD, Path(__file__).with_name("frontier_prompt.py")
    )
    _prompt = importlib.util.module_from_spec(_spec)
    sys.modules[_PROMPT_MOD] = _prompt
    _spec.loader.exec_module(_prompt)
_action_text = _prompt._action_text

DECISION_RE = re.compile(r"\[\s*decision\s*\]\s*(?:play|choose|bid|propose)?\s*([^\n<]+)", re.I)
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)
SECTION_RE = re.compile(r"\[\s*(Prior|Update|EV|Decision)\s*\]", re.I)
UNCERTAINTY_RE = re.compile(
    r"\b(?:posterior|probabilit(?:y|ies)|likelihood|chance|belief|distribution|"
    r"type|types|uncertain|expect(?:ed|ation)?)\b",
    re.I,
)
EV_ARITHMETIC_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|\b0?\.\d+\b).{0,80}(?:[+*×]|\b(?:times|weighted|average)\b)",
    re.I | re.S,
)
PROBABILITY_VALUE_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*%|\b0?\.\d+\b)")


def validate_pinned_action(text: str, solver_action: Any) -> list[str]:
    """Check that both output action fields agree with the solver label."""
    expected = _action_text(solver_action)
    errors: list[str] = []
    actions = ACTION_RE.findall(text)
    if len(actions) != 1:
        errors.append(f"expected one <action> block, found {len(actions)}")
    elif _action_text(actions[0]) != expected:
        errors.append(f"<action> is {_action_text(actions[0])!r}, expected {expected!r}")
    decisions = DECISION_RE.findall(text)
    if len(decisions) != 1:
        errors.append(f"expected one [DECISION], found {len(decisions)}")
    else:
        decision = decisions[0].strip().rstrip(". ")
        if not _decision_matches(decision, expected):
            errors.append(f"[DECISION] is {decision!r}, expected {expected!r}")
    return errors


def _decision_matches(decision: str, expected: str) -> bool:
    """Accept a bare [DECISION] token or a sentence that closes on it.

    The authoritative bare token is the <action> block (checked separately); the
    [DECISION] slot is the softer closing-line consistency check. The local 7B
    teacher routinely writes it as prose ("... therefore the decision is to play
    D") rather than a lone token, so match an exact hit or a trailing whole-word
    hit on the expected action.
    """
    want = expected.casefold().strip()
    got = decision.casefold().strip()
    if got == want:
        return True
    tokens = re.findall(r"[\w.\-]+", got)
    return bool(tokens) and tokens[-1] == want


def validate_reasoning_structure(
    reasoning: str, *, is_final_round: bool = False
) -> list[str]:
    """Validate ordered belief-state sections and posterior-weighted EV work.

    On a final round there is no future and no posterior to average over: the
    solver reduces to the stage-game best response and the [EV] slot states the
    raw stage payoffs (paper Appendix F, Example 1). The posterior-weighted
    arithmetic requirement is therefore waived for the terminal round only.
    """
    errors: list[str] = []
    matches = list(SECTION_RE.finditer(reasoning))
    names = [match.group(1).casefold() for match in matches]
    expected = ["prior", "update", "ev", "decision"]
    if names != expected:
        return [
            "reasoning must contain exactly [Prior], [Update], [EV], and "
            "[DECISION] in that order"
        ]

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reasoning)
        sections[names[index]] = reasoning[match.end():end].strip()
    for name in expected:
        if not sections[name]:
            errors.append(f"[{name.upper()}] section must not be empty")

    belief_text = f"{sections['prior']} {sections['update']}"
    # A distribution over opponent types/actions counts whether it is spelled out
    # ("cooperator type with probability 0.7") or written as the bare numbers the
    # local 7B teacher emits under greedy decoding ("[Prior] C 0.7, D 0.3"). This
    # mirrors the [EV] check below, which also accepts a bare probability value.
    if not UNCERTAINTY_RE.search(belief_text) and not PROBABILITY_VALUE_RE.search(belief_text):
        errors.append("[Prior]/[Update] must express uncertainty over opponent types or actions")
    ev = sections["ev"]
    if is_final_round:
        return errors
    if not UNCERTAINTY_RE.search(ev) and not PROBABILITY_VALUE_RE.search(ev):
        errors.append("[EV] must identify the posterior probability or expectation being used")
    if not EV_ARITHMETIC_RE.search(ev):
        errors.append("[EV] must show probability-weighted arithmetic")
    return errors


def _coupling_parser():
    """Paper Appendix E parser (eval/metric/coupling.py), shared with Table 3."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from eval.metric import coupling

    return coupling


_NUMERIC_EXPR = re.compile(r"^[\d\s.+\-*/()]+$")


def _numeric(expr: str) -> float | None:
    expr = expr.strip().replace("×", "*")
    if not expr or not _NUMERIC_EXPR.match(expr) or not re.search(r"\d", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # digits/operators only (regex-guarded)
    except Exception:
        return None


def _arithmetic_errors(reasoning: str, tolerance: float) -> list[str]:
    """Each EV(x) = e1 = e2 = ... chain: consecutive numeric steps must agree."""
    errors = []
    for clause in re.split(r"(?=EV\()|[;\n]", reasoning):
        if "=" not in clause:
            continue
        steps = [_numeric(part) for part in clause.split("=")[1:]]
        for a, b in zip(steps, steps[1:]):
            if a is not None and b is not None and abs(a - b) > tolerance:
                errors.append(f"[EV] arithmetic step is wrong: {clause.strip()[:80]!r}")
                break
    return errors


def validate_ev_supports_action(text: str, solver_action: Any, *, tolerance: float = 0.05) -> list[str]:
    """Paper §2.2: the rationale must support the solver action.

    * the supplied action is the unique argmax of the stated EVs, judged by the
      same parser that computes reasoning-action coupling (Table 3);
    * every purely numeric "= expr = value" step evaluates to its stated value
      (catches "0.9*1 + 0.1*5 = 2.7"-style arithmetic slips).
    """
    coupling = _coupling_parser()
    reasoning_match = re.search(r"<think>(.*?)</think>", text, re.I | re.S)
    reasoning = reasoning_match.group(1) if reasoning_match else text
    errors: list[str] = []
    evs = coupling.parse_ev_slot(reasoning)
    if len(evs) < 2:
        errors.append("[EV] must state expected values for at least two actions")
    else:
        result = coupling.coupling_for_round(reasoning, _action_text(solver_action))
        if not result.coupled:
            best = coupling.argmax_stated_ev(evs)
            errors.append(
                f"[EV] argmax is {best!r}, not the solver action {_action_text(solver_action)!r}"
            )
    errors.extend(_arithmetic_errors(reasoning, tolerance))
    return errors

