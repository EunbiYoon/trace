#!/usr/bin/env python3
"""Independent audit of TRACE DPO pair files against the paper (Alg. 1, Eq. 1-3, §2.2).

Recomputes every check from the blind trajectories instead of trusting
provenance, then exits non-zero if any paper invariant is violated. Run it on
freshly generated data before training:

    python data/audit_pairs.py                       # data/*/result defaults
    python data/audit_pairs.py --legacy-opponents    # audit pre-fix (ee390e0) data

Invariants (all must be 0):
  chosen action is not the Eq. 1 best response / solver_action
  Eq. 3 counterfactual return differs from provenance or is <= recorded return
  rejected / prompt differ from the blind round
  chosen and rejected share the same <think> (blind-verbatim chosen)
  chosen [EV] argmax != chosen action (paper Appendix E parser)
  prompt payoff table != payoff table used by the solver
  repeated-game pair against a non-replayable (stochastic) opponent
"""

import collections
import json
import re
import sys
from functools import lru_cache

R = next((a for a in sys.argv[1:] if not a.startswith("--")),
         str(__import__("pathlib").Path(__file__).resolve().parents[1]))
COOP = ("C", "Stag", "Opera", "H")
DEF = ("D", "Hare", "Football", "T")


def first(pref, legal, fb):
    return next((a for a in pref if a in legal), fb)


LEGACY = "--legacy-opponents" in sys.argv


def old_opp(name, hist, legal):
    """Replay the opponent's next move. Current policies come from
    eval/rollout/opponents.py; --legacy-opponents uses the pre-fix (ee390e0)
    Pavlov / tit-for-two-tats that generated the committed Qwen-7B data."""
    if not LEGACY:
        return _current_opp(name, hist, legal)
    last_agent = hist[-1][0] if hist else None
    if name == "always_cooperate":
        return first(COOP, legal, legal[0])
    if name == "always_defect":
        return first(DEF, legal, legal[-1])
    if name == "tit_for_tat":
        if last_agent is None:
            return first(COOP, legal, legal[0])
        return last_agent if last_agent in legal else legal[0]
    if name == "grim_trigger":
        return first(DEF, legal, legal[-1]) if any(a in DEF for a, _ in hist) else first(COOP, legal, legal[0])
    if name == "pavlov":  # old code: repeat own last move, start legal[0]
        return hist[-1][1] if hist else legal[0]
    if name == "tit_for_two_tats":  # old code: copy unless two consecutive defects
        streak = 0
        for a, _ in hist:
            streak = streak + 1 if a in DEF else 0
        if streak >= 2:
            return first(DEF, legal, legal[-1])
        return last_agent if (last_agent in legal and last_agent is not None) else legal[0]
    return None  # stochastic: not reconstructible


def _current_opp(name, hist, legal):
    import random

    sys.path.insert(0, R)
    from eval.rollout.opponents import OPPONENT_CLASSES, make_opponent

    if name not in ("always_cooperate", "always_defect", "tit_for_tat", "grim_trigger",
                    "pavlov", "tit_for_two_tats") or name not in OPPONENT_CLASSES:
        return None
    opp = make_opponent(name)
    opp.reset(random.Random(0), list(legal))
    last = None
    for a, b in hist:
        opp.act(legal_actions=list(legal), last_agent_action=last)
        opp.observe(a, b)
        last = a
    return opp.act(legal_actions=list(legal), last_agent_action=last)


def load_blind(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def payoff_table(traj):
    return {(p["own"], p["opponent"]): float(p["payoff"][0]) for p in traj.get("payoffs", [])}


def prompt_table(prompt):
    return {(a, b): float(u) for a, b, u, _ in
            re.findall(r"If you play (\S+) and opp plays (\S+):\s*\((-?[\d.]+),\s*(-?[\d.]+)\)", prompt)}


def br_matrix(u, legal, opp):
    best = max(u[(a, opp)] for a in legal)
    return [a for a in legal if u[(a, opp)] == best]


def ha_return(traj, t, a_star, u):
    """Eq. 3: recorded prefix + u0(a*, a1_t) + V(t+1, h*)."""
    legal = traj["legal_actions"]
    own, opp, name = traj["own_actions"], traj["opponent_actions"], traj["opponent"]
    T = len(own)
    prefix = sum(u[(own[s], opp[s])] for s in range(t))
    hist = tuple(zip(own[:t], opp[:t])) + ((a_star, opp[t]),)

    @lru_cache(None)
    def V(s, h):
        if s == T:
            return 0.0
        b = old_opp(name, h, legal)
        return max(u[(a, b)] + V(s + 1, h + ((a, b),)) for a in legal)

    return prefix + u[(a_star, opp[t])] + V(t + 1, hist)


def act_of(text):
    m = re.findall(r"<action>(.*?)</action>", text, re.S)
    return m[-1].strip() if m else None


def decision_of(text):
    m = re.findall(r"\[decision\]\s*(?:play\s+)?([A-Za-z]+|\[[\d,\s]+\]|\d+)", text, re.I)
    return m[-1] if m else None


def ev_of(text):
    ev = {}
    for m in re.finditer(r"EV\(\s*([^)]+?)\s*\)\s*=([^\n;]*?)(?=EV\(|\n|;|$)", text):
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(2).split("=")[-1])
        if nums:
            ev[m.group(1).strip()] = float(nums[0])
    return ev


LEAK = re.compile(r"opponent (played|chose|cooperat|defect)\w*\s+(this round|in round \d+)", re.I)


def norm(a):
    return re.sub(r"\s", "", str(a)) if a is not None else None


def audit_trace(pairs_path, blind):
    c = collections.Counter()
    ex = collections.defaultdict(list)
    rows = [json.loads(l) for l in open(pairs_path)]
    games = collections.Counter()
    opps = collections.Counter()
    for r in rows:
        pv = r["provenance"]
        g = pv["game"]
        games[g] += 1
        traj = blind[r["trajectory_id"]]
        opps[(pv["game_family"], traj["opponent"])] += 1
        t = pv["round_index"]
        chosen_a, rej_a = act_of(r["chosen"]), act_of(r["rejected"])
        c["reasoning:" + pv["chosen_reasoning"]] += 1
        c["filter:" + pv["filter"]] += 1
        # B/C: rejected and prompt are the blind round, verbatim
        if r["rejected"].strip() != traj["round_completions"][t].strip():
            c["rejected != blind completion"] += 1
        if r["prompt"].strip() != traj["round_prompts"][t].strip():
            c["prompt != blind round prompt"] += 1
        rec = traj["own_actions"][t]
        if norm(rej_a) != norm(rec):
            c["rejected action != recorded"] += 1
        if norm(pv["recorded_action"]) != norm(rec) or norm(pv["opponent_action"]) != norm(traj["opponent_actions"][t]):
            c["provenance recorded/opp action mismatch"] += 1
        # A: chosen action == solver == independent BR
        if norm(chosen_a) != norm(pv["solver_action"]):
            c["chosen action != solver_action"] += 1
        if norm(chosen_a) == norm(rec):
            c["chosen action == blind action (no flip)"] += 1
        fam = pv["game_family"]
        if fam in ("matrix", "bos", "matching-pennies"):
            u = payoff_table(traj)
            if chosen_a not in br_matrix(u, traj["legal_actions"], traj["opponent_actions"][t]):
                c["chosen not Eq.1 best response"] += 1
            pt = prompt_table(r["prompt"])
            if pt and pt != u:
                c["prompt payoffs != solver payoffs"] += 1
            recorded_R = sum(u[(a, b)] for a, b in zip(traj["own_actions"], traj["opponent_actions"]))
            if abs(recorded_R - pv["recorded_return"]) > 1e-9:
                c["recorded_return wrong"] += 1
            if len(traj["own_actions"]) > 1:
                # opponent replay consistency with the recorded play
                name = traj["opponent"]
                h = ()
                ok = True
                for a, b in zip(traj["own_actions"], traj["opponent_actions"]):
                    pred = old_opp(name, h, traj["legal_actions"])
                    if pred is not None and pred != b:
                        ok = False
                    h += ((a, b),)
                if old_opp(name, (), traj["legal_actions"]) is None:
                    c["repeated-game pair vs stochastic opponent"] += 1
                elif not ok:
                    c["recorded opponent moves != policy"] += 1
                else:
                    cf = ha_return(traj, t, chosen_a, u)
                    if abs(cf - pv["counterfactual_return"]) > 1e-9:
                        c["counterfactual_return != recomputed Eq.3"] += 1
                        ex["cf"].append((r["trajectory_id"], t, cf, pv["counterfactual_return"]))
                    if not cf > recorded_R:
                        c["Eq.3 return <= recorded (should be filtered)"] += 1
            else:
                if not u[(chosen_a, traj["opponent_actions"][t])] > recorded_R:
                    c["one-shot: no improvement"] += 1
        elif fam == "auction":
            v, b1 = traj["private_value"], int(traj["opponent_actions"][t])
            star = b1 + 1 if v > b1 + 1 else 0
            if int(chosen_a) != star:
                c["auction chosen != BR (b1+1 or 0)"] += 1
                ex["auction"].append((v, b1, chosen_a))
        elif fam == "bargaining":
            p1 = [int(x) for x in re.findall(r"\d+", str(traj["opponent_actions"][t]))]
            star = [4 - x for x in p1]
            if [int(x) for x in re.findall(r"\d+", chosen_a)] != star:
                c["bargaining chosen != take-the-rest"] += 1
        # §2.2 constraints on chosen reasoning
        if LEAK.search(r["chosen"]):
            c["leak regex hit (paper §D)"] += 1
        d = decision_of(r["chosen"])
        if d is not None and norm(d).lower() != norm(chosen_a).lower():
            c["[Decision] != <action>"] += 1
        sys.path.insert(0, R)
        from eval.metric.coupling import coupling_for_round, parse_ev_slot  # Appendix E parser

        think_m = re.search(r"<think>(.*?)</think>", r["chosen"], re.S)
        reasoning = think_m.group(1) if think_m else r["chosen"]
        if len(parse_ev_slot(reasoning)) >= 2:
            c["EV parsable"] += 1
            if not coupling_for_round(reasoning, str(chosen_a)).coupled:
                c["chosen EV argmax != action"] += 1
                c[f"  ...of which reasoning={pv['chosen_reasoning']}"] += 1
        think = lambda s: re.sub(r"<action>.*?</action>", "", s, flags=re.S).strip()
        if think(r["chosen"]) == think(r["rejected"]):
            c["chosen <think> == rejected <think> (blind-verbatim)"] += 1
        if pv.get("leakage_audit"):
            c["provenance leakage_audit non-empty"] += 1
    return len(rows), games, opps, c, ex


FATAL = (
    "chosen action != solver_action", "chosen not Eq.1 best response", "chosen action == blind action (no flip)",
    "counterfactual_return != recomputed Eq.3", "Eq.3 return <= recorded (should be filtered)",
    "rejected != blind completion", "prompt != blind round prompt", "rejected action != recorded",
    "chosen <think> == rejected <think> (blind-verbatim)", "chosen EV argmax != action",
    "prompt payoffs != solver payoffs", "repeated-game pair vs stochastic opponent",
    "recorded opponent moves != policy", "auction chosen != BR (b1+1 or 0)",
    "bargaining chosen != take-the-rest", "one-shot: no improvement",
)


def main():
    blind = load_blind(f"{R}/data/blind-rollout/result/a_beta_all.jsonl")
    bad = 0
    print("blind teacher model(s):", collections.Counter(t["frontier_model"] for t in blind.values()))
    print("blind opponents by family:", sorted(collections.Counter((t["game"], t["opponent"]) for t in blind.values() if t["game"] in ("pd-classic",)).items()))
    for v in ("core", "aux", "all", "rw"):
        n, games, opps, c, ex = audit_trace(f"{R}/data/alpha-beta/result/a_beta_{v}.jsonl", blind)
        print(f"\n===== a_beta_{v}: {n} pairs")
        print(" games:", dict(games))
        for k, val in sorted(c.items()):
            print(f"   {k}: {val}")
        for k, lst in ex.items():
            print("  examples", k, lst[:3])
        bad += sum(c[k] for k in FATAL)
    print("\nRESULT:", "PASS" if bad == 0 else f"FAIL ({bad} invariant violations)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
