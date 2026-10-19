"""CPU-only tests for the paper-aligned evaluation (no torch / model needed).

Run from the repo root:  python -m unittest eval.test_paper_eval -v
"""

from __future__ import annotations

import importlib.util
import itertools
import random
import re
import unittest
from pathlib import Path

import config
from eval.metric.metrics import aggregate_episode_results
from eval.rollout.agents import parse_response
from eval.rollout.games import TFT_SPLIT, opponent_schedule, run_game_batch
from eval.rollout.opponents import make_opponent
from eval.rollout import prompts

ROOT = Path(__file__).resolve().parents[1]


def _load_cf_utils():
    path = ROOT / "data" / "alpha-beta" / "counterfactual-horizon" / "utils.py"
    spec = importlib.util.spec_from_file_location("cf_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpponentSemantics(unittest.TestCase):
    L = ["C", "D"]

    def _play(self, name, agent_moves):
        opp = make_opponent(name)
        opp.reset(random.Random(0), self.L)
        out, last = [], None
        for a in agent_moves:
            o = opp.act(legal_actions=self.L, last_agent_action=last)
            out.append(o)
            opp.observe(a, o)
            last = a
        return out

    def test_tit_for_tat(self):
        self.assertEqual(self._play("tit_for_tat", list("CDDC")), list("CCDD"))

    def test_grim_trigger(self):
        self.assertEqual(self._play("grim_trigger", list("CDCC")), list("CCDD"))

    def test_pavlov_is_win_stay_lose_shift(self):
        # (C,C)->C ; (D,C)->D ; (D,D)->C ; (C,C)->C
        self.assertEqual(self._play("pavlov", list("CDDC")), list("CCDC"))

    def test_tit_for_two_tats_needs_two_defections(self):
        self.assertEqual(self._play("tit_for_two_tats", list("DCDDC")), list("CCCCD"))


class CounterfactualReplayMatchesEval(unittest.TestCase):
    """The training filter's opponent replay must equal the eval opponents,
    otherwise Eq. 2/3 credit is computed against a different opponent."""

    def test_all_histories_up_to_length_6(self):
        cf = _load_cf_utils()
        legal = ["C", "D"]
        for name in ("always_cooperate", "always_defect", "tit_for_tat",
                     "grim_trigger", "pavlov", "tit_for_two_tats"):
            for length in range(0, 7):
                for agent_moves in itertools.product(legal, repeat=length):
                    opp = make_opponent(name)
                    opp.reset(random.Random(0), legal)
                    history, last = [], None
                    for a in agent_moves:
                        o = opp.act(legal_actions=legal, last_agent_action=last)
                        self.assertEqual(o, cf.opponent_action(name, tuple(history), legal),
                                         f"{name} diverges after {history}")
                        opp.observe(a, o)
                        history.append((a, o))
                        last = a


class PromptPayoffConsistency(unittest.TestCase):
    """The payoff table shown to the model must equal the reward it receives."""

    PATTERN = re.compile(r"If you play (\S+) and opp plays (\S+):\s*\((-?[\d.]+), (-?[\d.]+)\)")

    def _check(self, prompt, payoff_fn):
        rows = self.PATTERN.findall(prompt)
        self.assertEqual(len(rows), 4)
        for a0, a1, u0, u1 in rows:
            self.assertEqual(payoff_fn(a0, a1), (float(u0), float(u1)), (a0, a1))

    def test_matrix_games(self):
        self._check(prompts.stag_hunt_prompt(round_idx=1, horizon=10, history=[]), prompts.stag_payoff)
        self._check(prompts.bos_prompt(), lambda a, b: tuple(map(float, prompts.bos_payoff(a, b))))
        self._check(prompts.matching_pennies_prompt(), lambda a, b: tuple(map(float, prompts.mp_payoff(a, b))))
        for g in ("pd-classic", "pd-tight", "pd-high-temptation"):
            p = prompts.repeated_pd_prompt(g, round_idx=1, horizon=10, history=[])
            self._check(p, lambda a, b, g=g: tuple(map(float, prompts.pd_payoff(g, a, b))))

    def test_stag_matches_paper_appendix_g(self):
        self.assertEqual(prompts.stag_payoff("Hare", "Hare"), (1.0, 1.0))
        self.assertEqual(prompts.stag_payoff("Hare", "Stag"), (3.0, 0.0))


class OpponentSchedule(unittest.TestCase):
    def test_round_robin_covers_pool(self):
        for game in ("auction", "negotiation", "bos"):
            pool = config.EVAL_OPPONENT_POOLS[game][1]
            sched = opponent_schedule(game, 12, 42)
            self.assertEqual(set(sched[: len(pool)]), set(pool), game)

    def test_schedule_is_model_independent_and_deterministic(self):
        self.assertEqual(opponent_schedule("pd-classic", 12, 42), opponent_schedule("pd-classic", 12, 42))

    def test_matrix_pool_weights_follow_appendix_i(self):
        sched = opponent_schedule("pd-classic", 20000, 7)
        share = sched.count("tit_for_tat") / sched.count("always_defect")
        self.assertAlmostEqual(share, 2.0, delta=0.15)


class ParseFlags(unittest.TestCase):
    OBS = {"game": "pd-classic", "legal_actions": ["C", "D"]}

    def test_tagged_exact(self):
        self.assertEqual(parse_response("<think>x</think><action>D</action>", self.OBS), ("x", "D", True))

    def test_untagged_is_failure(self):
        self.assertFalse(parse_response("<think>x</think>\nD", self.OBS)[2])

    def test_ambiguous_is_failure(self):
        self.assertFalse(parse_response("<think>x</think><action>C or D</action>", self.OBS)[2])

    def test_out_of_range_bid_is_failure(self):
        obs = {"game": "auction", "legal_actions": list(range(201)), "private_value": 90}
        self.assertFalse(parse_response("<think>x</think><action>250</action>", obs)[2])


class ScriptedSlotsAgent:
    """Emits slot-form text; always defects in PD and states EV(C) > EV(D)
    every third call (decoupled), and returns garbage every fifth call."""

    def __init__(self):
        self.calls = 0
        self.last_meta = {}

    def decide(self, obs):
        self.calls += 1
        legal = obs["legal_actions"]
        a = legal[-1] if isinstance(legal[-1], str) else legal[0]
        ev_hi, ev_lo = ("C", "D") if self.calls % 3 == 0 else ("D", "C")
        text = (f"<think>[Prior] p\n[Update] u\n[EV] EV({ev_hi}) = 0.5*4 + 0.5*2 = 3.0; "
                f"EV({ev_lo}) = 0.5*1 + 0.5*1 = 1.0\n[Decision] {a}</think><action>{a}</action>")
        if self.calls % 5 == 0:
            text = "I refuse to answer in the format."
        reasoning, action, ok = parse_response(text, obs)
        if action is None:
            action, ok = legal[0], False
        self.last_meta = {"parsed_ok": ok, "raw_response": text}
        return reasoning, action


class EndToEndMetrics(unittest.TestCase):
    def test_tft_split_and_parse_failures(self):
        agent = ScriptedSlotsAgent()
        eps = run_game_batch("pd-classic", agent, 6, 1)
        eps += run_game_batch("pd-classic", agent, 4, 1, opponent_name="tit_for_tat", split=TFT_SPLIT)
        m = aggregate_episode_results(eps)
        self.assertEqual(m.counts["main_episodes"], 6)
        self.assertEqual(m.counts["tft_episodes"], 4)
        tft_mean = sum(e.cumulative_reward for e in eps if e.split == TFT_SPLIT) / 4
        self.assertAlmostEqual(m.tft_cr, tft_mean)
        self.assertGreater(m.counts["parse_failures"], 0)
        self.assertIsNotNone(m.ac_id)
        self.assertLess(m.ac_id, 1.0)

    def test_no_parsable_coupling_is_none_not_zero(self):
        class Silent:
            def decide(self, obs):
                return "", obs["legal_actions"][0]
        m = aggregate_episode_results(run_game_batch("pd-tight", Silent(), 2, 0))
        self.assertIsNone(m.fc_id)


if __name__ == "__main__":
    unittest.main()
