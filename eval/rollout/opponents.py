"""Fixed opponent policies for eval rollouts (paper Appendix I).

Every policy is deterministic given the episode RNG, so a rollout is
reproducible from (game, episode seed). Matrix-game policies classify actions
as cooperative or defecting via ``COOPERATIVE`` / ``DEFECTING``.
"""

from __future__ import annotations

import random
from typing import Any

import config

COOPERATIVE = ("C", "Stag", "Opera", "H")
DEFECTING = ("D", "Hare", "Football", "T")


def _first_in(preferred: tuple[str, ...], legal: list[Any], fallback: Any) -> Any:
    for action in preferred:
        if action in legal:
            return action
    return fallback


def _cooperate(legal: list[Any]) -> Any:
    return _first_in(COOPERATIVE, legal, legal[0])


def _defect(legal: list[Any]) -> Any:
    return _first_in(DEFECTING, legal, legal[-1])


def _is_defect(action: Any) -> bool:
    return action in DEFECTING


class OpponentPolicy:
    name = "base"

    def reset(self, rng: random.Random, legal_actions: list[Any]) -> None:
        self._rng = rng
        self._legal = list(legal_actions)
        self._history: list[tuple[Any, Any]] = []

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        raise NotImplementedError

    def observe(self, agent_action: Any, opp_action: Any) -> None:
        self._history.append((agent_action, opp_action))


# --------------------------------------------------------------------------
# Repeated / one-shot 2x2 matrix games
# --------------------------------------------------------------------------
class AlwaysCooperate(OpponentPolicy):
    name = "always_cooperate"

    def act(self, *, legal_actions, last_agent_action=None):
        return _cooperate(legal_actions)


class AlwaysDefect(OpponentPolicy):
    name = "always_defect"

    def act(self, *, legal_actions, last_agent_action=None):
        return _defect(legal_actions)


class RandomOpponent(OpponentPolicy):
    name = "random"

    def act(self, *, legal_actions, last_agent_action=None):
        return self._rng.choice(legal_actions)


class EpsilonGreedy(OpponentPolicy):
    """Plays its cooperative action, but a uniformly random one with prob. 0.1."""

    name = "epsilon_greedy"
    epsilon = 0.1

    def act(self, *, legal_actions, last_agent_action=None):
        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal_actions)
        return _cooperate(legal_actions)


class TitForTat(OpponentPolicy):
    """Cooperates first, then copies the agent's previous action."""

    name = "tit_for_tat"

    def act(self, *, legal_actions, last_agent_action=None):
        if last_agent_action is None:
            return _cooperate(legal_actions)
        return last_agent_action if last_agent_action in legal_actions else _cooperate(legal_actions)


class GrimTrigger(OpponentPolicy):
    """Cooperates until the agent defects once, then defects forever."""

    name = "grim_trigger"

    def reset(self, rng, legal_actions):
        super().reset(rng, legal_actions)
        self._triggered = False

    def act(self, *, legal_actions, last_agent_action=None):
        if _is_defect(last_agent_action):
            self._triggered = True
        return _defect(legal_actions) if self._triggered else _cooperate(legal_actions)


class Pavlov(OpponentPolicy):
    """Win-stay, lose-shift: cooperate iff both players chose the same kind of
    action (both cooperative or both defecting) in the previous round."""

    name = "pavlov"

    def act(self, *, legal_actions, last_agent_action=None):
        if not self._history:
            return _cooperate(legal_actions)
        agent_prev, own_prev = self._history[-1]
        same = _is_defect(agent_prev) == _is_defect(own_prev)
        return _cooperate(legal_actions) if same else _defect(legal_actions)


class TitForTwoTats(OpponentPolicy):
    """Defects only after the agent defected in each of the last two rounds."""

    name = "tit_for_two_tats"

    def act(self, *, legal_actions, last_agent_action=None):
        last_two = [agent for agent, _ in self._history[-2:]]
        if len(last_two) == 2 and all(_is_defect(a) for a in last_two):
            return _defect(legal_actions)
        return _cooperate(legal_actions)


class GenerousTFT(OpponentPolicy):
    """Tit-for-tat that forgives a defection with probability 0.1."""

    name = "generous_tft"
    forgiveness = 0.1

    def act(self, *, legal_actions, last_agent_action=None):
        if _is_defect(last_agent_action) and self._rng.random() < self.forgiveness:
            return _cooperate(legal_actions)
        return TitForTat().act(legal_actions=legal_actions, last_agent_action=last_agent_action)


# --------------------------------------------------------------------------
# First-price auction: each bidder draws its own private value v ~ U{50..150}.
# --------------------------------------------------------------------------
class _Bidder(OpponentPolicy):
    def reset(self, rng, legal_actions):
        super().reset(rng, legal_actions)
        self._value = rng.randint(50, 150)

    def _bid(self) -> float:
        raise NotImplementedError

    def act(self, *, legal_actions, last_agent_action=None):
        target = self._bid()
        nums = [a for a in legal_actions if isinstance(a, (int, float))]
        return min(nums, key=lambda x: abs(float(x) - target)) if nums else legal_actions[0]


class TruthfulBidder(_Bidder):
    name = "truthful_bidder"

    def _bid(self):
        return self._value


class ShadedBidderLow(_Bidder):
    name = "shaded_bidder_low"

    def _bid(self):
        return 0.3 * self._value


class ShadedBidderHigh(_Bidder):
    name = "shaded_bidder_high"

    def _bid(self):
        return 0.7 * self._value


class RandomUniformBidder(_Bidder):
    name = "random_uniform_bidder"

    def _bid(self):
        return self._rng.uniform(0, self._value)


# --------------------------------------------------------------------------
# Three-issue bargaining, capacity c per issue, proposals are int triples.
# --------------------------------------------------------------------------
class GreedyBargainer(OpponentPolicy):
    name = "greedy_bargainer"

    def act(self, *, legal_actions, last_agent_action=None):
        c = config.BARGAIN_CAPACITY
        return (c, c, c)


class EqualSplitBargainer(OpponentPolicy):
    name = "equal_split_bargainer"

    def act(self, *, legal_actions, last_agent_action=None):
        h = config.BARGAIN_CAPACITY // 2
        return (h, h, h)


class NashBargainer(OpponentPolicy):
    """Claims all of its own highest-weight issue and half of the others."""

    name = "nash_bargainer"

    def act(self, *, legal_actions, last_agent_action=None):
        c = config.BARGAIN_CAPACITY
        weights = config.BARGAIN_OPP_WEIGHTS
        top = max(range(3), key=lambda i: weights[i])
        return tuple(c if i == top else c // 2 for i in range(3))


class RandomBargainer(OpponentPolicy):
    name = "random_bargainer"

    def act(self, *, legal_actions, last_agent_action=None):
        c = config.BARGAIN_CAPACITY
        return tuple(self._rng.randint(0, c) for _ in range(3))


# --------------------------------------------------------------------------
# Divide-the-dollar and p-beauty (pools not specified in the paper).
# --------------------------------------------------------------------------
class FairSplitter(OpponentPolicy):
    name = "fair_splitter"

    def act(self, *, legal_actions, last_agent_action=None):
        return min(legal_actions, key=lambda x: abs(float(x) - 0.5))


class GreedySplitter(OpponentPolicy):
    name = "greedy_splitter"

    def act(self, *, legal_actions, last_agent_action=None):
        return min(legal_actions, key=lambda x: abs(float(x) - 0.7))


class Level1Guesser(OpponentPolicy):
    """Best response to a uniformly random co-player: 2/3 of 50."""

    name = "level1_guesser"

    def act(self, *, legal_actions, last_agent_action=None):
        return min(legal_actions, key=lambda x: abs(float(x) - 100 / 3))


OPPONENT_CLASSES: dict[str, type[OpponentPolicy]] = {
    cls.name: cls
    for cls in [
        AlwaysCooperate, AlwaysDefect, RandomOpponent, EpsilonGreedy,
        TitForTat, GrimTrigger, Pavlov, TitForTwoTats, GenerousTFT,
        TruthfulBidder, ShadedBidderLow, ShadedBidderHigh, RandomUniformBidder,
        GreedyBargainer, EqualSplitBargainer, NashBargainer, RandomBargainer,
        FairSplitter, GreedySplitter, Level1Guesser,
    ]
}


def make_opponent(name: str) -> OpponentPolicy:
    if name not in OPPONENT_CLASSES:
        raise KeyError(f"Unknown opponent: {name}")
    return OPPONENT_CLASSES[name]()
