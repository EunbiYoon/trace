"""12-game rollout engine for Tables 1-7."""

from __future__ import annotations

import random
import json
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

from config import ALL_GAMES, EVAL_OPPONENT_POOLS, OPPONENT_AXIS, PD_VARIANTS
import config
from eval.rollout.opponents import make_opponent
from eval.rollout.prompts import (
    auction_prompt,
    bos_payoff,
    bos_prompt,
    divide_dollar_prompt,
    ipd_stage_prompt,
    matching_pennies_prompt,
    mp_payoff,
    negotiation_prompt,
    p_beauty_prompt,
    pd_payoff,
    repeated_pd_prompt,
    stag_hunt_prompt,
    stag_payoff,
    tic_tac_toe_prompt,
)
from eval.rollout.progress import EvalLogger, ProgressAgent

TFT_SPLIT = "tft"

ALL_OPPONENTS: list[str] = [
    name for members in OPPONENT_AXIS.values() for name in members
]

DEFAULT_OPPONENT: dict[str, str] = {
    "pd-classic": "tit_for_tat",
    "pd-tight": "tit_for_tat",
    "pd-high-temptation": "tit_for_tat",
    "stag-hunt": "tit_for_tat",
    "negotiation": "random",
    "bos": "random",
    "matching-pennies": "random",
    "tic-tac-toe": "random",
    "auction": "truthful_bidder",
    "divide-dollar": "random",
    "p-beauty": "random",
    "ipd-stage": "tit_for_tat",
}


@dataclass
class RoundResult:
    reasoning: str
    agent_action: Any
    opponent_action: Any = None
    prompt: str | None = None
    # False when the model's output had no parsable legal action and the
    # engine substituted a default; None for agents that do not report it.
    parsed_ok: bool | None = None
    raw_response: str | None = None


@dataclass
class EpisodeResult:
    game: str
    opponent: str
    cumulative_reward: float
    rounds: list[RoundResult] = field(default_factory=list)
    split: str = "main"  # "main" = Table 1/2/3 pool; "tft" = TFT-CR split

    @property
    def final_round(self) -> RoundResult | None:
        return self.rounds[-1] if self.rounds else None

    def to_training_trajectory(
        self, *, trajectory_id: str, game_spec: dict[str, Any], frontier_model: str
    ) -> dict[str, Any]:
        """Convert a lossless blind rollout to Algorithm 1's trajectory schema."""
        if not self.rounds or any(r.prompt is None for r in self.rounds):
            raise ValueError("training trajectories require a prompt for every round")
        if any(r.opponent_action is None for r in self.rounds):
            raise ValueError("training trajectories require an opponent action for every round")
        def action_text(action: Any) -> str:
            if isinstance(action, (list, tuple)):
                return json.dumps(list(action), separators=(",", ":"))
            return str(action)

        completions = [
            f"<think>{r.reasoning}</think><action>{action_text(r.agent_action)}</action>"
            for r in self.rounds
        ]
        return {
            "id": trajectory_id,
            "game": self.game,
            "opponent": self.opponent,
            "frontier_model": frontier_model,
            "round_prompts": [r.prompt for r in self.rounds],
            "round_completions": completions,
            "own_actions": [r.agent_action for r in self.rounds],
            "opponent_actions": [r.opponent_action for r in self.rounds],
            **game_spec,
        }


def opponent_schedule(game: str, n_episodes: int, seed_base: int = 0) -> list[str]:
    """Opponents for episodes 0..n-1 (paper Appendix I).

    ``weighted``: i.i.d. draws by pool weight. ``round_robin``: the first
    |pool| episodes cover every member once (shuffled), the rest are weighted.
    The schedule depends only on (game, n, seed_base), so it is identical for
    every evaluated model.
    """
    if game not in EVAL_OPPONENT_POOLS:
        return [DEFAULT_OPPONENT.get(game, "random")] * n_episodes
    mode, pool = EVAL_OPPONENT_POOLS[game]
    names, weights = list(pool), list(pool.values())
    rng = random.Random(stable_hash(f"opponents|{game}|{seed_base}"))
    schedule: list[str] = []
    if mode == "round_robin":
        warmup = names[:]
        rng.shuffle(warmup)
        schedule.extend(warmup[:n_episodes])
    while len(schedule) < n_episodes:
        schedule.append(rng.choices(names, weights=weights, k=1)[0])
    return schedule


def _decide(agent, obs: dict[str, Any], legal: list[Any]) -> tuple[str, Any, bool | None, str | None]:
    """Query the agent and validate the action against ``legal``.

    Invalid actions are replaced by ``legal[0]`` exactly as before, but the
    substitution is now recorded as ``parsed_ok=False`` instead of being silent.
    """
    reasoning, action = agent.decide(obs)
    meta = getattr(agent, "last_meta", None) or {}
    parsed_ok = meta.get("parsed_ok")
    raw = meta.get("raw_response")
    if action not in legal:
        action = legal[0]
        parsed_ok = False
    return reasoning, action, parsed_ok, raw


def _wrap_agent(agent, opponent: str, logger: EvalLogger | None):
    if logger is None:
        return agent
    return ProgressAgent(agent, logger, opponent=opponent)


def _ttt_terminal(board: list[str | None], player: str) -> int | None:
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] == board[b] == board[c] == player:
            return 1 if player == "X" else -1
    if all(v is not None for v in board):
        return 0
    return None


def _run_ttt(agent, opponent_name: str, rng: random.Random, logger: EvalLogger | None) -> EpisodeResult:
    board: list[str | None] = [None] * 9
    rounds: list[RoundResult] = []
    reward = 0.0
    opp = make_opponent(opponent_name)
    opp.reset(rng, list(range(9)))
    current = "X"
    last_agent_action = None

    for step in range(9):
        legal = [i for i, v in enumerate(board) if v is None]
        if not legal:
            break
        if current == "X":
            prompt = tic_tac_toe_prompt(board)
            obs = {
                "game": "tic-tac-toe",
                "round": step,
                "horizon": 9,
                "prompt": prompt,
                "legal_actions": legal,
            }
            reasoning, action, ok, raw = _decide(agent, {**obs, "seed": rng.randrange(2**31)}, legal)
            board[action] = "X"
            last_agent_action = action
            rounds.append(RoundResult(reasoning, action, prompt=prompt, parsed_ok=ok, raw_response=raw))
            status = _ttt_terminal(board, "X")
            if status is not None:
                reward = float(status)
                break
            current = "O"
        else:
            o_action = opp.act(legal_actions=legal, last_agent_action=last_agent_action)
            board[o_action] = "O"
            opp.observe(last_agent_action, o_action)
            status = _ttt_terminal(board, "O")
            if status is not None:
                reward = float(-status)
                break
            current = "X"

    return EpisodeResult("tic-tac-toe", opponent_name, reward, rounds)


def _run_repeated_matrix(
    game: str,
    agent,
    opponent_name: str,
    rng: random.Random,
    *,
    horizon: int,
    legal: list[str],
    prompt_fn,
    payoff_fn,
) -> EpisodeResult:
    history: list[dict] = []
    rounds: list[RoundResult] = []
    total = 0.0
    opp = make_opponent(opponent_name)
    opp.reset(rng, legal)
    last_agent_action = None

    for rnd in range(horizon):
        prompt = prompt_fn(round_idx=rnd + 1, horizon=horizon, history=history)
        obs = {
            "game": game,
            "round": rnd,
            "horizon": horizon,
            "prompt": prompt,
            "legal_actions": legal,
        }
        reasoning, action, ok, raw = _decide(agent, {**obs, "seed": rng.randrange(2**31)}, legal)
        opp_action = opp.act(legal_actions=legal, last_agent_action=last_agent_action)
        r0, r1 = payoff_fn(action, opp_action)
        total += r0
        history.append({"actions": {"p0": action, "p1": opp_action}, "rewards": {"p0": r0, "p1": r1}})
        rounds.append(RoundResult(reasoning, action, opp_action, prompt, ok, raw))
        opp.observe(action, opp_action)
        last_agent_action = action

    return EpisodeResult(game, opponent_name, total, rounds)


def _run_one_shot(
    game: str,
    agent,
    opponent_name: str,
    rng: random.Random,
    *,
    prompt: str,
    legal: list[Any],
    payoff_fn,
    extra: dict | None = None,
) -> EpisodeResult:
    opp = make_opponent(opponent_name)
    opp.reset(rng, legal)
    obs = {
        "game": game,
        "round": 0,
        "horizon": 1,
        "prompt": prompt,
        "legal_actions": legal,
        **(extra or {}),
    }
    obs["seed"] = rng.randrange(2**31)
    if game == "negotiation":
        # Proposals are tuples; the parser already coerces to a valid triple.
        reasoning, action = agent.decide(obs)
        meta = getattr(agent, "last_meta", None) or {}
        ok, raw = meta.get("parsed_ok"), meta.get("raw_response")
        if tuple(action) not in legal:
            action, ok = legal[0], False
        action = tuple(action)
    else:
        reasoning, action, ok, raw = _decide(agent, obs, legal)
    opp_action = opp.act(legal_actions=legal, last_agent_action=None)
    reward, _ = payoff_fn(action, opp_action)
    return EpisodeResult(
        game,
        opponent_name,
        float(reward),
        [RoundResult(reasoning, action, opp_action, prompt, ok, raw)],
    )


def _run_episode(game: str, agent, opponent_name: str, rng: random.Random, logger: EvalLogger | None) -> EpisodeResult:
    agent = _wrap_agent(agent, opponent_name, logger)

    if game in PD_VARIANTS and game != "ipd-stage":
        horizon = PD_VARIANTS[game].horizon
        return _run_repeated_matrix(
            game,
            agent,
            opponent_name,
            rng,
            horizon=horizon,
            legal=["C", "D"],
            prompt_fn=lambda **kw: repeated_pd_prompt(game, **kw),
            payoff_fn=lambda a0, a1: pd_payoff(game, a0, a1),
        )

    if game == "stag-hunt":
        return _run_repeated_matrix(
            game,
            agent,
            opponent_name,
            rng,
            horizon=10,
            legal=["Stag", "Hare"],
            prompt_fn=stag_hunt_prompt,
            payoff_fn=stag_payoff,
        )

    if game == "bos":
        return _run_one_shot(
            game,
            agent,
            opponent_name,
            rng,
            prompt=bos_prompt(),
            legal=["Opera", "Football"],
            payoff_fn=bos_payoff,
        )

    if game == "matching-pennies":
        return _run_one_shot(
            game,
            agent,
            opponent_name,
            rng,
            prompt=matching_pennies_prompt(),
            legal=["H", "T"],
            payoff_fn=mp_payoff,
        )

    if game == "negotiation":
        proposals = [(a, b, c) for a in range(5) for b in range(5) for c in range(5)]

        def neg_payoff(action, opp_action):
            a0 = tuple(action) if isinstance(action, (list, tuple)) else (1, 1, 1)
            a1 = tuple(opp_action) if isinstance(opp_action, (list, tuple)) else (1, 1, 1)
            cap = config.BARGAIN_CAPACITY
            if all(a0[i] + a1[i] <= cap for i in range(3)):
                return float(3 * a0[0] + 4 * a0[1] + 3 * a0[2]), 0.0
            return 0.0, 0.0

        return _run_one_shot(
            game,
            agent,
            opponent_name,
            rng,
            prompt=negotiation_prompt(),
            legal=proposals,
            payoff_fn=neg_payoff,
        )

    if game == "tic-tac-toe":
        return _run_ttt(agent, opponent_name, rng, logger)

    if game == "auction":
        value = rng.randint(50, 150)
        bids = list(range(0, 201))

        def auction_payoff(action, opp_action):
            b0 = int(action)
            b1 = int(opp_action)
            if b0 > b1:
                return float(value - b0), 0.0
            return 0.0, 0.0

        return _run_one_shot(
            game,
            agent,
            opponent_name,
            rng,
            prompt=auction_prompt(value=value),
            legal=bids,
            payoff_fn=auction_payoff,
            extra={"private_value": value},
        )

    if game == "divide-dollar":
        shares = [i / 100 for i in range(101)]

        def dd_payoff(action, opp_action):
            s0 = float(action)
            s1 = float(opp_action)
            if s0 + s1 <= 1.0:
                return s0, s1
            return 0.0, 0.0

        return _run_one_shot(
            game,
            agent,
            opponent_name,
            rng,
            prompt=divide_dollar_prompt(),
            legal=shares,
            payoff_fn=dd_payoff,
        )

    if game == "p-beauty":
        guesses = list(range(101))

        def pb_payoff(action, opp_action):
            # Win indicator on the paper's 0-1 scale (Table 2 reports 0.50 for
            # every model): closer to 2/3 of the mean wins 1, a tie gets 0.5.
            # The previous 100-minus-distance reward (0-100) dominated CR(HO).
            target = (2 / 3) * (int(action) + int(opp_action)) / 2
            d0, d1 = abs(int(action) - target), abs(int(opp_action) - target)
            if d0 < d1:
                return 1.0, 0.0
            if d0 > d1:
                return 0.0, 1.0
            return 0.5, 0.5

        return _run_one_shot(
            game,
            agent,
            opponent_name,
            rng,
            prompt=p_beauty_prompt(),
            legal=guesses,
            payoff_fn=pb_payoff,
        )

    if game == "ipd-stage":
        return _run_one_shot(
            game,
            agent,
            opponent_name,
            rng,
            prompt=ipd_stage_prompt(),
            legal=["C", "D"],
            payoff_fn=lambda a0, a1: pd_payoff("ipd-stage", a0, a1),
        )

    raise KeyError(f"Unknown game: {game}")


def run_game_batch(
    game: str,
    agent,
    n_episodes: int,
    seed_base: int = 0,
    logger: EvalLogger | None = None,
    skip_episodes: set[int] | None = None,
    on_episode: Callable[[int, EpisodeResult], None] | None = None,
    opponent_name: str | None = None,
    split: str = "main",
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    skip = skip_episodes or set()
    schedule = opponent_schedule(game, n_episodes, seed_base)
    for ep in range(n_episodes):
        ep_num = ep + 1
        if ep_num in skip:
            continue
        opponent = opponent_name or schedule[ep]
        if logger:
            logger.episode_start(game, ep_num, n_episodes, opponent)
        rng = random.Random(episode_seed(game, ep, seed_base, split))
        ep_result = _run_episode(game, agent, opponent, rng, logger)
        ep_result.split = split
        if logger:
            logger.episode_done(game, ep_num, ep_result.cumulative_reward, len(ep_result.rounds))
        if on_episode is not None:
            on_episode(ep_num, ep_result)
        results.append(ep_result)
    return results


def stable_hash(text: str) -> int:
    """Process-independent hash (Python's ``hash`` is salted per process)."""
    return zlib.crc32(text.encode("utf-8"))


def episode_seed(game: str, episode_idx: int, seed_base: int = 0, split: str = "main") -> int:
    """Per-episode seed for the environment, the opponent and LLM sampling."""
    return seed_base + episode_idx * 997 + stable_hash(f"{split}|{game}")
