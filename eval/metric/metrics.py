"""Aggregate metrics for the paper tables.

Definitions (paper §3.1, Tables 1-3, Figure 1):

* CR(ID) / CR(HO): mean cumulative reward over all ``main``-split episodes of
  the in-domain / held-out games. Every game contributes n episodes, so this is
  the mean of per-game means. Reported with a 95% normal-approximation CI.
* TFT-CR: mean cumulative reward over the dedicated ``tft`` split, i.e. n
  episodes per repeated game (config.TFT_GAMES) against tit-for-tat only.
  Never computed from whichever main-split episodes happened to draw TFT.
* Opponent-class CR (Figure 1 left): main-split episodes bucketed by
  ID/HO x stationary/stochastic/strategic.
* fc / ac: final-round / all-round coupling over parsable rounds (Appendix E).
* parse_fail_rate: rounds whose action had to be substituted (parsed_ok False)
  over rounds whose agent reported parse status.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from config import ALL_GAMES, HO_GAMES, ID_GAMES, OPPONENT_AXIS, TFT_GAMES
from eval.metric.coupling import coupling_for_round
from eval.metric.exploitability import exploitability_2x2
from eval.rollout.games import TFT_SPLIT, EpisodeResult, run_game_batch
from eval.rollout.progress import EvalLogger
from eval.rollout.resume import ParsedEpisode


def _mean_ci(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "ci95": None, "n": n}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"mean": mean, "ci95": 1.96 * math.sqrt(var / n), "n": n}


@dataclass
class EvalMetrics:
    fc_id: float | None = None
    fc_ho: float | None = None
    ac_id: float | None = None
    cr_id: float = 0.0
    cr_ho: float = 0.0
    tft_cr: float | None = None
    per_game_cr: dict[str, float | None] = field(default_factory=dict)
    per_axis_cr: dict[str, float] = field(default_factory=dict)
    per_game_fc: dict[str, float | None] = field(default_factory=dict)
    per_game_ac: dict[str, float | None] = field(default_factory=dict)
    per_game_exploitability: dict[str, float | None] = field(default_factory=dict)
    per_game_tft_cr: dict[str, float | None] = field(default_factory=dict)
    per_game_parse_fail: dict[str, float | None] = field(default_factory=dict)
    ci: dict[str, dict] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    episodes: list[EpisodeResult] = field(default_factory=list)


def _axis_for_opponent(name: str) -> str | None:
    for axis, members in OPPONENT_AXIS.items():
        if name in members:
            return axis
    return None


def aggregate_episode_results(episodes: list[EpisodeResult]) -> EvalMetrics:
    """Aggregate persisted rollout episodes without running a model."""
    main = [ep for ep in episodes if ep.split != TFT_SPLIT]
    tft = [ep for ep in episodes if ep.split == TFT_SPLIT]

    per_game: dict[str, list[float]] = defaultdict(list)
    axis_cr: dict[str, list[float]] = defaultdict(list)
    fc_num: dict[str, int] = defaultdict(int)
    fc_den: dict[str, int] = defaultdict(int)
    ac_num: dict[str, int] = defaultdict(int)
    ac_den: dict[str, int] = defaultdict(int)
    exp: dict[str, list[float]] = defaultdict(list)
    pf_num: dict[str, int] = defaultdict(int)
    pf_den: dict[str, int] = defaultdict(int)

    for ep in main:
        game = ep.game
        per_game[game].append(ep.cumulative_reward)
        axis = _axis_for_opponent(ep.opponent)
        if axis:
            axis_cr[f"{'ID' if game in ID_GAMES else 'HO'} {axis.title()}"].append(ep.cumulative_reward)
        for rnd in ep.rounds:
            if rnd.parsed_ok is not None:
                pf_den[game] += 1
                pf_num[game] += int(not rnd.parsed_ok)
            if rnd.reasoning:
                c = coupling_for_round(rnd.reasoning, str(rnd.agent_action))
                if c.parsable:
                    ac_den[game] += 1
                    ac_num[game] += int(c.coupled)
        final = ep.final_round
        if final and final.reasoning:
            c = coupling_for_round(final.reasoning, str(final.agent_action))
            if c.parsable:
                fc_den[game] += 1
                fc_num[game] += int(c.coupled)
        if final:
            e = exploitability_2x2(game, str(final.agent_action))
            if e is not None:
                exp[game].append(e)

    tft_game: dict[str, list[float]] = defaultdict(list)
    for ep in tft:
        tft_game[ep.game].append(ep.cumulative_reward)

    def rate(num, den, games) -> float | None:
        d = sum(den[g] for g in games)
        return sum(num[g] for g in games) / d if d else None

    cr_id = [v for g in ID_GAMES for v in per_game[g]]
    cr_ho = [v for g in HO_GAMES for v in per_game[g]]
    tft_all = [v for g in TFT_GAMES for v in tft_game[g]]
    ci = {
        "cr_id": _mean_ci(cr_id),
        "cr_ho": _mean_ci(cr_ho),
        "tft_cr": _mean_ci(tft_all),
        **{f"cr:{g}": _mean_ci(per_game[g]) for g in ALL_GAMES},
        **{f"tft:{g}": _mean_ci(tft_game[g]) for g in TFT_GAMES},
    }
    return EvalMetrics(
        fc_id=rate(fc_num, fc_den, ID_GAMES),  # None = no parsable round
        fc_ho=rate(fc_num, fc_den, HO_GAMES),
        ac_id=rate(ac_num, ac_den, ID_GAMES),
        cr_id=ci["cr_id"]["mean"] or 0.0,
        cr_ho=ci["cr_ho"]["mean"] or 0.0,
        tft_cr=ci["tft_cr"]["mean"],
        per_game_cr={g: ci[f"cr:{g}"]["mean"] for g in ALL_GAMES},
        per_axis_cr={k: sum(v) / len(v) for k, v in sorted(axis_cr.items())},
        per_game_fc={g: (fc_num[g] / fc_den[g] if fc_den[g] else None) for g in ALL_GAMES},
        per_game_ac={g: (ac_num[g] / ac_den[g] if ac_den[g] else None) for g in ALL_GAMES},
        per_game_exploitability={g: (sum(exp[g]) / len(exp[g]) if exp[g] else None) for g in ALL_GAMES},
        per_game_tft_cr={g: ci[f"tft:{g}"]["mean"] for g in TFT_GAMES},
        per_game_parse_fail={g: (pf_num[g] / pf_den[g] if pf_den[g] else None) for g in ALL_GAMES},
        ci=ci,
        counts={
            "main_episodes": len(main),
            "tft_episodes": len(tft),
            "fc_id_parsable": sum(fc_den[g] for g in ID_GAMES),
            "fc_ho_parsable": sum(fc_den[g] for g in HO_GAMES),
            "ac_id_parsable": sum(ac_den[g] for g in ID_GAMES),
            "rounds_with_parse_status": sum(pf_den.values()),
            "parse_failures": sum(pf_num.values()),
        },
        episodes=episodes,
    )


def evaluate_agent(
    agent,
    n_episodes: int = 12,
    seed_base: int = 0,
    logger: EvalLogger | None = None,
    games: list[str] | None = None,
    prior_episodes: dict[str, list[ParsedEpisode]] | None = None,
    tft_episodes: int | None = None,
) -> EvalMetrics:
    """Run the main split (and the TFT split) live and aggregate.

    ``prior_episodes`` (parsed from legacy logs) carry only rewards, so they
    are folded into CR but cannot contribute coupling or parse statistics.
    """
    games_to_run = set(games) if games else set(ALL_GAMES)
    prior = prior_episodes or {}
    episodes: list[EpisodeResult] = []
    for gi, game in enumerate(ALL_GAMES, start=1):
        prior_list = prior.get(game, [])
        skip = {p.episode for p in prior_list}
        episodes.extend(EpisodeResult(game, p.opponent, p.cr, []) for p in prior_list)
        if game not in games_to_run or len(skip) >= n_episodes:
            continue
        if logger:
            logger.game_start(gi, len(ALL_GAMES), game, n_episodes)
        episodes.extend(run_game_batch(game, agent, n_episodes, seed_base, logger=logger, skip_episodes=skip))
    n_tft = n_episodes if tft_episodes is None else tft_episodes
    for game in TFT_GAMES:
        if game in games_to_run and n_tft > 0:
            episodes.extend(run_game_batch(
                game, agent, n_tft, seed_base, logger=logger,
                opponent_name="tit_for_tat", split=TFT_SPLIT,
            ))
    return aggregate_episode_results(episodes)


def metrics_to_dict(m: EvalMetrics) -> dict:
    return {
        "fc_id": m.fc_id,
        "fc_ho": m.fc_ho,
        "ac_id": m.ac_id,
        "cr_id": m.cr_id,
        "cr_ho": m.cr_ho,
        "tft_cr": m.tft_cr,
        "per_game_cr": m.per_game_cr,
        "per_axis_cr": m.per_axis_cr,
        "per_game_fc": dict(m.per_game_fc),
        "per_game_ac": dict(m.per_game_ac),
        "per_game_exploitability": dict(m.per_game_exploitability),
        "per_game_tft_cr": dict(m.per_game_tft_cr),
        "per_game_parse_fail": dict(m.per_game_parse_fail),
        "ci95": m.ci,
        "counts": m.counts,
    }
