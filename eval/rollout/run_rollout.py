#!/usr/bin/env python3
"""Run model/game rollouts and persist lossless episode records as JSONL.

This stage deliberately does not aggregate metrics.  The JSONL records contain
everything needed to compute CR, opponent-axis CR, reasoning/action coupling,
and final-action exploitability without loading a model again.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from eval.rollout.games import TFT_SPLIT, EpisodeResult, opponent_schedule, run_game_batch
from eval.rollout.progress import EvalLogger
from history.paths import active_session_id


SCHEMA_VERSION = 2  # v2: split field, parsed_ok/raw_response per round


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _load_completed(path: Path) -> set[tuple[str, str, int]]:
    """(split, game, episode) keys already persisted; v1 rows count as split=main."""
    completed: set[tuple[str, str, int]] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("status", "completed") == "completed":
                    completed.add((str(row.get("split", "main")), str(row["game"]), int(row["episode"])))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid rollout JSONL {path}:{line_no}: {exc}") from exc
    return completed


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, default=_json_default)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _episode_row(
    *,
    episode: int,
    result: EpisodeResult,
) -> dict[str, Any]:
    """Return only per-episode data; shared metadata lives in _config.json."""
    return {
        "split": result.split,
        "game": result.game,
        "episode": episode,
        "opponent": result.opponent,
        "cumulative_reward": result.cumulative_reward,
        "rounds": [asdict(round_result) for round_result in result.rounds],
    }


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sharded rollout workers write concurrently. Give each atomic replacement
    # its own temporary file so workers cannot rename one another's temp file.
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _resolve_run_id(explicit: str | None) -> str | None:
    run_id = explicit or active_session_id()
    if run_id is None:
        latest = config.RUNS_DIR / "latest.json"
        if latest.is_file():
            try:
                run_id = str(json.loads(latest.read_text(encoding="utf-8"))["run_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return run_id


def _build_agent(args, variant: str, staging_dir: Path | None):
    """Return (agent, tag, model_description)."""
    if args.mode == "heuristic":
        from eval.rollout.agents import HeuristicAgent

        return HeuristicAgent(), "heuristic", "heuristic"
    if args.mode == "litellm":
        from eval.rollout.agents import LiteLLMAgent

        return LiteLLMAgent(args.model_id, max_new_tokens=args.max_tokens), variant, args.model_id
    if args.mode == "ollama":
        from eval.rollout.agents import OllamaAgent

        return OllamaAgent(args.model_id, max_new_tokens=args.max_tokens), variant, args.model_id
    from eval.rollout.agents import LoRALLMAgent

    if args.mode == "hf":
        from eval.rollout.hf_loading import load_causal_lm

        model, tokenizer = load_causal_lm(args.model_id, use_4bit=args.reference_4bit)
        return LoRALLMAgent(model, tokenizer, max_new_tokens=args.max_tokens), variant, args.model_id

    import importlib.util as _ilu  # train/dpo-lora is a hyphenated (non-importable) dir

    spec = _ilu.spec_from_file_location("_dpolora_gpu_env", ROOT / "train" / "dpo-lora" / "gpu_env.py")
    spec.loader.exec_module(_ilu.module_from_spec(spec))  # BNB_CUDA_VERSION / LD paths before bnb
    from eval.table.run_table import build_agent

    eval_args = argparse.Namespace(
        mode="lora",
        STUDENT_MODEL=args.model_id,
        checkpoint_dir=staging_dir,
        no_4bit=not config.USE_4BIT,
        lora_r=config.PAPER_LORA_R,
        lora_alpha=config.PAPER_LORA_ALPHA,
        lora_target=config.PAPER_LORA_TARGET_MODULES,
        max_tokens=args.max_tokens,
    )
    agent, tag = build_agent(eval_args, variant)
    return agent, tag, f"{args.model_id} + {tag}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persist eval rollouts without computing metrics")
    parser.add_argument("--run-id", default=None, help="Session under runs/; default: RUN_ID or runs/latest.json")
    parser.add_argument("--mode", choices=["lora", "hf", "litellm", "ollama", "heuristic"], default="lora",
                        help="lora: student + adapters (default). hf / litellm / ollama: a frontier "
                             "reference row, written as <--name>.jsonl. heuristic: CPU smoke test.")
    parser.add_argument("--name", default=None,
                        help="Row name for non-lora modes, e.g. gemma_27b, llama_70b, haiku")
    parser.add_argument("--variants", default="", help="lora mode: comma-separated; default all available + base")
    parser.add_argument("--games", default="all", help="Comma-separated; default: all 12 games")
    parser.add_argument("--episodes", type=int, default=config.EPISODES_PER_ENV,
                        help="Main split: episodes per game (paper n=12)")
    parser.add_argument("--tft-episodes", type=int, default=None,
                        help="TFT split: episodes per repeated game vs tit-for-tat (default = --episodes; 0 skips)")
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    parser.add_argument("--max-tokens", type=int, default=config.EVAL_MAX_TOKENS)
    parser.add_argument("--model-id", default=None,
                        help="lora: base model (default STUDENT_MODEL); hf: HF id; litellm: provider/model")
    parser.add_argument("--reference-4bit", action="store_true",
                        help="hf mode: NF4 load. Use the same precision the teacher used.")
    parser.add_argument("--resume", action="store_true", help="Skip completed JSONL episode records")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    tft_episodes = args.episodes if args.tft_episodes is None else args.tft_episodes

    run_id = _resolve_run_id(args.run_id)
    if run_id is None:
        parser.error("--run-id is required when RUN_ID and runs/latest.json are unavailable")
    args.run_id = run_id
    session_dir = config.RUNS_DIR / run_id
    rollout_dir = session_dir / "eval" / "rollouts"

    if args.games.lower() == "all":
        games = list(config.ALL_GAMES)
    else:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
        invalid_games = [g for g in games if g not in config.ALL_GAMES]
        if invalid_games:
            parser.error("unknown game(s): " + ", ".join(invalid_games))
    tft_games = [g for g in config.TFT_GAMES if g in games] if tft_episodes > 0 else []

    staging_dir = None
    if args.mode == "lora":
        args.model_id = args.model_id or config.PAPER_STUDENT_MODEL
        lora_dir = session_dir / "lora"
        if not lora_dir.is_dir():
            parser.error(f"LoRA session not found: {lora_dir}")
        from eval.rollout.checkpoints import list_available_variants, prepare_checkpoint_dir
        from eval.table.run_paper_tables import _maybe_merge

        # Each worker gets its own staging directory so concurrent game shards
        # cannot unlink an adapter another worker is loading.
        staging_dir = rollout_dir / "staging" / f"worker-{os.getpid()}"
        available = list_available_variants(lora_dir=lora_dir)
        prepare_checkpoint_dir(staging_dir, lora_dir=lora_dir, variants=available)
        if not args.no_merge:
            _maybe_merge(staging_dir, alpha=config.MERGE_ALPHA)
        available = list_available_variants(lora_dir=staging_dir)
        requested = [v.strip() for v in args.variants.split(",") if v.strip()] or ["base", *available]
        missing = [v for v in requested if v != "base" and v not in available]
        if missing:
            parser.error(
                "requested adapter(s) are incomplete or missing: " + ", ".join(missing)
                + f"; completed adapters: {', '.join(available) or 'none'}"
            )
    else:
        if args.mode in ("hf", "litellm", "ollama") and not args.model_id:
            parser.error(f"--model-id is required for --mode {args.mode}")
        requested = [args.name or args.mode]

    manifest_path = rollout_dir / "_config.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "updated_at": _utc_now(),
        "games": games,
        "episodes_per_game": args.episodes,
        "tft_games": tft_games,
        "tft_episodes_per_game": tft_episodes,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "decoding": {
            "do_sample": config.EVAL_DO_SAMPLE,
            "temperature": config.EVAL_TEMPERATURE,
            "top_p": config.EVAL_TOP_P,
            "reasoning_format": config.EVAL_REASONING_FORMAT,
            "seeding": "torch.manual_seed(per-round seed drawn from the episode RNG)",
        },
        "student_4bit": config.USE_4BIT,
        "merge_alpha": config.MERGE_ALPHA,
        "opponent_schedule": {g: opponent_schedule(g, args.episodes, args.seed) for g in games},
    })
    manifest.setdefault("files", {})
    manifest.setdefault("models", {})
    _write_manifest(manifest_path, manifest)

    for variant in requested:
        output = rollout_dir / f"{variant}.jsonl"
        if output.exists() and not args.resume:
            parser.error(f"rollout already exists: {output}; pass --resume to continue")
        completed = _load_completed(output) if args.resume else set()
        logger = EvalLogger(
            log_path=rollout_dir / "logs" / f"{variant}.log",
            append=args.resume,
            label=f"variant={variant}",
        )
        agent, tag, described = _build_agent(args, variant, staging_dir)
        manifest["models"][variant] = {"mode": args.mode, "model": described,
                                       "reference_4bit": bool(args.reference_4bit)}
        logger.variant_start(variant, tag, "rollout", args.episodes)

        def save_episode(ep_num: int, result: EpisodeResult) -> None:
            _append_jsonl(output, _episode_row(episode=ep_num, result=result))

        plan = [("main", g, args.episodes, None) for g in config.ALL_GAMES if g in games]
        plan += [(TFT_SPLIT, g, tft_episodes, "tit_for_tat") for g in tft_games]
        for idx, (split, game, n, forced_opponent) in enumerate(plan, start=1):
            skip = {ep for s_, g_, ep in completed if s_ == split and g_ == game}
            logger.game_start(idx, len(plan), f"{game} [{split}]", n)
            run_game_batch(
                game, agent, n, args.seed,
                logger=logger, skip_episodes=skip, on_episode=save_episode,
                opponent_name=forced_opponent, split=split,
            )

        completed_now = _load_completed(output)
        manifest["files"][variant] = {
            "path": str(output),
            "completed_episodes": len(completed_now),
            "expected_episodes": len(games) * args.episodes + len(tft_games) * tft_episodes,
        }
        manifest["updated_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)
        del agent
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    manifest["status"] = "completed"
    manifest["updated_at"] = _utc_now()
    _write_manifest(manifest_path, manifest)
    print(f"Saved rollout records under {rollout_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
