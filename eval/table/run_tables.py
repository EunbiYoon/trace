#!/usr/bin/env python3
"""Build table artifacts from one session's per-variant metric JSON."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config
from eval.table.build_latex_md import write_latex_md
from eval.table.run_title import infer_train_epochs
from history.paths import active_session_id, write_latest_pointer


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _paper_refs() -> dict:
    return {f"table{i}": getattr(config, f"PAPER_TABLE{i}") for i in range(1, 8)}


def _training_manifest(lora_dir: Path) -> dict:
    manifest: dict[str, dict] = {}
    if not lora_dir.is_dir():
        return manifest
    for variant_dir in sorted(path for path in lora_dir.iterdir() if path.is_dir()):
        info_path = variant_dir / "run_info.json"
        info = {}
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                info = {}
        manifest[variant_dir.name] = {
            "completed": info.get("status") == "completed",
            "adapter": str(variant_dir / "adapter"),
            "run_info": str(info_path),
        }
    return manifest


def _result_md(suite: dict, variants: list[str]) -> str:
    lines = [
        f"# Evaluation tables — {suite['run_id']}",
        "",
        f"Generated: {suite['finished_at']}",
        "",
        "Pipeline: `rollouts/` → `metrics/` → `tables/`",
        "",
        "## Table 1 — Headline metrics",
        "",
        "| Variant | fc(ID) | fc(HO) | CR(ID) | CR(HO) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        row = suite["table1"][variant]
        cell = lambda key: "—" if row.get(key) is None else f"{row[key]:.3f}"  # noqa: E731
        lines.append(f"| {variant} | {cell('fc_id')} | {cell('fc_ho')} | {cell('cr_id')} | {cell('cr_ho')} |")
    lines += [
        "",
        "## Files",
        "",
        "- `table1.json` … `table7.json`: paper table data",
        "- `suite.json`: combined metrics and metadata",
        "- `paper_refs.json`: fixed paper reference values",
        "- `latex.md`: LaTeX table output",
        "",
    ]
    return "\n".join(lines)


def _fmt(v, signed: bool = False, ci: float | None = None) -> str:
    if v is None:
        return "—"
    text = f"{v:+.2f}" if signed else f"{v:.2f}"
    return f"{text} ± {ci:.2f}" if ci is not None else text


def paper_tables_md(rows: dict, variants: list[str], rollout_manifest: dict) -> str:
    """Tables laid out as in the paper (Table 1-3, Figure 1), with n and 95% CIs."""
    dec = rollout_manifest.get("decoding", {})
    out = [
        "# Paper tables",
        "",
        f"Main split: {rollout_manifest.get('episodes_per_game', '?')} episodes/game, opponents per "
        f"Appendix I pools. TFT split: {rollout_manifest.get('tft_episodes_per_game', '?')} episodes per "
        f"game in {', '.join(rollout_manifest.get('tft_games', []))} vs tit-for-tat. Decoding: "
        f"do_sample={dec.get('do_sample')} T={dec.get('temperature')} format={dec.get('reasoning_format')}.",
        "",
        "## Table 1 — aggregate reward (mean ± 95% CI)",
        "",
        "| Model | CR(ID) | CR(HO) | TFT-CR | parse fail |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in variants:
        r, ci, cnt = rows[v], rows[v].get("ci95", {}), rows[v].get("counts", {})
        known = cnt.get("rounds_with_parse_status") or 0
        pf = f"{cnt.get('parse_failures', 0) / known:.1%}" if known else "n/a"
        out.append(
            f"| {v} | {_fmt(r.get('cr_id'), ci=(ci.get('cr_id') or {}).get('ci95'))} "
            f"| {_fmt(r.get('cr_ho'), True, (ci.get('cr_ho') or {}).get('ci95'))} "
            f"| {_fmt(r.get('tft_cr'), ci=(ci.get('tft_cr') or {}).get('ci95'))} | {pf} |"
        )
    out += ["", "## Table 2 — cumulative reward per game", "",
            "| Game | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
    for group, games in (("ID", config.ID_GAMES), ("HO", config.HO_GAMES)):
        out.append(f"| **{group}** |" + " |" * len(variants))
        for g in games:
            out.append(f"| {g} | " + " | ".join(_fmt(rows[v].get("per_game_cr", {}).get(g)) for v in variants) + " |")
    out += ["", "## Table 3 — reasoning-action coupling", "",
            "| Model | fc(ID) | ac(ID) | fc(HO) | parsable final rounds ID/HO |", "|---|---:|---:|---:|---:|"]
    for v in variants:
        r, cnt = rows[v], rows[v].get("counts", {})
        out.append(f"| {v} | {_fmt(r.get('fc_id'))} | {_fmt(r.get('ac_id'))} | {_fmt(r.get('fc_ho'))} "
                   f"| {cnt.get('fc_id_parsable', '?')}/{cnt.get('fc_ho_parsable', '?')} |")
    buckets = sorted({k for v in variants for k in rows[v].get("per_axis_cr", {})})
    out += ["", "## Figure 1 (left) — CR by opponent class", "",
            "| Model | " + " | ".join(buckets) + " |", "|---|" + "---:|" * len(buckets)]
    for v in variants:
        out.append(f"| {v} | " + " | ".join(_fmt(rows[v].get("per_axis_cr", {}).get(b)) for b in buckets) + " |")
    out += ["", "## Figure 1 (right) — TFT-CR by game", "",
            "| Model | " + " | ".join(config.TFT_GAMES) + " |", "|---|" + "---:|" * len(config.TFT_GAMES)]
    for v in variants:
        out.append(f"| {v} | " + " | ".join(_fmt(rows[v].get("per_game_tft_cr", {}).get(g)) for g in config.TFT_GAMES) + " |")
    out.append("")
    return "\n".join(out)


def _run_id(explicit: str | None) -> str | None:
    if explicit or active_session_id():
        return explicit or active_session_id()
    latest = config.RUNS_DIR / "latest.json"
    if latest.is_file():
        try:
            return str(json.loads(latest.read_text(encoding="utf-8"))["run_id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="metrics/*.json -> tables/*")
    parser.add_argument("--run-id", default=None, help="Session under runs/; default: RUN_ID or latest")
    parser.add_argument("--variants", default="", help="Comma-separated; default: every metric JSON")
    parser.add_argument("--games", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-merge", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    run_id = _run_id(args.run_id)
    if run_id is None:
        parser.error("--run-id is required when RUN_ID and runs/latest.json are unavailable")
    session_dir = config.RUNS_DIR / run_id
    eval_dir = session_dir / "eval"
    metrics_dir = eval_dir / "metrics"
    tables_dir = eval_dir / "tables"
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    paths = [metrics_dir / f"{variant}.json" for variant in requested]
    if not requested:
        paths = sorted(path for path in metrics_dir.glob("*.json") if path.name != "_config.json")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("missing metric file(s): " + ", ".join(str(path) for path in missing))
    if not paths:
        parser.error(f"no metric JSON files found under {metrics_dir}")

    rows = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in paths}
    ok_variants = [variant for variant, row in rows.items() if "error" not in row]
    rollout_manifest_path = eval_dir / "rollouts" / "_config.json"
    rollout_manifest = (
        json.loads(rollout_manifest_path.read_text(encoding="utf-8"))
        if rollout_manifest_path.is_file()
        else {}
    )
    manifest = _training_manifest(session_dir / "lora")
    STUDENT_MODEL = rollout_manifest.get("STUDENT_MODEL") or config.PAPER_STUDENT_MODEL
    hyperparams = {
        "STUDENT_MODEL": STUDENT_MODEL,
        "lora_r": config.PAPER_LORA_R,
        "lora_alpha": config.PAPER_LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "dpo_beta": config.DPO_BETA,
        "merge_alpha": config.MERGE_ALPHA,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_EPOCHS,
        "train_epochs": infer_train_epochs(session_dir / "lora"),
        "episodes_per_env": rollout_manifest.get("episodes_per_game", config.EPISODES_PER_ENV),
        "gradient_accumulation": config.GRADIENT_ACCUMULATION,
        "checkpoint_dir": str(session_dir / "lora"),
        "variants_evaluated": ok_variants,
        "paper_mode": True,
        "use_4bit": config.USE_4BIT,
    }
    suite = {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": 0.0,
        "timings_seconds": {},
        "hyperparams": hyperparams,
        "table1": {v: {k: rows[v].get(k) for k in ("fc_id", "fc_ho", "cr_id", "cr_ho", "tft_cr")} for v in ok_variants},
        "table2": {v: rows[v].get("per_game_cr", {}) for v in ok_variants},
        "table3": {v: rows[v].get("per_axis_cr", {}) for v in ok_variants},
        "table4": {v: rows[v].get("per_game_fc", {}) for v in ok_variants},
        "table5_manifest": manifest,
        "table6": {"paper_reference": config.PAPER_TABLE6},
        "table7": {
            v: {
                "ac": rows[v].get("per_game_ac", {}),
                "fc": rows[v].get("per_game_fc", {}),
                "exploitability": rows[v].get("per_game_exploitability", {}),
            }
            for v in ok_variants
        },
        "variants": ok_variants,
        "rows": rows,
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_json(tables_dir / "suite.json", suite)
    save_json(tables_dir / "paper_refs.json", _paper_refs())
    for number in range(1, 8):
        key = f"table{number}"
        save_json(tables_dir / f"{key}.json", suite.get(key, {}))
    (tables_dir / "result.md").write_text(_result_md(suite, ok_variants), encoding="utf-8")
    (tables_dir / "paper_tables.md").write_text(
        paper_tables_md(rows, ok_variants, rollout_manifest), encoding="utf-8")
    write_latex_md(tables_dir, suite, ok_variants=ok_variants)
    write_latest_pointer(run_id)
    print(f"{len(ok_variants)} variants -> {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
