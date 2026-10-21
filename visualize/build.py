#!/usr/bin/env python3
"""Build the TRACE project page (index.html) from template.html.

Usage
-----
# Placeholder build (teacher blind-rollout episode, clearly labelled):
python build.py

# Real build from an evaluation rollout of the TRACE-RW checkpoint:
python build.py --rollout runs/<RUN_ID>/eval/rollout/rw.jsonl --variant rw \
                --games pd-classic,stag-hunt,auction

Every episode of the selected games is embedded (no cherry-picking); the
replay opens on the first one, ordered by game then episode number.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Empty while under double-blind review: the submission PDF says "Do not
# distribute", so the Paper button is omitted until there is a public link
# (OpenReview forum or arXiv).
PAPER_URL = ""
# Empty = no BibTeX section. Paste the OpenReview / arXiv entry once public.
BIBTEX = ""
CODE_URL = "https://github.com/EunbiYoon/trace"
MODEL_URL = "https://huggingface.co/Bianca2/trace/tree/main"

VARIANT_LABEL = {
    "base": "Qwen 2.5–3B base", "core": "Qwen 2.5–3B + TRACE-CORE", "aux": "Qwen 2.5–3B + TRACE-AUX",
    "all": "Qwen 2.5–3B + TRACE-ALL", "rw": "Qwen 2.5–3B + TRACE-RW", "merge": "Qwen 2.5–3B + TRACE-MERGE",
}

# ---- Paper numbers (TRACE paper, Tables 1-3, 5) ----
TABLE1 = [  # model, CR(ID), CR(HO), TFT-CR, kind
    ("Qwen 2.5–3B base", 14.04, -1.45, 1.50, ""),
    ("Qwen 2.5–3B + frontier DPO", 13.51, -0.23, 1.50, ""),
    ("Qwen 2.5–3B + TRACE-ALL", 14.34, -0.27, 15.31, ""),
    ("Qwen 2.5–3B + TRACE-RW", 13.93, 1.91, 14.06, "hl"),
    ("Qwen 2.5–3B + TRACE-MERGE", 13.92, 1.75, None, ""),
    ("Haiku 4.5", 14.60, 2.05, 1.50, "ref"),
    ("Gemma 3–27B", 13.36, 0.28, 1.50, "ref"),
    ("Llama 3.3–70B", 12.29, 1.52, 1.50, "ref"),
]
T2_COLS = ["base", "core", "aux", "all", "rw", "merge", "haiku"]
TABLE2 = {
    "In-domain (ID)": {
        "pd-classic": [21.96, 22.08, 23.08, 23.88, 21.67, 23.50, 25.68],
        "pd-tight": [25.92, 27.08, 26.21, 26.21, 25.42, 25.92, 27.25],
        "pd-high-temptation": [29.33, 28.50, 29.17, 30.42, 28.42, 29.17, 27.92],
        "stag-hunt": [28.62, 28.71, 28.38, 30.54, 30.42, 27.88, 27.79],
        "negotiation": [5.75, 3.17, 3.46, 3.08, 4.67, 4.04, 7.54],
        "bos": [0.75, 0.75, 0.62, 0.62, 0.88, 0.88, 0.62],
        "matching-pennies": [0.00] * 7,
        "tic-tac-toe": [0.00] * 7,
    },
    "Held-out (HO)": {
        "auction": [-8.67, -0.71, -0.10, -3.96, 4.97, 4.18, 5.82],
        "divide-dollar": [0.27, 0.11, 0.22, 0.34, 0.32, 0.21, 0.25],
        "p-beauty": [0.50] * 7,
        "ipd-stage": [2.08, 1.88, 2.08, 2.04, 1.83, 2.12, 1.62],
    },
}
TABLE3 = [  # model, fc(ID), ac(ID), fc(HO), is_3b
    ("Qwen 2.5–3B + TRACE-CORE", 0.61, 0.74, 0.00, True),
    ("Qwen 2.5–3B + TRACE-AUX", 0.56, 0.72, 0.29, True),
    ("Qwen 2.5–3B + TRACE-ALL", 0.59, 0.65, 0.29, True),
    ("Qwen 2.5–3B + TRACE-RW", 0.68, 0.70, 0.46, True),
    ("Qwen 2.5–3B + TRACE-MERGE", 0.69, 0.73, 0.50, True),
    ("Haiku 4.5", 0.85, 0.57, 0.79, False),
    ("Gemma 3–27B", 0.74, 0.80, 0.43, False),
    ("Llama 3.3–70B", 0.69, 0.52, 0.30, False),
]
TABLE5 = [  # variant, pairs, best step, best eval loss
    ("TRACE-CORE", 503, 50, 0.2973), ("TRACE-AUX", 613, 80, 0.3128),
    ("TRACE-ALL", 1338, 40, 0.2278), ("TRACE-RW", 1749, 70, 0.2454),
    ("TRACE-MERGE", None, None, None),
]
TFT_BARS = [
    {"name": "TRACE-RW (3B)", "v": 14.06, "us": True},
    {"name": "TRACE-ALL (3B)", "v": 15.31, "us": False},
    {"name": "Qwen 2.5–3B base", "v": 1.50, "us": False},
    {"name": "Frontier DPO (3B)", "v": 1.50, "us": False},
    {"name": "Haiku 4.5", "v": 1.50, "us": False},
    {"name": "Gemma 3–27B", "v": 1.50, "us": False},
    {"name": "Llama 3.3–70B", "v": 1.50, "us": False},
]


def f2(v, signed=False):
    if v is None:
        return "—"
    s = f"{v:+.2f}" if signed else f"{v:.2f}"
    return s.replace("-", "−")


def table1() -> str:
    rows = []
    for name, cid, cho, tft, kind in TABLE1:
        rows.append(
            f'<tr class="{kind}"><td>{name}</td><td>{f2(cid)}</td>'
            f'<td class="{"neg" if cho < 0 else ""}">{f2(cho, True)}</td><td>{f2(tft)}</td></tr>')
    return ('<div class="table-container"><table><thead><tr><th>Model</th><th>CR (ID)</th><th>CR (HO)</th>'
            '<th>TFT-CR</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def table2() -> str:
    head = "".join(f"<th>{c if c != 'haiku' else 'Haiku 4.5'}</th>" for c in T2_COLS)
    body = []
    for group, games in TABLE2.items():
        body.append(f'<tr class="section-row"><td colspan="{len(T2_COLS) + 1}">{group}</td></tr>')
        for game, vals in games.items():
            three_b = vals[:-1]
            best = max(three_b)
            tie = len(set(three_b)) == 1
            cells = []
            for i, v in enumerate(vals):
                cls = []
                if i < len(vals) - 1 and not tie and v == best:
                    cls.append("best")
                if v < 0:
                    cls.append("neg")
                cells.append(f'<td class="{" ".join(cls)}">{f2(v)}</td>')
            body.append(f"<tr><td>{game}</td>{''.join(cells)}</tr>")
    return ('<div class="table-container"><table><thead><tr><th>Game</th>' + head +
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def table3() -> str:
    best = [max(r[i] for r in TABLE3 if r[4]) for i in (1, 2, 3)]
    rows = []
    for name, *vals, is3b in TABLE3:
        cells = "".join(
            f'<td class="{"best" if is3b and v == best[i] else ""}">{v:.2f}</td>' for i, v in enumerate(vals))
        rows.append(f'<tr class="{"" if is3b else "ref"}"><td>{name}</td>{cells}</tr>')
    return ('<div class="table-container"><table><thead><tr><th>Model</th><th>fc (ID)</th><th>ac (ID)</th>'
            '<th>fc (HO)</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def table5() -> str:
    rows = "".join(
        f"<tr><td>{n}</td><td>{p if p else '— (LoRA average of AUX + ALL)'}</td><td>{s or '—'}</td>"
        f"<td>{f'{l:.4f}' if l else '—'}</td></tr>" for n, p, s, l in TABLE5)
    return ('<div class="table-container"><table><thead><tr><th>Variant</th><th>DPO pairs</th>'
            '<th>Best ckpt step</th><th>Best eval loss</th></tr></thead><tbody>' + rows + "</tbody></table></div>")


def replay_data(args) -> dict:
    if not args.rollout:
        ep = json.loads((HERE / "demo_episode.json").read_text(encoding="utf-8"))
        return {
            "meta": {
                "placeholder": True,
                "model": "Qwen 2.5–7B-Instruct teacher (blind rollout)",
                "agent_label": "Teacher (blind rollout)",
                "source": "data/blind-rollout/result/a_beta_rw.jsonl · pd-classic-tit_for_tat-s142-1",
                "note": ("<strong>Placeholder — not the TRACE checkpoint.</strong> This is the teacher's own play against "
                         "tit-for-tat: it defects in round 1, triggers retaliation, and never re-establishes cooperation "
                         "(mutual cooperation would earn 30). Rebuild with <code>python visualize/build.py --rollout …/rw.jsonl</code> "
                         "after evaluating TRACE-RW."),
            },
            "episodes": [ep],
        }
    games = [g.strip() for g in args.games.split(",") if g.strip()] if args.games else None
    eps = []
    for line in Path(args.rollout).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("rounds"):
            continue
        if games and row["game"] not in games:
            continue
        eps.append(row)
    if not eps:
        raise SystemExit(f"no episodes found in {args.rollout} for games={games}")
    order = {g: i for i, g in enumerate(games)} if games else {}
    eps.sort(key=lambda r: (order.get(r["game"], 99), r["game"], r.get("episode", 0)))
    label = VARIANT_LABEL.get(args.variant, args.variant)
    return {
        "meta": {
            "placeholder": False,
            "model": f"{label} · checkpoint {MODEL_URL}",
            "agent_label": label,
            "source": f"{Path(args.rollout).name} · {len(eps)} episodes, all episodes of the selected games",
            "note": "Actions are logged after the evaluator's action parser; unparsable outputs fall back to a default legal action.",
        },
        "episodes": eps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", help="evaluation rollout .jsonl (rows with game/episode/opponent/rounds)")
    ap.add_argument("--variant", default="rw")
    ap.add_argument("--games", default="pd-classic,stag-hunt,pd-tight,pd-high-temptation,auction",
                    help="comma-separated games to embed (order = dropdown order)")
    ap.add_argument("--out", default=str(HERE / "index.html"))
    args = ap.parse_args()

    data = json.dumps(replay_data(args), ensure_ascii=False).replace("</", "<\\/")
    page = (HERE / "template.html").read_text(encoding="utf-8")
    if not PAPER_URL:
        page = re.sub(r'\s*<a class="btn" href="__PAPER_URL__"[^\n]*</a>', "", page)
    if not BIBTEX:
        page = re.sub(r'\s*<a href="#bibtex">BibTeX</a>', "", page)
        page = re.sub(r'\n<section id="bibtex">.*?</section>\n', "\n", page, flags=re.S)
        page = re.sub(r"\n  document\.getElementById\('copyBib'\).*?\n  \}\);\n", "\n", page, flags=re.S)
    else:
        page = re.sub(r'(<pre id="bib">).*?(</pre>)',
                      lambda m: m.group(1) + html.escape(BIBTEX.strip()) + m.group(2), page, flags=re.S)
    for key, val in {
        "__PAPER_URL__": html.escape(PAPER_URL), "__CODE_URL__": html.escape(CODE_URL),
        "__MODEL_URL__": html.escape(MODEL_URL), "__TABLE1__": table1(), "__TABLE2__": table2(),
        "__TABLE3__": table3(), "__TABLE5__": table5(), "__TFT_BARS__": json.dumps(TFT_BARS),
        "__REPLAY_DATA__": data,
    }.items():
        if key == "__PAPER_URL__" and not PAPER_URL:
            continue  # button removed above
        assert key in page, key
        page = page.replace(key, val)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
