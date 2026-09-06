"""Guard the BATTLE build: the one thing we add must survive a real rerun.

Why this file exists. The lab kernel put the turn-budget hint inside
`if not true_submission:`, next to the own-games swap. Phase A printed
"BATCH HINT added" and the numbers were real -- but a competition rerun takes
the other branch, so a submission from that kernel would have shipped plain
stock while every log said otherwise. Nothing would have failed; the score
would just have come back as stock's and we would have drawn a wrong
conclusion about the hint from it.

That is the same shape as V47, where the A* loader sat BELOW the ablation
block and quietly undid it. Both are invisible to a green run, so both need a
guard rather than a careful reading.

usage:
    python scripts/test_stock_battle.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "notebooks_stockbattle") / "submission.ipynb"   # optional: dir name to check
META = ROOT / "notebooks_stockbattle" / "kernel-metadata.json"
DUCK = ROOT / "notebooks_duck" / "submission.ipynb"

fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  {'ok  ' if ok else 'СБОЙ'}  {msg}")
    if not ok:
        fails.append(msg)


def main() -> int:
    if not NB.exists():
        sys.exit(f"нет сборки: {NB} (собрать: ATLAS_STOCKLAB_BATTLE=1 python scripts/build_stocklab_notebook.py)")

    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    srcs = ["".join(c["source"]) for c in cells]
    i_bat = next(i for i, s in enumerate(srcs) if "BATTLE build" in s)
    bat = srcs[i_bat]

    print("боевая сборка -- проверки:")

    # 1. THE point of the file: the hint must not sit behind a submission gate.
    i_hint = bat.index("PYTHON_ADDENDUM = ")
    i_guard = bat.index("if not true_submission:") if "if not true_submission:" in bat else len(bat)
    check(i_hint < i_guard, "подсказка применяется ДО гейта true_submission")

    # 2. A submission must not depend on our own games in any way.
    check("our_games" not in bat, "боевая ячейка не трогает наши игры")
    mounts = json.loads(META.read_text(encoding="utf-8"))["dataset_sources"]
    check(not any("atlas" in d for d in mounts), f"наш датасет не смонтирован: {mounts}")

    # 3. The hint is applied by rebinding tool_agent.PYTHON_ADDENDUM, which
    #    _build_system_prompt() reads as a module global at prompt-build time
    #    (tool_agent.py:1024, inside the function). So the patch only has to
    #    beat the RUN, not the solver's construction -- but it does have to
    #    beat the run, and an earlier draft of this check looked for
    #    `_competition_games()` and found its DEFINITION cell instead, which
    #    reported a false failure. Anchor on the cell that actually runs.
    i_run = next(i for i, s in enumerate(srcs)
                 if "bm.games = _competition_games()" in s or "bm.run" in s)
    check(i_bat < i_run, f"наша ячейка ({i_bat}) раньше запуска прогона ({i_run})")

    # 4. Everything else is stock, byte for byte.
    orig = json.loads(DUCK.read_text(encoding="utf-8"))["cells"]
    built = [c for i, c in enumerate(cells) if i != i_bat]
    same = len(built) == len(orig) and all(
        "".join(a["source"]) == "".join(b["source"]) for a, b in zip(orig, built))
    check(same, "остальные ячейки побайтово равны стоковым")

    # 5. Nothing ships that cannot parse.
    bad = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            compile(srcs[i], f"<cell {i}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            bad.append(f"{i}: {e}")
    check(not bad, f"все ячейки компилируются{'' if not bad else ': ' + '; '.join(bad)}")

    # 6. The hint itself must actually be in there -- an empty patch would pass
    #    every check above and change nothing.
    check("TURN BUDGET IS SMALL" in bat or "NEW-LEVEL PROTOCOL" in bat or "EVENT PROTOCOL" in bat,
          "текст подсказки на месте (batch / observe / protocol / lean)")
    if "EVENT PROTOCOL" in bat:
        check("ToolAgent._build_user_prompt = _ln_prompt" in bat and "ToolAgent._run_python_tool = _ln_run" in bat
              and "_ln_agent.build_chat_payload = _ln_payload" in bat,
              "обёртки lean (дифф/HUD в промпте, событийный протокол, мышление по маркеру) в ячейке")
        check(bat.index("ToolAgent._run_python_tool = _ln_run") < bat.index("if not true_submission"),
              "обёртки lean стоят ДО гейта true_submission")
        if "_ls_system" in bat:
            check("_ls_agent._build_system_prompt = _ls_system" in bat and "TOOL_CALL_FORMAT_GUIDANCE" in bat,
                  "leanshort: компактный системный промпт установлен и несёт правила формата tool-call")
    if "MANDATORY NEW-LEVEL PROTOCOL" in bat:
        check("_pr_agent.run_sandboxed_python = _pr_run" in bat and "ToolAgent._build_user_prompt = _pr_prompt" in bat,
              "обёртки протокола (отказ без PROTOCOL, NOTICE о нулевом диффе) в ячейке")
        check(bat.index("_pr_agent.run_sandboxed_python = _pr_run") < bat.index("if not true_submission"),
              "обёртки стоят ДО гейта true_submission")

    print(f"\n{'ВСЁ ЧИСТО' if not fails else f'ПРОВАЛЕНО ПРОВЕРОК: {len(fails)}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
