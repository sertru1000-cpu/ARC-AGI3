"""Сборка трёх своих слоёв (26.09), каждый — ячейкой на датасет nextfork v3, потолок 1 ч на игру.

Основание — наши же данные за 26.09, не чужие работы:
  * колея: застрявший прогон расходится с успешным на 1–4 ходу уровня, и в 83% развилок он уже пробовал
    выигрышный тип хода — беда в суждении о цели, а не в памяти;
  * cons (согласование двух ответов) и низкая температура у gedouluhui (0.95 против 1.43) вредят:
    подавление разброса гипотез убивает поиск;
  * вход запроса ~20 тыс. токенов, из них прошлые сообщения пользователя ~45 тыс. знаков — это одна и та же
    инструкция, повторённая в каждом из ~10 прошлых ходов; очередь — 87% задержки запроса.

open1h  — разнообразие на старте уровня: первые 2 вызова уровня модель пишет 2–3 гипотезы РАЗНЫХ классов и одну
          различающую пробу (противоположность cons).
fresh1h — сброс гипотезы при застревании: >=30 ходов на уровне без взятия -> цель, план и открытые вопросы
          стираются, история сообщений очищается, модели даётся список испробованного и отброшенная гипотеза
          (не более 2 сбросов на уровень). Порог — scripts/stall_signal.py: на 30 ходах ложные тревоги 40%
          взятых уровней, охват 60% невзятых; факты о механике (world/action model) сохраняются.
short1h — короче вход: в прошлых сообщениях пользователя остаются только строки состояния (что исполнено,
          уровень, допустимые ходы); повторяющаяся инструкция и устаревший блок модели мира вынимаются.
usage: .venv/bin/python scripts/build_own3_notebooks.py
"""
import ast, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OPEN_CELL = r'''
# =====================================================================
# OPEN: разнообразие гипотез в начале каждого уровня (26.09). Ячейкой на датасет v3.
# =====================================================================
import sys as _osys
import inference.agent.tool_agent as _ota
_OPEN_BLOCK = (
    "\nOpening protocol for this level (first turns only): before committing to a plan, write 2-3 genuinely "
    "DIFFERENT hypotheses about this level's goal and key mechanic, labelled H1, H2, H3 -- different classes "
    "(different target object, different win condition, different interaction), not variations of one idea. "
    "Choose ONE short probe (1-3 actions) whose outcome distinguishes them, state what each hypothesis predicts "
    "for it, and execute only that probe. On the next turn, drop the hypotheses the result contradicts before "
    "acting further."
)
_o_orig = _ota.ToolAgent._build_user_prompt


def _o_build(self, action_num, **kw):
    text = _o_orig(self, action_num, **kw)
    try:
        fr = kw.get("current_frame")
        key = (str(self.__dict__.get("_session_runtime_dir")), fr.level if fr is not None else 0)
        if self.__dict__.get("_o_key") != key:
            self._o_key = key; self._o_turns = 0
        self._o_turns += 1
        if self._o_turns <= 2:
            text += _OPEN_BLOCK
            print("OPEN: блок гипотез, уровень %s, вызов %d" % (key[1], self._o_turns), file=_osys.__stderr__, flush=True)
    except Exception as exc:
        print("OPEN: сбой слоя: %r" % (exc,), file=_osys.__stderr__, flush=True)
    return text


_ota.ToolAgent._build_user_prompt = _o_build
print("OPEN: слой включён — первые 2 вызова каждого уровня, поверх датасета v3", flush=True)
'''

FRESH_CELL = r'''
# =====================================================================
# FRESH: сброс гипотезы при застревании (26.09). Ячейкой на датасет v3.
# =====================================================================
import sys as _fsys
from collections import Counter as _FCounter
import inference.agent.tool_agent as _fta
_FRESH_MOVES = 30        # ходов на уровне с последнего сброса (scripts/stall_signal.py)
_FRESH_MAX = 2           # сбросов на уровень
_f_orig = _fta.ToolAgent._build_user_prompt


def _f_build(self, action_num, **kw):
    block = ""
    try:
        fr = kw.get("current_frame"); hist = kw.get("history_entries") or []
        lvl = fr.level if fr is not None else 0
        key = (str(self.__dict__.get("_session_runtime_dir")), lvl)
        st = self.__dict__.setdefault("_f_state", {})
        if st.get("key") != key:
            st.clear(); st.update(key=key, base=0, resets=0)
        on_level = [e for e in hist if getattr(getattr(e, "frame", None), "level", None) == lvl]
        moves = len(on_level)
        if moves - st["base"] >= _FRESH_MOVES and st["resets"] < _FRESH_MAX:
            st["base"] = moves; st["resets"] += 1
            know = self._summarized_knowledge
            old_goal = (know.get("goal_model") or "").strip().replace("\n", " ")[:300]
            old_plan = (know.get("current_plan") or "").strip().replace("\n", " ")[:200]
            for k in ("goal_model", "current_plan", "open_questions"):
                know[k] = ""
            self._history_messages = []
            tried = _FCounter(str(e.action) for e in on_level).most_common(10)
            block = (
                "\nFRESH START on this level: %d moves on this level have not cleared it, so the previous goal "
                "hypothesis is DISCARDED and the conversation history was cleared (facts about mechanics are kept above).\n"
                "- Discarded goal hypothesis: %s\n- Discarded plan: %s\n- Actions already tried on this level (count): %s\n"
                "Do not resume the old plan. Propose a goal hypothesis of a DIFFERENT class (a different target object, a "
                "different win condition, or an interaction you have not tried yet), design one short probe that tests it, and act."
                % (moves, old_goal or "(none recorded)", old_plan or "(none recorded)",
                   ", ".join("%s x%d" % (a, n) for a, n in tried) or "(none)"))
            print("FRESH: сброс %d на уровне %s после %d ходов" % (st["resets"], lvl, moves), file=_fsys.__stderr__, flush=True)
    except Exception as exc:
        print("FRESH: сбой слоя: %r" % (exc,), file=_fsys.__stderr__, flush=True)
    return _f_orig(self, action_num, **kw) + block


_fta.ToolAgent._build_user_prompt = _f_build
print("FRESH: слой включён — сброс после %d ходов на уровне, не более %d раз, поверх датасета v3" % (_FRESH_MOVES, _FRESH_MAX), flush=True)
'''

SHORT_CELL = r'''
# =====================================================================
# SHORT: короче вход — в прошлых сообщениях пользователя только строки состояния (26.09). Ячейкой на датасет v3.
# =====================================================================
import sys as _ssys
import inference.agent.tool_agent as _sta
_S_KEEP = ("The code executed", "Executed actions", "You are still", "You have progressed", "You have completed",
           "The game is over", "No previous", "Current state:", "Valid actions right now:")
_S_IMG = "\n\nCurrent grid image:"
_s_orig = _sta.ToolAgent._persistent_history_messages
_s_stat = {"calls": 0, "before": 0, "after": 0}


def _s_text(t):
    tail = _S_IMG if t.endswith(_S_IMG) else ""
    body = t[: -len(tail)] if tail else t
    lines = [l for l in body.split("\n") if l.startswith(_S_KEEP)]
    short = "\n".join(lines) if lines else body.split(". ")[0][:200]
    return short + tail


def _s_msg(m):
    c = m.get("content")
    if isinstance(c, str):
        return dict(m, content=_s_text(c))
    if isinstance(c, list):
        return dict(m, content=[dict(p, text=_s_text(p.get("text", ""))) if isinstance(p, dict) and p.get("type") == "text" else p for p in c])
    return m


def _s_persist(self, messages, *a, **kw):
    hist = _s_orig(self, messages, *a, **kw)
    try:
        before = sum(len(str(m.get("content", ""))) for m in hist if m.get("role") == "user")
        hist = [_s_msg(m) if m.get("role") == "user" else m for m in hist]
        after = sum(len(str(m.get("content", ""))) for m in hist if m.get("role") == "user")
        _s_stat["calls"] += 1; _s_stat["before"] += before; _s_stat["after"] += after
        if _s_stat["calls"] in (1, 10, 100, 500, 1000):
            print("SHORT: %d сохранений истории, знаков в прошлых сообщениях пользователя: было %d, стало %d" % (
                _s_stat["calls"], _s_stat["before"], _s_stat["after"]), file=_ssys.__stderr__, flush=True)
    except Exception as exc:
        print("SHORT: сбой слоя: %r" % (exc,), file=_ssys.__stderr__, flush=True)
    return hist


_sta.ToolAgent._persistent_history_messages = _s_persist
print("SHORT: слой включён — прошлые сообщения пользователя сжимаются до строк состояния, поверх датасета v3", flush=True)
'''

LAYERS = {"open": OPEN_CELL, "fresh": FRESH_CELL, "short": SHORT_CELL}


def build(name: str, cell: str) -> str:
    nb = json.load(open(os.path.join(ROOT, "kernels/notebooks_nextfork/submission.ipynb"), encoding="utf-8"))
    c13 = "".join(nb["cells"][13]["source"]); old = "bm.solver.max_runtime_s_per_game = 7920.0"
    if c13.count(old) != 1:
        raise SystemExit("не нашёл потолок 7920 в ячейке 13")
    nb["cells"][13]["source"] = c13.replace(old, "bm.solver.max_runtime_s_per_game = 3600.0   # потолок 1 ч: сравнивать только с прогонами, обрезанными до 3600 с").splitlines(keepends=True)
    body = "".join(nb["cells"][15]["source"]); anchor = "    seconds=budget - 600.0\n)"; at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона")
    at += len(anchor); code = body[:at] + "\n" + cell + body[at:]
    nb["cells"][15]["source"] = code.splitlines(keepends=True)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "c", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    out = os.path.join(ROOT, "kernels/notebooks_nextfork_%s1h" % name); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(cell)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open(os.path.join(ROOT, "kernels/notebooks_nextfork/kernel-metadata.json")))
    meta["id"] = "sergueimakarov/arc3-nextfork-%s1h" % name; meta["title"] = "arc3 nextfork %s1h" % name
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    assert code.find(cell.strip()[:40]) < code.find("await bm.run(")
    return out


if __name__ == "__main__":
    for n, c in LAYERS.items():
        print("собрано:", build(n, c))
