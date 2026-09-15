"""Замкнутый цикл «модель думает, алгоритм ходит» (14.09, слово владельца «без вариантов»).

  PROBE   : ~30 скриптовых ходов (стрелки ×2, клики по центрам объектов, случайные клетки) -> переходы.
  SYNTH   : модель пишет state_of(grid)/predict(state, action) по переходам (промпт оффлайн-теста 12.09,
            где Flash дал рабочие программы в 5/24 играх; tn36 -- 118/120 и 200/200 контрфактически).
  PLAN    : WMPlanner исчерпывает абстрактные состояния программы в настоящей среде (tn36: 32/32 за 31 ход).
  REPAIR  : уровень не взят, а фронтир пуст или предсказания врут -> модели показываются контрпримеры
            (доска менялась, а state_of -- нет; predict != реальность) с требованием расширить состояние;
            новая программа -> PLAN. Не больше REPAIRS починок на уровень; дальше -- Go-Explore-разведка
            до конца бюджета (без модели).
  LEVEL   : взят -> та же программа на новом уровне (фронтир строится заново); при расхождениях -- REPAIR.

Модель нужна только в SYNTH/REPAIR: 1 + до REPAIRS вызовов на уровень, ходами она не управляет.
Интерфейс как у GoExplore: observe(frame) + decide(frame, simple_actions, has_click) -> (name, payload).
"""
from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field

import numpy as np

from .goexplore import GoExplore
from .perception import latest_grid, segment, spread_click_targets
from .wmplan import Program, WMPlanner, action_display

logger = logging.getLogger(__name__)
HEX = "0123456789abcdef"
LETTERS = {i: HEX[i] for i in range(16)}

PROMPT = """You are given recorded transitions from an unknown grid game (ARC-AGI-3). The board is a 64x64 grid; each cell is one hex digit (0-f) = a colour. Actions are UP, DOWN, LEFT, RIGHT, SPACE, or MOUSE(row=r, col=c).

Your task: infer the game's mechanics from the examples and write a Python predictor that generalises to UNSEEN transitions of the same game (including later levels with different layouts). An algorithm will use it to PLAN: it enumerates the states your state_of can distinguish and executes real moves to reach each new one, so state_of must capture EVERYTHING that matters for completing a level (positions of movable things, toggles, counters, collected items, doors), not only the most obvious counter.

Write exactly two functions in one ```python block, no other code outside them, no imports except from: collections, itertools, math, re, copy, json:

def state_of(grid):
    # grid: list of 64 strings, each 64 hex chars. Return a hashable, JSON-serialisable summary of the game-relevant state
    # (e.g. tuple of object positions/colours). It must change whenever the game state changes and ignore purely cosmetic detail.
    ...

def predict(state, action):
    # state: a value returned by state_of; action: one of the strings above.
    # Return the state after the action (same format as state_of) or None if you cannot predict this case.
    ...

def progress(state):
    # OPTIONAL third function: a number that is HIGHER when the level is closer to completion, computed from the state only
    # (e.g. minus the distance between the movable object and its apparent target, number of matched/collected items,
    # number of target cells that already equal the reference pattern). NEVER base it on counters, timers or the number of
    # moves made -- those change on every action and would mislead the search. The planner explores high-progress states first.
    ...

def is_goal(state):
    # OPTIONAL fourth function: True when the state completes the level (e.g. the target region equals the reference
    # pattern, all items collected, the movable object stands on its target). The planner searches for such a state
    # inside your model first, then executes the path for real. Omit it if the completion condition is unknown.
    ...

Scoring: a prediction is correct if predict(state_of(before), action) == state_of(after). Only transitions where state_of(before) != state_of(after) count, so a constant state_of scores zero. Cover as many cases as you can; return None only when truly unknown.

EXAMPLES ({n} transitions, in chronological order; "before" of a transition equals "after" of the previous one when consecutive):
{examples}

Now write the two functions."""

REPAIR = """Your previous predictor was used by a planner in the real game. Result: {verdict}

Previous program:
```python
{code}
```

Evidence the planner collected ({n} transitions). Transitions marked STATE_UNCHANGED are ones where the board changed but your state_of returned the same value before and after -- your state misses whatever changed there. Transitions marked PREDICT_WRONG show predicted vs actual state.
{examples}

Rewrite BOTH functions (same contract, one ```python block; you may also add/revise the optional progress(state) and is_goal(state) functions). Extend state_of so that it distinguishes every board change that could matter for completing the level, and make predict consistent with the evidence. Rules: (1) state_of must include the concrete visible things that changed in the STATE_UNCHANGED transitions (read the cell-change lists: rows, columns, colours); (2) predict must NOT return None for every action -- when a click on some region visibly changed cells in the evidence, encode that change; return None only for actions you have no evidence about; (3) if a board change looks like a hidden counter, still model it as visible cells."""


STRATEGY = """You are the strategist for an algorithmic player of an unknown grid game (ARC-AGI-3, 64x64 board, hex digits = colours;
actions UP, DOWN, LEFT, RIGHT, SPACE, MOUSE(row=r, col=c)). The player already has your mechanics model (state_of/predict below) and
has explored {n_states} distinct states of it WITHOUT completing the level. It needs to know WHAT THE LEVEL WANTS.

Mechanics model currently in use:
```python
{code}
```

START board of the level:
{start}

Representative states reached (as cell changes relative to the START board, row,col:old>new), with the moves that led there:
{states}

What single moves did to the board on this level (per move: how many cells changed, where):
{effects}

Answer with ONE ```python block containing:
1. a comment with your hypothesis of the level's goal (one or two sentences: what must the board look like / what must happen);
2. def is_goal(state): -> True when the level is completed according to your hypothesis (state = state_of(grid) of the model above);
3. def progress(state): -> a number that grows as the board gets closer to the goal (NEVER based on counters/timers/number of moves);
4. SUGGESTED_MOVES = [...]: up to 8 concrete actions to execute NEXT from the START board, as strings like "MOUSE(row=12, col=40)" or "UP",
   chosen to test your hypothesis (e.g. click the element you believe is the trigger, move the object to its target).
If the state_of above cannot express your goal (it lacks the relevant elements), say so in the comment and define
is_goal/progress on the BOARD instead: they will receive state=None and a second argument grid (list of 64 strings), i.e. write
def is_goal(state, grid=None) / def progress(state, grid=None)."""


def fmt_transition(i, t, show_before, note=""):
    s = "### transition %d  action: %s  level: %d%s%s\n" % (i, t["action"], t["level"], "  (LEVEL COMPLETED after this action)" if t["level_up"] else "", note)
    if show_before:
        s += "before:\n" + t["before"] + "\n"
    diff = []
    b, a = t["before"].split("\n"), t["after"].split("\n")
    for r, (x, y) in enumerate(zip(b, a)):
        for c, (p, q) in enumerate(zip(x, y)):
            if p != q:
                diff.append((r, c, p, q))
    if len(diff) > 400 and not os.getenv("MY_AGENT_WM_COMPACT"):
        s += "after:\n" + t["after"] + "\n"
    else:
        s += "after = before with these cell changes (row,col: old->new): " + " ".join("%d,%d:%s>%s" % d for d in diff) + "\n"
    return s


def extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.S)
    if m:
        return m.group(1)
    if "```python" in text:
        return text.split("```python", 1)[1]
    return text


def board_text(grid: np.ndarray) -> str:
    return "\n".join("".join(LETTERS.get(int(v), "?") for v in row) for row in grid)


@dataclass
class Transition:
    before: str
    action: str
    after: str
    level: int
    level_up: bool
    note: str = ""


class WMLoop:
    PROBE_CLICKS = 40
    PROBE_RANDOM = 6
    REPAIRS = 2
    STRATEGIES = 4         # вызовов стратега на уровень
    STRATEGY_TESTS = 60    # проверок планировщика без уровня -> стратег
    PROMPT_CAP = 40000     # символов промпта (контекст vLLM 32K токенов)
    PLAN_STALL = 40        # ходов подряд с пустым фронтиром -> починка
    PLAN_TESTS = 80        # проверенных кандидатов без уровня -> починка (программа верна, но цели нет)
    MAX_TOKENS = int(os.getenv("MY_AGENT_WM_MAX_TOKENS", "10000"))
    VARIANTS = 3           # параллельных вариантов на пачку (экономия 14.09 23:55); MY_AGENT_WM_VARIANTS; потолок MY_AGENT_WM_MAX_VARIANTS
    TRACE_DIR = None       # каталог для записи промптов/ответов/программ (MY_AGENT_TRACE_DIR)

    def __init__(self, backend, seed: int = 0, temperature: float = 0.6, goal_hint: str | None = None):
        self.backend = backend
        self.goal_hint = goal_hint      # тест «цель раскрыта» (критик, раунд 5): целевая доска уровня 1 без пути
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.phase = "probe"
        self.probe_plan: list = []
        self.transitions: list[Transition] = []
        self.program: Program | None = None
        self.planner: WMPlanner | None = None
        self.fallback: GoExplore | None = None
        self.repairs = 0
        self.strategies = 0
        self.level = 0
        self.calls = 0
        self.prev_grid: np.ndarray | None = None
        self.last_action: tuple | None = None
        self.stall = 0
        self.stats: dict = {"calls": 0, "repairs": 0, "synth_fail": 0, "phase_log": []}
        import types as _types
        self.st = _types.SimpleNamespace(need_reset=False)   # совместимость с циклом my_agent (GAME_OVER -> RESET)

    # ---------------------------------------------------------------- observe
    def observe(self, frame) -> None:
        grid = latest_grid(frame)
        if grid is None:
            return
        level = int(getattr(frame, "levels_completed", 0) or 0)
        state = str(getattr(frame, "state", ""))
        self.stats["frames"] = self.stats.get("frames", 0) + 1
        if state.endswith("GAME_OVER"):
            self.stats["game_over_frames"] = self.stats.get("game_over_frames", 0) + 1
        if self.prev_grid is not None and self.last_action is not None and self.last_action[0] != "RESET" and not state.endswith("GAME_OVER"):
            self.transitions.append(Transition(board_text(self.prev_grid), action_display(self.last_action), board_text(grid), self.level, level > self.level))
        if level != self.level:
            self.level = level
            self.repairs = 0; self.strategies = 0; self.stall = 0; self._su_seen = 0; self._fe_seen = 0; self._st_seen = 0
            # новый уровень: раскладка другая -- новая проба и синтез с нуля (прошлая программа идёт подсказкой)
            self.prev_program_source = getattr(self, "program_source", None)
            self.phase = "probe"; self.probe_plan = []
            self.stats["phase_log"].append((level, "level_up", self.repairs, len(self.transitions)))
        self.prev_grid = grid
        if self.planner is not None and self.phase == "plan":
            self.planner.observe(frame)
        if self.fallback is not None and self.phase == "fallback":
            self.fallback.observe(frame)

    # ----------------------------------------------------------------- decide
    def decide(self, frame, simple_actions: list[str], has_click: bool) -> tuple[str, dict | None]:
        grid = latest_grid(frame)
        if self.phase == "probe":
            if not self.probe_plan:
                self.probe_plan = self._probe_actions(grid, simple_actions, has_click)
                if not self.probe_plan:
                    self.phase = "synth"
                    return self.decide(frame, simple_actions, has_click)
            act = self.probe_plan.pop(0)
            if not self.probe_plan:
                self.phase = "synth"
            return self._emit(act)
        if self.phase == "synth":
            ok = self._synthesize(repair=False)
            self.phase = "plan" if ok else "fallback"
            self._log_phase()
            if ok and self.goal_hint and self.level == 0 and self.strategies == 0:
                self.strategies += 1
                self._strategize(); self._log_phase()
            return self.decide(frame, simple_actions, has_click)
        if self.phase == "plan":
            act = self.planner.decide(frame, simple_actions, has_click)
            st = self.planner.st.stats
            self.stall = self.stall + 1 if st.get("frontier_empty", 0) > getattr(self, "_fe_seen", 0) else 0
            self._fe_seen = st.get("frontier_empty", 0)
            wrong = st.get("pred_wrong", 0); okp = st.get("pred_ok", 0)
            executed = st.get("executed", 0)
            budget_hit = executed >= self.PLAN_TESTS * (self.repairs + 1)
            # самый быстрый сигнал «состояние неполное»: доска изменилась, а state_of -- нет (>= 2 раз) -> починка сразу
            missing = st.get("state_unchanged", 0) >= 2 and st.get("state_unchanged", 0) > getattr(self, "_su_seen", 0)
            if missing:
                self._su_seen = st.get("state_unchanged", 0)
            # стратег: план исчерпан или потратил STRATEGY_TESTS проверок без уровня, механика при этом не врёт
            mech_bad = (wrong >= 8 and wrong > 2 * okp) or missing
            # v7-проба (15.09 08:54): стратег ни разу не вызван — его затеняла ветка «механика врёт» (починка, потом резерв).
            # Теперь: механика врёт и починки остались -> починка; иначе стратег (по стагнации или бюджету проверок); потом резерв.
            need_strategy = (self.stall >= self.PLAN_STALL or executed >= self.STRATEGY_TESTS * (self.strategies + 1)
                             or (mech_bad and self.repairs >= self.REPAIRS))
            if mech_bad and self.repairs < self.REPAIRS:
                need_strategy = False
            if need_strategy and self.strategies < self.STRATEGIES:
                self.strategies += 1
                if self._strategize():
                    self.stall = 0; self._fe_seen = 0
                self._log_phase()
                return self._emit(act)
            if mech_bad or self.stall >= self.PLAN_STALL or budget_hit:
                if self.repairs < self.REPAIRS:
                    self.repairs += 1
                    if self._synthesize(repair=True):
                        self.stall = 0; self._fe_seen = 0
                    self._log_phase()
                else:
                    self.phase = "fallback"
                    self.fallback = GoExplore(seed=self.rng.randint(0, 1 << 16))
                    self._log_phase()
            return self._emit(act)
        # fallback
        if self.fallback is None:
            self.fallback = GoExplore(seed=self.rng.randint(0, 1 << 16))
        self.fallback.observe(frame)
        return self._emit(self.fallback.decide(frame, simple_actions, has_click))

    def _emit(self, act):
        self.last_action = act
        if act[0] == "RESET":
            # RESET, выданный внешним циклом (GAME_OVER): сообщить внутренним мозгам
            if self.planner is not None and self.phase == "plan" and self.planner.st.last_action != act:
                self.planner._emit(act)
            if self.fallback is not None and self.phase == "fallback" and self.fallback.st.last_action != act:
                self.fallback._emit(act)
        return act

    def _log_phase(self):
        self.stats["phase_log"].append((self.level, self.phase, self.repairs, len(self.transitions)))

    # ------------------------------------------------------------------ probe
    def _probe_actions(self, grid, simple_actions, has_click) -> list:
        plan: list = [(a, None) for a in simple_actions for _ in range(2)]
        if has_click and grid is not None:
            plan += [("ACTION6", {"x": x, "y": y}) for x, y in spread_click_targets(grid, self.PROBE_CLICKS)]
            plan += [("ACTION6", {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)}) for _ in range(self.PROBE_RANDOM)]
        return plan

    # --------------------------------------------------------------- synthesis
    def _synthesize(self, repair: bool) -> bool:
        cur = [t for t in self.transitions if t.level == self.level]
        import os as _os0
        n_tr = int(_os0.getenv("MY_AGENT_WM_TRANS", "40"))
        trans = (cur if len(cur) >= 5 else self.transitions)[-n_tr:]   # переходы ТЕКУЩЕГО уровня: программа для новой раскладки
        if not trans:
            return False
        if not repair or self.program is None:
            parts = []; prev = None
            compact = bool(_os0.getenv("MY_AGENT_WM_COMPACT"))   # Kaggle/vLLM: контекст 32K, hex-доска ~3K токенов -- доска только раз
            for i, t in enumerate(trans):
                parts.append(fmt_transition(i + 1, {"before": t.before, "action": t.action, "after": t.after, "level": t.level, "level_up": t.level_up},
                                            show_before=(i == 0) if compact else (t.before != prev)))
                prev = t.after
            prompt = PROMPT.format(n=len(trans), examples="\n".join(parts))
            if getattr(self, "prev_program_source", None):
                prompt += ("\n\nFor reference, this program worked on the PREVIOUS level of the same game (layout may differ now; "
                           "generalise rather than copy coordinates):\n```python\n" + self.prev_program_source[-6000:] + "\n```")
        else:
            ev = self._evidence(trans)[-12:]
            # экономия: полная доска только один раз (первая улика), остальные -- диффами клеток
            parts = []; prev = None
            for i, (t, note) in enumerate(ev):
                parts.append(fmt_transition(i + 1, {"before": t.before, "action": t.action, "after": t.after, "level": t.level, "level_up": t.level_up}, show_before=(i == 0), note=note))
                prev = t.after
            st = self.planner.st.stats if self.planner else {}
            nst = len(self.planner.st.nodes) if self.planner else 0
            if st.get("frontier_empty", 0) > 0:
                verdict = ("the level was NOT completed; the planner reached all %d states your state_of distinguishes (frontier exhausted), "
                           "so state_of misses what matters; predictions correct %d, wrong %d." % (nst, st.get("pred_ok", 0), st.get("pred_wrong", 0)))
            else:
                verdict = ("the level was NOT completed after testing %d candidate moves over %d distinct states (predictions correct %d, wrong %d). "
                           "Either your state misses something, or the search needs direction: add/revise progress(state) so that states closer to "
                           "completing the level score higher." % (st.get("executed", 0), nst, st.get("pred_ok", 0), st.get("pred_wrong", 0)))
            prompt = REPAIR.format(verdict=verdict, code=self.program_source, n=len(ev), examples="\n".join(parts))
        if len(prompt) > self.PROMPT_CAP:
            prompt = prompt[: self.PROMPT_CAP] + "\n...(truncated)"
        self.calls += 1; self.stats["calls"] += 1
        if repair:
            self.stats["repairs"] += 1
        # K параллельных вариантов, отбор на собранных переходах (без движка)
        import concurrent.futures as _cf
        msgs = [{"role": "user", "content": prompt}]
        import os as _os, time as _time
        k = int(_os.getenv("MY_AGENT_WM_VARIANTS", str(self.VARIANTS)))
        # две формулировки: половина вариантов -- «состояние = объекты и их позиции», половина -- «состояние = ячейки,
        # которые менялись в примерах»; разные абстракции ценнее десяти пересказов одной
        hints = [
            "\n\nHINT for this attempt: represent the state as OBJECTS (connected same-colour components) with their positions/colours; movable objects and their targets first.",
            "\n\nHINT for this attempt: represent the state by the CELLS THAT CHANGED in the examples (list their coordinates and colours as state components); build predict from the observed per-action changes.",
            "\n\nHINT for this attempt: first write down, as a comment, the rule of the game in one sentence (what the player controls, what reacts to clicks, what a level probably requires); then make state_of contain exactly the things named in that sentence.",
            "\n\nHINT for this attempt: keep state_of SMALL and DISCRETE (a tuple of a few integers/characters), so that the number of distinct states is small; put every interactive element you can see on the board into it, even ones never touched in the examples.",
        ]
        max_k = int(_os.getenv("MY_AGENT_WM_MAX_VARIANTS", "12"))
        good_enough = 0.6
        def _one(i):
            m = [{"role": "user", "content": prompt + hints[i % len(hints)]}]
            for attempt in range(2):   # одна повторная попытка на сетевой сбой / лимит запросов
                try:
                    return self.backend.chat(m, max_tokens=self.MAX_TOKENS, temperature=self.temperature)
                except Exception as exc:
                    logger.warning("wmloop: variant %d attempt %d failed: %r", i, attempt, exc); _time.sleep(3 + 5 * attempt)
            return ""
        scored = []; start = 0
        # адаптивно: пачками по k, пока лучший не наберёт good_enough или не кончится max_k; при полном нуле -- две пачки
        while start < max_k:
            idx = list(range(start, min(start + k, max_k)))
            with _cf.ThreadPoolExecutor(max_workers=k) as ex:
                replies = list(ex.map(_one, idx))
            for i, reply in zip(idx, replies):
                code = extract_code(reply)
                self._trace("call%d_%s_v%d" % (self.calls, "repair" if repair else "synth", i), prompt if i == 0 else "", reply, code)
                try:
                    prog = Program(code, LETTERS)
                    prog.raw_state(self.prev_grid if self.prev_grid is not None else np.zeros((64, 64), dtype=np.int16))
                except Exception as exc:
                    scored.append((None, i, ("rejected", str(exc)[:80]), code)); continue
                scored.append((self._score_program(prog, trans), i, None, code, prog))
            start += k
            best_so_far = max((x[0][0] for x in scored if x[0] is not None), default=-1.0)
            if best_so_far >= good_enough:
                break
            if start >= 2 * k and best_so_far <= 0.05:
                break   # два нуля подряд -- дело не в удаче выборки
        self.stats["variants_used"] = self.stats.get("variants_used", 0) + start
        good = [x for x in scored if x[0] is not None]
        self.stats.setdefault("variants", []).append([(x[1], x[0] if x[0] is not None else x[2]) for x in scored])
        if not good:
            self.stats["synth_fail"] += 1
            logger.warning("wmloop: all %d variants rejected", len(scored))
            return False
        good.sort(key=lambda x: x[0], reverse=True)
        best = good[0]
        code, prog = best[3], best[4]
        logger.warning("wmloop: variants scored %s -> chosen v%d", [(x[1], tuple(round(v, 2) for v in x[0])) for x in good], best[1])
        self.program = prog; self.program_source = code
        self.planner = WMPlanner(prog, seed=self.rng.randint(0, 1 << 16))
        self.planner.st.need_reset = True   # план начинается со старта уровня
        return True

    def _score_program(self, prog: Program, trans: list[Transition]) -> tuple:
        """Оценка без движка: (точность × покрытие на нетривиальных переходах, информативность, покрытие).
        информативность = различных состояний / различных досок, штраф за 1 состояние и за состояние == доска."""
        ok = cov = nontriv = 0; states = set(); boards = set(); errs = 0
        for t in trans:
            try:
                sb = prog.state_of(t.before.split("\n")); sa = prog.state_of(t.after.split("\n"))
            except Exception:
                errs += 1; continue
            from .wmplan import _norm
            nb, na = _norm(sb), _norm(sa)
            boards.add(t.after); states.add(na)
            if nb == na:
                continue
            nontriv += 1
            try:
                p = prog.predict(sb, t.action)
            except Exception:
                errs += 1; continue
            if p is None:
                continue
            cov += 1; ok += int(_norm(p) == na)
        acc = ok / nontriv if nontriv else 0.0
        coverage = cov / nontriv if nontriv else 0.0
        nb_ = max(1, len(boards)); ns_ = len(states)
        info = 0.0 if ns_ <= 1 else (1.0 if ns_ < nb_ else 0.5)   # 1 состояние -- бесполезно; состояние == доска -- слабо
        if errs > len(trans) // 2:
            return (-1.0, 0.0, 0.0)
        return (acc * coverage + 0.1 * info, info, coverage)

    def _trace(self, name: str, prompt: str, reply: str, code: str) -> None:
        import os
        d = self.TRACE_DIR or os.getenv("MY_AGENT_TRACE_DIR")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            base = os.path.join(d, "%s_%s" % (getattr(self, "game_tag", "game"), name))
            open(base + ".prompt.txt", "w", encoding="utf-8").write(prompt)
            open(base + ".reply.txt", "w", encoding="utf-8").write(reply)
            open(base + ".py", "w", encoding="utf-8").write(code)
        except Exception as exc:
            logger.warning("wmloop: trace failed: %r", exc)

    def _strategize(self) -> bool:
        """Модель-стратег: дайджест найденного -> гипотеза цели, is_goal/progress, подсказанные ходы."""
        import os as _os2, re as _re2
        pl = self.planner
        dg = pl.digest()
        if dg["start"] is None:
            return False
        start = dg["start"]
        def diff(a, b):
            out = []
            for r, (x, y) in enumerate(zip(a.split("\n"), b.split("\n"))):
                for c, (p, q) in enumerate(zip(x, y)):
                    if p != q:
                        out.append("%d,%d:%s>%s" % (r, c, p, q))
            return out
        states = []
        for n in dg["picks"]:
            d = diff(start.board, n.board)
            states.append("- after moves [%s]: %d cells changed: %s%s" % (", ".join(action_display(a) for a in n.path[-8:]), len(d), " ".join(d[:120]), " ..." if len(d) > 120 else ""))
        # эффекты одиночных ходов на этом уровне (агрегат по действию)
        eff: dict = {}
        for t in [x for x in self.transitions if x.level == self.level][-400:]:
            d = diff(t.before, t.after)
            e = eff.setdefault(t.action, {"n": 0, "changed": 0, "cells": 0, "rows": set(), "cols": set()})
            e["n"] += 1
            if d:
                e["changed"] += 1; e["cells"] += len(d)
                for item in d[:50]:
                    rc = item.split(":")[0].split(","); e["rows"].add(int(rc[0])); e["cols"].add(int(rc[1]))
        eff_lines = []
        for act, e in sorted(eff.items(), key=lambda kv: -kv[1]["changed"])[:40]:
            if e["changed"]:
                eff_lines.append("- %s: changed board %d/%d times, ~%d cells, rows %s, cols %s" % (act, e["changed"], e["n"], e["cells"] // max(1, e["changed"]), sorted(e["rows"])[:6], sorted(e["cols"])[:6]))
        noop = [act for act, e in eff.items() if not e["changed"]]
        if noop:
            eff_lines.append("- no visible change: " + ", ".join(noop[:30]))
        prompt = STRATEGY.format(n_states=dg["n_states"], code=self.program_source[-5000:], start=start.board, states="\n".join(states) or "- none", effects="\n".join(eff_lines) or "- none")
        if self.goal_hint and self.level == 0:
            gd = diff(start.board, self.goal_hint)
            prompt += ("\n\nTARGET STATE (oracle hint): the board below is a state of THIS level from which ONE more move completes the level. "
                       "The completing move and the path to this board are NOT given. Derive the goal criterion from it (what differs from the START board: %d cells: %s%s), "
                       "write is_goal/progress accordingly, and plan moves that transform the START board into this one.\n%s"
                       % (len(gd), " ".join(gd[:150]), " ..." if len(gd) > 150 else "", self.goal_hint))
        if len(prompt) > self.PROMPT_CAP:
            prompt = prompt[: self.PROMPT_CAP] + "\n...(truncated)"
        self.calls += 1; self.stats["calls"] += 1; self.stats["strategies"] = self.stats.get("strategies", 0) + 1
        try:
            reply = self.backend.chat([{"role": "user", "content": prompt}], max_tokens=self.MAX_TOKENS, temperature=self.temperature)
        except Exception as exc:
            logger.warning("wmloop: strategist call failed: %r", exc); return False
        code = extract_code(reply)
        self._trace("call%d_strategy" % self.calls, prompt, reply, code)
        ns: dict = {}
        try:
            exec(code, ns)
        except Exception as exc:
            logger.warning("wmloop: strategist code rejected: %r", exc); return False
        prog = self.program
        def wrap(fn):
            if fn is None:
                return None
            nparams = fn.__code__.co_argcount
            def f(state, _grid_cache={}):
                if nparams >= 2:
                    node = pl.st.nodes.get(pl.st.cur_abstract)
                    grid_rows = (node.board.split("\n") if node is not None else None)
                    return fn(state, grid_rows)
                return fn(state)
            return f
        pl.set_goal_functions(progress=wrap(ns.get("progress")), is_goal=wrap(ns.get("is_goal")))
        moves = []
        for m in (ns.get("SUGGESTED_MOVES") or [])[:8]:
            m = str(m).strip()
            mm = _re2.match(r"MOUSE\(row=(\d+),\s*col=(\d+)\)", m)
            if mm:
                moves.append(("ACTION6", {"x": int(mm.group(2)), "y": int(mm.group(1))}))
            elif m.upper() in ("UP", "DOWN", "LEFT", "RIGHT", "SPACE"):
                moves.append(({"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5"}[m.upper()], None))
        if moves:
            # от старта уровня: RESET + подсказанные ходы
            pl.suggest_moves([("RESET", None)] + moves)
        self.stats["phase_log"].append((self.level, "strategy", self.strategies, len(moves)))
        return True

    def _evidence(self, trans: list[Transition]) -> list[tuple[Transition, str]]:
        """Контрпримеры: доска менялась, а состояние нет; предсказание не совпало; взятия уровня."""
        out = []
        for t in trans:
            note = ""
            try:
                sb = self.program.state_of(t.before.split("\n")); sa = self.program.state_of(t.after.split("\n"))
                if t.before != t.after and sb == sa:
                    note = "  STATE_UNCHANGED"
                else:
                    p = self.program.predict(sb, t.action)
                    if p is not None and p != sa:
                        note = "  PREDICT_WRONG (predicted %r, actual %r)" % (p, sa)
            except Exception:
                note = "  STATE_OF_ERROR"
            if t.level_up or note:
                out.append((t, note))
        if len(out) < 8:
            out += [(t, "") for t in trans[-(8 - len(out)):]]
        return out[-24:]

    def summary(self) -> dict:
        d = dict(self.stats)
        d["phase"] = self.phase; d["transitions"] = len(self.transitions); d["level"] = self.level
        if self.planner is not None:
            d["planner"] = self.planner.summary()
        return d
