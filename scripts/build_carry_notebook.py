"""Перенос модели мира через взятие уровня + полный путь уровня (17.09, слово владельца: «формулировку цели и механики
модель мира при переходе на новый уровень плюс давать полный путь — передаём при взятии уровня; собирай и пуш на 1 час фаза А»).

Что меняет в стоковом Duck (tool_agent.py публичного бандла):
1. ToolAgent._update_summarized_knowledge_from_step_summary при level_transition обнуляет шесть полей рабочей модели мира
   (world/goal/action model, findings, questions, plan). Слой сохраняет world_model, goal_model, action_model с пометкой
   «[carried from level N -- re-check on this board]»; findings/questions/plan по-прежнему стираются (они про старую доску).
   При run_complete/game_over -- стоковое поведение.
2. ToolAgent._build_user_prompt: на всех промптах нового уровня перед стоковым текстом -- блок LEVEL N SOLUTION PATH:
   точная последовательность ходов, взявшая уровень N (от последнего RESET на этом уровне до завершающего хода, в сжатой
   записи «UP x3»), и сколько ходов всего ушло на уровень. Если путь длиннее _CR_PATH_MAX -- последние _CR_PATH_MAX ходов
   с пометкой. Путь восстанавливается из history_entries (полная история партии в runtime state; action -- строка
   вида «MOUSE(row=46, col=38)», RESET -- «RESET»).
Слой стоит в ОБОИХ режимах (TRUE_SUBMISSION тоже), выключатель -- переменная окружения CARRY_LAYER=0.
Маркер [[CARRY]] в stdout на каждый перенос, итог /kaggle/working/carry_stats.json.
Сверка на настоящих историях (события трёх прогонов базы): число ходов на уровень, восстановленное слоем, совпало
с actions_per_level в 121 случае из 121; путь от последнего RESET -- медиана 23, максимум 119 (кап 150 не срабатывает).

Повод (ИЗМЕРЕНО 17.09): в полном прогоне базы поле Cross-level notes, единственное переживающее переход, донесено
в 2 промпта из 146 с пометкой о новом уровне; в транскрипте ft09 модель мира уровня 1 содержательна
(«goal = make grid match the mini-map»), в первом промпте уровня 2 пусто. Прошлый слой lvfact (13–14.09) давал только
завершающий ход + 8 ходов и доску, модель мира при этом стиралась.

Проба (--probe): потолок игры 3600 с. Точка сравнения -- обрезка на 3600 с трёх прогонов одной базовой конфигурации
(scripts/truncate_run.py, h115): flash_v1_phaseA 6.11 / keithtyser 5.40 / tufa 3.91 (среднее 5.14);
второй уровень в 5 / 5 / 5 играх; уровней всего 28 / 27 / 22.
Пороги, записанные ДО пуска:
  механизм -- переходов (transitions) >= 15 (у баз 22–28 уровней за час), путей (paths) = переходов, ошибок 0;
    carried_nonempty >= 30% переходов (модель мира есть в 54% промптов базы: 724 из 1351), иначе перенос инертен и
    проба меряет только путь -- так и докладывать;
  сигнал -- второй уровень в >= 8 играх (у всех трёх баз ровно 5) И уровней >= 30 (максимум баз 28);
  не отличимо -- второй уровень в 4–7 играх или балл в 3.9–6.1;
  вред -- второй уровень в <= 3 играх ИЛИ балл < 3.5 (ниже минимума баз с запасом).

usage:  .venv/bin/python scripts/build_carry_notebook.py --probe
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lvfact_reset_notebook import build  # noqa: E402

PROBE_CAP_S = 3600.0

CARRY_CELL = r'''
# =====================================================================
# ПЕРЕНОС МОДЕЛИ МИРА ЧЕРЕЗ ВЗЯТИЕ УРОВНЯ + ПОЛНЫЙ ПУТЬ УРОВНЯ (17.09). Выключатель CARRY_LAYER=0.
# =====================================================================
import os as _cr_os, json as _cr_json, atexit as _cr_atexit, re as _cr_re
import inference.agent.tool_agent as _cr_wta
_CR_KEEP = ("world_model", "goal_model", "action_model")
_CR_PATH_MAX = __PATH_MAX__
_CR_TAG = _cr_re.compile(r"^\[carried from level \d+ -- re-check on this board\] ")
_cr_stats = {"transitions": 0, "carried_nonempty": 0, "carried_fields": 0, "paths": 0, "path_moves": [], "level_moves": [],
             "prompts_with_path": 0, "errors": 0}

def _cr_rle(acts):
    out = []
    for a in acts:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return ", ".join(a if n == 1 else "%s x%d" % (a, n) for a, n in out)

def _cr_path(history_entries, new_level):
    """(ходов на уровне всего, ходы от последнего RESET до завершающего включительно) для уровня new_level-1."""
    ents = list(history_entries or [])
    idx = None
    for i, e in enumerate(ents):
        try:
            if int(getattr(e.frame, "level", 0) or 0) >= int(new_level):
                idx = i; break
        except Exception:
            continue
    if idx is None or idx == 0:
        return None
    # кадр записи -- ПОСЛЕ хода: первая запись с уровнем new_level-1 -- это завершающий ход предыдущего уровня
    # (или стартовый кадр без хода), поэтому путь начинается со следующей за ней.
    j = idx - 1
    while j - 1 >= 0 and int(getattr(ents[j - 1].frame, "level", 0) or 0) == int(new_level) - 1:
        j -= 1
    start = j + 1
    acts = [str(getattr(e, "action", "") or "").strip() for e in ents[start:idx + 1]]
    acts = [a for a in acts if a]
    total = len(acts)
    last_reset = max([k for k, a in enumerate(acts) if a.upper().startswith("RESET")], default=-1)
    eff = acts[last_reset + 1:]
    return total, eff

def _cr_block(level_done, total, eff):
    shown = eff[-_CR_PATH_MAX:]
    cut = len(eff) - len(shown)
    head = ("LEVEL %d SOLUTION PATH (exact, from the harness log): level %d was completed by this move sequence "
            "(%d moves since the last RESET of that level; %d moves spent on the level in total; the final move completed it):"
            % (level_done, level_done, len(eff), total))
    body = ("(first %d moves omitted) " % cut if cut > 0 else "") + _cr_rle(shown) + "."
    tail = ("Your world/goal/action model from level %d is carried into the world model below, marked [carried]. "
            "Check on the new board whether the same goal and mechanics still hold and whether an analogous sequence applies, "
            "then act; revise the carried model where the new board contradicts it." % level_done)
    return head + "\n" + body + "\n" + tail

if _cr_os.environ.get("CARRY_LAYER", "1") != "0":
    _cr_orig_update = _cr_wta.ToolAgent._update_summarized_knowledge_from_step_summary
    def _cr_update(self):
        s = getattr(self, "_last_step_summary", None) or {}
        if not (s.get("level_transition") and not s.get("run_complete") and not s.get("game_over")):
            return _cr_orig_update(self)
        try:
            know = getattr(self, "_summarized_knowledge", None) or {}
            saved = {k: _CR_TAG.sub("", str(know.get(k, "") or "")) for k in _CR_KEEP}
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой чтения модели мира: %r" % (_e,), flush=True)
            return _cr_orig_update(self)
        out = _cr_orig_update(self)
        try:
            try:
                new_level = int(s.get("level"))
            except (TypeError, ValueError):
                new_level = None
            prev = (new_level - 1) if new_level else 0
            n = 0
            for k, v in saved.items():
                if v.strip():
                    self._summarized_knowledge[k] = "[carried from level %d -- re-check on this board] %s" % (prev, v); n += 1
            _cr_stats["transitions"] += 1; _cr_stats["carried_fields"] += n
            if n:
                _cr_stats["carried_nonempty"] += 1
            print("[[CARRY]] переход на уровень %s: перенесено полей %d (%s)" % (new_level, n, ", ".join(k for k, v in saved.items() if v.strip()) or "-"), flush=True)
            _cr_dump()
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой переноса: %r" % (_e,), flush=True)
        return out
    _cr_wta.ToolAgent._update_summarized_knowledge_from_step_summary = _cr_update

    _cr_orig_prompt = _cr_wta.ToolAgent._build_user_prompt
    def _cr_prompt(self, action_num, *args, **kwargs):
        text = _cr_orig_prompt(self, action_num, *args, **kwargs)
        try:
            st = getattr(self, "_cr_state", None)
            if st is None:
                st = {"level": None, "block": None}; self._cr_state = st
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is not None and lv > st["level"]:
                    st["block"] = None
                    got = _cr_path(kwargs.get("history_entries"), lv)
                    if got is not None:
                        total, eff = got
                        st["block"] = _cr_block(lv - 1, total, eff)
                        _cr_stats["paths"] += 1; _cr_stats["path_moves"].append(len(eff)); _cr_stats["level_moves"].append(total)
                        print("[[CARRY]] путь уровня %d: %d ходов от последнего RESET, всего %d" % (lv - 1, len(eff), total), flush=True)
                        _cr_dump()
                st["level"] = lv if st["level"] is None else max(st["level"], lv)
            if st.get("block"):
                _cr_stats["prompts_with_path"] += 1
                return st["block"] + "\n\n" + text
            return text
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой промпта: %r" % (_e,), flush=True)
            return text
    _cr_wta.ToolAgent._build_user_prompt = _cr_prompt

    def _cr_dump():
        try:
            _cr_os.makedirs("/kaggle/working", exist_ok=True)
            _cr_json.dump(_cr_stats, open("/kaggle/working/carry_stats.json", "w"), indent=1)
        except Exception:
            pass
    _cr_atexit.register(_cr_dump)
    print("[[CARRY]] слой установлен: модель мира (world/goal/action) переносится через взятие уровня, путь уровня во входе "
          "(до %d ходов); TRUE_SUBMISSION=%s" % (_CR_PATH_MAX, TRUE_SUBMISSION), flush=True)
'''


def cell(path_max: int) -> str:
    return CARRY_CELL.replace("__PATH_MAX__", str(int(path_max)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="потолок игры %s с (проба 1 ч)" % PROBE_CAP_S)
    ap.add_argument("--path-max", type=int, default=150)
    a = ap.parse_args()
    out = "kernels/notebooks_stockflash_carry"
    slug = "sergueimakarov/arc3-stock-flash-carry"
    build(cell(a.path_max), out, slug, "arc3 stock flash carry", "_cr_stats = ")
    if a.probe:
        p = os.path.join(out, "submission.ipynb")
        nb = json.load(open(p, encoding="utf-8"))
        c15 = "".join(nb["cells"][15]["source"])
        marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
        assert marker in c15
        c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n\n" % PROBE_CAP_S + marker, 1)
        nb["cells"][15]["source"] = c15.splitlines(keepends=True)
        json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ok   проба: потолок игры %s с" % PROBE_CAP_S)


if __name__ == "__main__":
    main()
