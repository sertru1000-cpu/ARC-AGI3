"""ЦЕЛЬ КАК ПРЕДИКАТ, БЕЗ ВСЯКОГО ПЕРЕБОРА (18.09, слово владельца: «делаем так — вообще без перебора, но предикат
передаём, 80 минут на фазу А»).

Почему без перебора. Перебор в бою тратит зачётные ходы: даже когда он НЕ берёт уровень, его ходы остаются в знаменателе
и обнуляют этот уровень. ИЗМЕРЕНО на трёх базах-80: в 16-18 играх из 25 перебор уровень не возьмёт, и потеря составила бы
2.20 / 2.27 / 2.29 балла (30-42% всего балла). Здесь слой не делает НИ ОДНОГО хода движка.

Что делает слой:
  1. следит за ходами САМОЙ модели (обёртка _HarnessGameSession._execute_action, только чтение после хода) и копит
     до 200 образцов состояний текущего уровня;
  2. когда МОДЕЛЬ берёт уровень, снимает кадр взятия (кадры анимации завершающего хода; последний уже принадлежит
     следующему уровню) и выводит цель в СИЛЬНОМ режиме: утверждения из словаря, истинные в кадре цели и ложные во всех
     образцах этого уровня (офлайн-замер: цель отделяется в 9 играх из 10);
  3. кладёт до трёх таких утверждений во вход модели на следующих уровнях с пометкой «вид цели тот же, числа другие»
     (перенос вида цели измерен: 80% из 30 пар соседних уровней; параметры переносятся лишь в 33%);
  4. обновляет цель после каждого нового взятия уровня.
Ходов движка: ноль. Токенов: три строки во входе.

usage:  .venv/bin/python scripts/build_predhint_notebook.py --probe [--cap 4800]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lvfact_reset_notebook import build  # noqa: E402
from build_goalhint_notebook import GOAL_HELPERS  # noqa: E402

PRED_WIRE = r'''
import inference.framework.solver as _ph_solver

_ph_stats = {"levels": 0, "goals": 0, "empty": 0, "prompts": 0, "errors": 0, "statements": []}
_ph_orig_exec = _ph_solver._HarnessGameSession._execute_action


def _ph_exec(self, action, *a, **k):
    """только ЧТЕНИЕ после хода модели: образцы состояний уровня и кадр взятия. Ходов не добавляет."""
    payload = _ph_orig_exec(self, action, *a, **k)
    try:
        st = getattr(self, "_ph_state", None)
        if st is None:
            st = {"samples": [], "n": 0}; self._ph_state = st
        cur = self.game.current_state
        if isinstance(payload, dict) and payload.get("level_completed"):
            frames = [_gh_np.asarray(x, dtype=_gh_np.int16) for x in _ph_solver._raw_frames(cur)]
            goal = (frames[:-1] or frames)[-1] if frames else None
            stmts = _gh_infer(goal, st["samples"])
            lvl = int(getattr(cur, "levels_completed", 0) or 0)
            _ph_stats["levels"] += 1
            if stmts:
                _ph_stats["goals"] += 1; _ph_stats["statements"].append(stmts)
                self._ph_goal = {"level_done": lvl, "stmts": stmts}
                print("[[PRED]] уровень %d взят моделью; цель: %s" % (lvl, " | ".join(stmts)), flush=True)
            else:
                _ph_stats["empty"] += 1
                print("[[PRED]] уровень %d взят моделью, ни одно утверждение цель не отделило (образцов %d)"
                      % (lvl, len(st["samples"])), flush=True)
            st["samples"] = []; st["n"] = 0
            _ph_dump()
        else:
            st["n"] += 1
            if len(st["samples"]) < 200 and st["n"] % 2 == 0:
                st["samples"].append(_gh_np.asarray(_ph_solver._grid_from_state(cur), dtype=_gh_np.int16))
    except Exception as _e:
        _ph_stats["errors"] += 1
        print("[PRED] сбой съёма: %r" % (_e,), flush=True)
    return payload


def _ph_block(level_done, stmts):
    return ("HARNESS-INFERRED GOAL OF THE PREVIOUS LEVEL. When level %d was completed, the harness compared that board "
            "with every other board seen on that level. These statements held only at completion:\n%s\n"
            "Levels of one game usually share the KIND of goal and differ in the numbers (more objects, obstacles, "
            "distractors). Aim for the same kind of condition here, re-derive the exact numbers yourself, write the goal "
            "you settle on in `Goal model:` and check it against the board before long plans."
            % (level_done, "\n".join("  - %s" % s for s in stmts)))


if _gh_os.environ.get("PREDHINT", "1") != "0":
    _ph_solver._HarnessGameSession._execute_action = _ph_exec
    _ph_orig_prompt = _gh_wta.ToolAgent._build_user_prompt

    def _ph_prompt(self, action_num, *args, **kwargs):
        text = _ph_orig_prompt(self, action_num, *args, **kwargs)
        try:
            cb = getattr(self, "_step_env_callback", None); sess = getattr(cb, "__self__", None)
            goal = getattr(sess, "_ph_goal", None) if sess is not None else None
            if goal and goal.get("stmts"):
                _ph_stats["prompts"] += 1
                return _ph_block(int(goal.get("level_done") or 0), list(goal["stmts"])) + "\n\n" + text
        except Exception as _e:
            _ph_stats["errors"] += 1; print("[PRED] сбой промпта: %r" % (_e,), flush=True)
        return text

    _gh_wta.ToolAgent._build_user_prompt = _ph_prompt

    def _ph_dump():
        try:
            _gh_os.makedirs("/kaggle/working", exist_ok=True)
            _gh_json.dump(_ph_stats, open("/kaggle/working/predhint_stats.json", "w"), ensure_ascii=False, indent=1)
        except Exception:
            pass

    _gh_atexit.register(_ph_dump)
    print("[[PRED]] слой установлен: ходов движка не тратит; цель выводится по кадру взятия уровня МОДЕЛЬЮ и идёт "
          "во вход на следующих уровнях (до %d утверждений); TRUE_SUBMISSION=%s" % (_GH_MAX, TRUE_SUBMISSION), flush=True)
else:
    def _ph_dump():
        pass
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--cap", type=float, default=4800.0)
    a = ap.parse_args()
    cell = GOAL_HELPERS + PRED_WIRE
    out = "kernels/notebooks_stockflash_predhint"
    build(cell, out, "sergueimakarov/arc3-stock-flash-predhint", "arc3 stock flash predhint", "_ph_stats = ")
    if a.probe:
        p = os.path.join(out, "submission.ipynb")
        nb = json.load(open(p, encoding="utf-8"))
        c15 = "".join(nb["cells"][15]["source"])
        marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
        assert marker in c15
        c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n\n" % a.cap + marker, 1)
        nb["cells"][15]["source"] = c15.splitlines(keepends=True)
        json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ok   проба: потолок игры %s с" % a.cap)


if __name__ == "__main__":
    main()
