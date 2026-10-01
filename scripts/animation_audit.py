"""Аудит анимаций на публичных играх: сколько ходов отдают промежуточные кадры и сколько в них сведений,
которых нет в паре «доска до — доска после». Ходы берутся из записанного прогона и повторяются на локальном движке.
usage: .venv/bin/python scripts/animation_audit.py runs/flash_dose_conc13
"""
import glob, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_battle_local import Scripted, load_actions

class Anim(Scripted):
    def take_action(self, action):
        prev = self.trace[-1][0] if self.trace else None
        n0 = len(self.trace)
        fr = super().take_action(action)
        if fr is not None and fr.frame and len(self.trace) > n0:
            fs = [np.asarray(f, dtype=np.int16) for f in fr.frame]
            last = fs[-1]; transient = 0
            if prev is not None and len(fs) > 1 and prev.shape == last.shape:
                mid = np.stack([f for f in fs[:-1] if f.shape == last.shape] or [last])
                # клетка «мелькнула»: в промежуточном кадре значение, которого нет ни до хода, ни после
                transient = int(((mid != prev) & (mid != last)).any(axis=0).sum())
            changed = prev is not None and prev.shape == last.shape and bool((prev != last).any())
            self.stats.append((len(fs), transient, changed, self.trace[-1][1]))
        return fr

if __name__ == "__main__":
    run = sys.argv[1]
    logging.disable(logging.WARNING)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})
    print(f"{'игра':5s} {'ходов':>6s} {'ур.':>4s} {'кадров>1 %':>10s} {'медиана кадров':>14s} {'макс':>5s} {'мелькание %':>11s} {'пустой итог, но кадры %':>23s}")
    tot = []
    for g in envs:
        if not glob.glob(f"{run}/artifacts/{g}-*_p0_events.jsonl"):
            continue
        acts = load_actions(run, g)
        if not acts:
            continue
        env = arc.make(g)
        ag = Anim(card_id="replay", game_id=g, agent_name=f"anim.{g}", ROOT_URL="http://localhost", record=False, arc_env=env, tags=["replay"], script=acts)
        ag.stats = []
        ag.main()
        s = ag.stats
        if not s:
            continue
        n = len(s); multi = [x for x in s if x[0] > 1]; tr = [x for x in s if x[1] > 0]
        hidden = [x for x in s if x[1] > 0 and not x[2]]     # итоговая доска не изменилась, а в кадрах что-то было
        lv = max(x[3] for x in s)
        tot.append((g, n, lv, len(multi) / n, len(tr) / n, len(hidden) / n))
        print(f"{g:5s} {n:6d} {lv:4d} {100*len(multi)/n:10.1f} {int(np.median([x[0] for x in multi])) if multi else 1:14d} "
              f"{max(x[0] for x in s):5d} {100*len(tr)/n:11.1f} {100*len(hidden)/n:23.1f}")
    N = sum(t[1] for t in tot)
    print(f"\nигр {len(tot)}, ходов {N}")
    print(f"ходов с промежуточными кадрами: {100*sum(t[3]*t[1] for t in tot)/N:.1f}%; с мельканием: {100*sum(t[4]*t[1] for t in tot)/N:.1f}%")
    rich = [t for t in tot if t[4] >= 0.10]; poor = [t for t in tot if t[4] < 0.10]
    for name, grp in (("с мельканием в >=10% ходов", rich), ("остальные", poor)):
        if grp:
            print(f"игры {name}: {len(grp)} шт., уровней в этом прогоне в среднем {np.mean([t[2] for t in grp]):.2f} ({', '.join(t[0] for t in grp)})")
