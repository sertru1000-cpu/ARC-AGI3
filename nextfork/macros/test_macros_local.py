"""Проверка макро-ходов в НАСТОЯЩЕЙ песочнице обвязки Франзена на локальном движке игр (без модели).
usage: .venv/bin/python nextfork/macros/test_macros_local.py <src root с ARC3-Inference после apply_macros.py>"""
import logging, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "ARC3-Inference"))
from inference.agent import python_tool_sandbox as pts  # noqa: E402
CC = pts.ARC_COLOR_CHARS

logging.disable(logging.WARNING)
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

NAME = {"UP": GameAction.ACTION1, "DOWN": GameAction.ACTION2, "LEFT": GameAction.ACTION3,
        "RIGHT": GameAction.ACTION4, "SPACE": GameAction.ACTION5, "MOUSE": GameAction.ACTION6,
        "UNDO": GameAction.ACTION7}


class Host:
    def __init__(self, game):
        self.arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        gid = [e.game_id for e in self.arc.get_environments() if e.game_id.startswith(game)][0]
        self.env = self.arc.make(gid)
        fr = self.env.reset()
        self.frame = np.asarray(fr.frame[-1], dtype=np.int16)
        self.levels = int(fr.levels_completed or 0)
        self.step = 0
        self.history = [{"action": "", "frame": self._fp()}]
        self.moves = []

    def _fp(self):
        g = self.frame.tolist()
        return {"ascii": "\n".join("".join(CC[v] for v in row) for row in g), "step": self.step,
                "level": self.levels, "shape": list(self.frame.shape), "grid": g}

    def state(self):
        return {"current_frame": self._fp(), "history": self.history[-6:],
                "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "UNDO"]}

    def handle(self, actions, stale_after=None):
        res = {}
        for a in actions:
            name = str(a.get("action")).upper()
            ga = NAME[name]
            if name == "MOUSE":
                ga.set_data({"x": int(a["col"]), "y": int(a["row"])})
            fr = self.env.step(ga, data={"x": int(a["col"]), "y": int(a["row"])} if name == "MOUSE" else None)
            new = np.asarray(fr.frame[-1], dtype=np.int16)
            inner = (slice(2, -2), slice(2, -2))
            changed = new.shape != self.frame.shape or bool((new[inner] != self.frame[inner]).any())
            lv = int(fr.levels_completed or 0)
            self.step += 1
            label = name if name != "MOUSE" else "MOUSE(%s,%s)" % (a["row"], a["col"])
            self.moves.append((label, changed, lv > self.levels, str(fr.state)))
            self.frame, done_level = new, lv > self.levels
            self.levels = lv
            self.history.append({"action": label, "frame": self._fp(), "result": {}})
            res = {"executed": True, "executed_count": 1, "executed_actions": [label], "action_display": label,
                   "gameplay_changed": changed, "board_changed": changed, "level_completed": done_level,
                   "game_over": "GAME_OVER" in str(fr.state), "done": "WIN" in str(fr.state)}
        return {"action_result": res, "state": self.state()}


def go(game, code):
    h = Host(game)
    out = pts.run_sandboxed_python(code=code, timeout_seconds=60, initial_state=h.state(), action_handler=h.handle)
    print(f"--- {game}: {code!r}")
    print("stdout:", (out.get("stdout") or "").strip())
    if out.get("error"):
        print("ERROR:", out["error"][-600:])
    print("ходов на движке:", len(h.moves), [m[0] + ("" if m[1] else "·") for m in h.moves])
    return h, out


if __name__ == "__main__":
    go("ls20", 'run("U3 L2")')
    go("ls20", 'run("L*")')
    go("ls20", 'run("U* D2")')
    go("ls20", 'until("R", lambda f: False, max_steps=4)')
    go("ls20", 'run(["UP", {"action": "LEFT"}])')
    go("vc33", 'print(sorted({n["color"] for n in current_frame.segmentation["nodes"]}))')


def level_path_test():
    # останов на взятии уровня: путь первого уровня из записанного прогона, затем лишние ходы
    import glob, json, re
    run_dir = "runs/flash_dose_conc13"
    for ev in sorted(glob.glob(run_dir + "/artifacts/*_p0_events.jsonl")):
        g = ev.split("/")[-1][:4]
        acts = [json.loads(l) for l in open(ev)]
        acts = [a for a in acts if a.get("type") == "action"]
        k = next((i for i, a in enumerate(acts) if a.get("level_completed") in (True, "True")), None)
        if k is None or k > 25 or any(str(a.get("action_name")) == "RESET" or a.get("board_changed") in (False, "False") for a in acts[:k + 1]):
            continue
        plan = []
        for a in acts[:k + 1]:
            m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", str(a.get("action_display") or ""))
            plan.append({"action": "MOUSE", "row": int(m.group(1)), "col": int(m.group(2))} if m else
                        {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE"}[str(a["action_name"])])
        plan += ["UP", "UP", "UP"]
        go(g, "run(%r)" % (plan,))
        return


if __name__ == "__main__" and len(sys.argv) > 2:
    level_path_test()
    go("vc33", 'click_each("b")')
