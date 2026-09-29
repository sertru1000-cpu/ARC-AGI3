
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
