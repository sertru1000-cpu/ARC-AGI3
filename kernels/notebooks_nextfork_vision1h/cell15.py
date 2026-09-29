
# =====================================================================
# VISION: картинка доски в разрешении ×16 вместо ×4 (26.09). Ячейкой на датасет v3.
# Было ×4: патч зрения 16 px сливал 4×4 клетки. Картинки из истории вынимаются (окно).
# =====================================================================
import os as _vos, sys as _vsys
import inference.agent.tool_agent as _vta
_vos.environ["MULTIMODAL_CONTEXT"] = "current_grid"
_vos.environ["MULTIMODAL_UPSCALE"] = "16"
_v_orig_persist = _vta.ToolAgent._persistent_history_messages


def _v_strip_images(msg):
    c = msg.get("content")
    if not isinstance(c, list):
        return msg
    texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
    return dict(msg, content="\n".join(t for t in texts if t).replace("\n\nCurrent grid image:", ""))


def _v_persist(self, messages, *a, **kw):
    try:
        messages = [_v_strip_images(m) if m.get("role") == "user" else m for m in messages]
    except Exception as exc:
        print("VISION: сбой очистки истории: %r" % (exc,), file=_vsys.__stderr__, flush=True)
    return _v_orig_persist(self, messages, *a, **kw)


_vta.ToolAgent._persistent_history_messages = _v_persist
print("VISION: картинка текущей доски в каждый вызов (×16), из истории вынимается", flush=True)
