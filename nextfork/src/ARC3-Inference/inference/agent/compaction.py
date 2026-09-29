"""Сжатие истории вместо обрезки (29.09) — по статье OpenAI «How enabling two settings tripled our scores on
ARC-AGI-3»: сохранённое рассуждение + сжатие подняли GPT-5.6 Sol с 13.3% до 38.3% при вшестеро меньшем выходе.
Рассуждение наша обвязка уже сохраняет; вместо сжатия у неё обрезка (_drop_oldest_history_block): во входе живут
медианно 5 последних ходов модели, старое пропадает, а вход всё время почти полный (~17–21 тыс. токенов) — это и
память модели, и кэш vLLM (3–6 одновременных запросов из 16).

Включение: NEXTFORK_COMPACT=1 (ставится из inference/framework/solver.py). ВЫКЛЮЧЕНО по умолчанию.
  NEXTFORK_COMPACT_TOKENS  — порог оценки истории, после которого сжимать (по умолчанию 12000);
  NEXTFORK_COMPACT_KEEP    — сколько последних ходов модели оставить дословно (по умолчанию 2);
  NEXTFORK_COMPACT_MAXOUT  — предел ответа на запрос сжатия, вместе с рассуждением (по умолчанию 4000).

Как сделано, чтобы переносилось хорошо (наша модель на сжатие не обучена, в отличие от моделей OpenAI):
  * факты пишет обвязка, а не модель: уровень, ходов на уровне, какие ходы меняли доску и сколько раз, последние
    ходы — из журнала игры (history). Модели остаётся пересказать смысл по строгому шаблону разделов;
  * прежний пересказ входит в историю, поэтому следующий пересказ накапливает память, а не начинает с нуля;
  * последние KEEP ходов модели (с вызовами кода и их результатами) остаются как есть — нить не рвётся;
  * любой сбой (ошибка сервера, пустой ответ, обрыв по длине) — сжатия нет, работает прежняя обрезка.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any

import inference.agent.tool_agent as _ta
from inference.agent.runtime_state import load_runtime_state

THRESHOLD = int(os.environ.get("NEXTFORK_COMPACT_TOKENS", "12000") or 12000)
KEEP = int(os.environ.get("NEXTFORK_COMPACT_KEEP", "2") or 2)
MAXOUT = int(os.environ.get("NEXTFORK_COMPACT_MAXOUT", "4000") or 4000)
MARK = "[GAME SUMMARY — written by you at a context compaction; your earlier turns were replaced by it]"
STATS = {"compactions": 0, "failed": 0, "chars": 0}

COMPACT_PROMPT = (
    # 29.09: формат как у собственного агента Qwen (qwen-code, getCompressionPrompt): сначала <analysis>, потом
    # XML-снимок <state_snapshot> — привычная модели структура переносится лучше самодельной; разделы — про игру.
    "CONTEXT COMPACTION. Your earlier turns are about to be removed from your context and replaced by the state "
    "snapshot you write now. You will continue this same game with only this snapshot, the last two turns and the next "
    "observation, so write it for yourself: dense, concrete, no filler. If an older <state_snapshot> is in your context, "
    "merge it in: keep everything still true, drop what was disproved.\n"
    "First think inside <analysis>...</analysis> (it will be discarded). Then output exactly:\n"
    "<state_snapshot>\n"
    "<goal>what completes this level and the evidence for it; what completed earlier levels</goal>\n"
    "<controls>every action used and what it does - mark each 'verified' (seen more than once) or 'hypothesis'</controls>\n"
    "<key_knowledge>important objects (letter/colour, where, role) and confirmed rules: blocking, pushing, counters, "
    "the HUD/timer bar, what resets the level</key_knowledge>\n"
    "<tried_and_failed>moves, sequences and click targets already tried on this level and their outcome - never repeat "
    "these</tried_and_failed>\n"
    "<current_state>positions now and what changed most recently</current_state>\n"
    "<next_steps>the next 3-5 concrete steps</next_steps>\n"
    "</state_snapshot>\n"
    "Under 350 words inside the snapshot, no code, do not call any tool.\n"
)


def _facts(state_path) -> str:
    """Надёжные факты из журнала игры — пишет обвязка, не модель."""
    try:
        cur, hist = load_runtime_state(state_path)
    except Exception:  # noqa: BLE001
        return ""
    if cur is None:
        return ""
    lv = cur.level
    level_hist = [h for h in hist if getattr(h.frame, "level", None) == lv]
    changed, same = Counter(), Counter()
    prev = None
    for h in hist:
        if prev is not None and getattr(h.frame, "level", None) == lv and h.action:
            name = str(h.action).split("(")[0]
            (changed if h.frame.grid != prev.grid else same)[name] += 1
        prev = h.frame
    lines = ["Reliable facts from the game log (written by the harness, not by you):",
             f"- level {lv}; moves made on this level: {max(0, len(level_hist) - 1)}; moves in the whole game: {max(0, len(hist) - 1)}"]
    for name in sorted(set(changed) | set(same)):
        lines.append(f"- {name}: changed the board {changed[name]} time(s), no change {same[name]} time(s) on this level")
    last = [str(h.action) for h in hist[-12:] if h.action]
    if last:
        lines.append("- last moves: " + ", ".join(last))
    return "\n".join(lines)


def _tail_turns(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Последние keep ходов модели дословно — начиная с сообщения обвязки, которое их вызвало."""
    idx = [i for i, m in enumerate(history) if m.get("role") == "assistant"]
    if keep <= 0 or not idx:
        return []
    start = idx[-keep] if len(idx) >= keep else idx[0]
    while start > 0 and history[start].get("role") != "user":
        start -= 1
    return history[start:] if history[start].get("role") == "user" else history[start + 1:]


def _compact(self, state_path, request_timeout_seconds=None) -> bool:
    history = list(self._history_messages)
    if not history:
        return False
    system = {"role": "system", "content": self._system_prompt}
    try:
        est = self._estimate_request_input_tokens([system, *history], tools=None)
    except Exception:  # noqa: BLE001
        return False
    if est < THRESHOLD:
        return False
    facts = _facts(state_path)
    ask = {"role": "user", "content": COMPACT_PROMPT + ("\n" + facts if facts else "")}
    saved = self._max_output_tokens
    try:
        self._max_output_tokens = MAXOUT
        kw = {"tools": None}
        if request_timeout_seconds is not None:
            kw["request_timeout_seconds"] = request_timeout_seconds
        result = self._chat_completion([system, *history, ask], **kw)
        self._accumulate_usage_tokens(result.usage)
        summary = str((result.message or {}).get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001
        STATS["failed"] += 1
        print("NEXTFORK COMPACT: сбой сжатия, остаётся обрезка: %r" % (exc,), file=sys.__stderr__, flush=True)
        return False
    finally:
        self._max_output_tokens = saved
    if "<state_snapshot>" in summary:
        summary = summary[summary.index("<state_snapshot>"):]
        if "</state_snapshot>" in summary:
            summary = summary[:summary.index("</state_snapshot>") + len("</state_snapshot>")]
    elif "</analysis>" in summary:
        summary = summary.split("</analysis>", 1)[1].strip()
    if result.finish_reason == "length" or len(summary) < 80:
        STATS["failed"] += 1
        return False
    tail = _tail_turns(history, KEEP)
    note = {"role": "user", "content": MARK + "\n" + summary + ("\n\n" + facts if facts else "")}
    ack = {"role": "assistant", "content": "Noted. I will continue from this summary and not repeat failed attempts."}
    self._history_messages = [note, ack, *tail]
    STATS["compactions"] += 1
    STATS["chars"] += len(summary)
    if STATS["compactions"] in (1, 10, 50, 200):
        print("NEXTFORK COMPACT: сжатие №%d (оценка истории %d токенов -> пересказ %d знаков)"
              % (STATS["compactions"], est, len(summary)), file=sys.__stderr__, flush=True)
    return True


_orig_analyze = _ta.ToolAgent.analyze


def _analyze(self, state_path, action_num, *args, **kwargs):
    try:
        if state_path.exists():
            self._ensure_session(state_path)
            _compact(self, state_path, kwargs.get("request_timeout_seconds"))
    except Exception as exc:  # noqa: BLE001
        print("NEXTFORK COMPACT: пропуск: %r" % (exc,), file=sys.__stderr__, flush=True)
    return _orig_analyze(self, state_path, action_num, *args, **kwargs)


_ta.ToolAgent.analyze = _analyze
print("NEXTFORK COMPACT: сжатие истории при оценке > %d токенов, дословно последних ходов %d" % (THRESHOLD, KEEP), flush=True)
