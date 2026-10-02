"""Журнал разговора в песочнице (идея PRO-LONG, 02.10): сообщения, выброшенные обрезкой истории, не теряются —
модель может найти их кодом: `journal` (список записей) и `search_journal('шаблон')`.
Всегда собирается; в песочницу и в промпт попадает только при NEXTFORK_JOURNAL=1.
usage: python apply_journal.py <src root с ARC3-Inference>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
ta = root / "ARC3-Inference/inference/agent/tool_agent.py"
sb = root / "ARC3-Inference/inference/agent/python_tool_sandbox.py"
t = ta.read_text()
s = sb.read_text()
if "_nf_journal" in t:
    print("journal already applied"); sys.exit(0)


def sub(text, old, new, what):
    assert text.count(old) == 1, (what, text.count(old))
    return text.replace(old, new)


# 1) архив выброшенных сообщений
t = sub(t, """        history = self._drop_until_first_user_message(history)
        # A dropped message invalidates the server's prefix for this
""", """        history = self._drop_until_first_user_message(history)
        _nf_dropped = list(messages[1:])[: max(0, len(messages) - 1 - len(history))]
        if _nf_dropped:
            _nf_archive_dropped(self, _nf_dropped)
        # A dropped message invalidates the server's prefix for this
""", "trim hook")

# 2) функция архивации (модульная)
t = sub(t, "def _context_drain_tokens() -> int:\n", '''def _nf_message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        parts.append("[thinking] " + str(reasoning))
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    if content:
        parts.append(str(content))
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments")
        try:
            args = json.loads(args).get("code", args) if isinstance(args, str) else (args or {}).get("code", args)
        except Exception:  # noqa: BLE001
            pass
        parts.append("[code] " + str(args))
    return "\\n".join(parts)


def _nf_archive_dropped(agent: Any, dropped: list[dict[str, Any]]) -> None:
    journal = getattr(agent, "_nf_journal", None)
    if journal is None:
        journal = agent._nf_journal = []
    for message in dropped:
        text = _nf_message_text(message)
        if not text.strip():
            continue
        journal.append({"n": len(journal), "role": str(message.get("role", "")), "text": text[:2500]})
    del journal[:-300]


def _context_drain_tokens() -> int:
''', "archive fn")

# 3) журнал — в состояние песочницы, один раз за вызов python
t = sub(t, """                "last_action_call_result": (
                    dict(persisted_action_result)
                    if isinstance(persisted_action_result, dict)
                    else {}
                ),
            }
""", """                "last_action_call_result": (
                    dict(persisted_action_result)
                    if isinstance(persisted_action_result, dict)
                    else {}
                ),
                **_nf_journal_part(),
            }
""", "state journal")
t = sub(t, """        def _serialized_runtime_state(
            *,
""", """        _nf_first_call = [True]

        def _nf_journal_part() -> dict[str, Any]:
            # the journal travels once per python call (with the initial state), not with every action result
            if not _nf_first_call[0]:
                return {}
            _nf_first_call[0] = False
            if os.environ.get("NEXTFORK_JOURNAL", "0").strip().lower() not in ("1", "true", "on"):
                return {}
            return {"journal": list(getattr(self, "_nf_journal", []) or [])}

        def _serialized_runtime_state(
            *,
""", "state closure vars")

# 4) строка в промпте
anchor2 = "        if undo_exposure_mode() == \"on\":\n            prompt += UNDO_INFO_ADDENDUM\n"
t = sub(t, anchor2, anchor2 + '''        if os.environ.get("NEXTFORK_JOURNAL", "0").strip().lower() in ("1", "true", "on"):
            prompt += (
                "- Turns dropped from your context when history is trimmed are kept in python as `journal` "
                "(entries with n, role, text: your thinking, code and tool output). `search_journal('pattern')` "
                "returns matching entries. Check it before re-testing something you may already have tested.\\n"
            )
''', "prompt")
ta.write_text(t)

# 5) песочница: journal и search_journal
s = sub(s, """            runtime_globals["last_action_call_result"] = action_result
""", """            runtime_globals["last_action_call_result"] = action_result
            if "journal" in state_payload:
                runtime_globals["journal"] = state_payload.get("journal") or []
            runtime_globals.setdefault("journal", [])
""", "sandbox refresh")
s = sub(s, '        runtime_globals["action"] = action\n', '''        runtime_globals["action"] = action

        def search_journal(pattern, limit=8, width=300):
            # entries of `journal` whose text matches pattern (regex, case-insensitive)
            import re as _re
            rx = _re.compile(pattern, _re.I)
            hits = []
            for entry in runtime_globals.get("journal") or []:
                m = rx.search(entry.get("text", ""))
                if m:
                    a = max(0, m.start() - width // 2)
                    hits.append((entry.get("n"), entry.get("role"), entry.get("text", "")[a:a + width]))
            for h in hits[-limit:]:
                print("#%s %s: %s" % (h[0], h[1], h[2].replace(chr(10), " ")))
            if not hits:
                print("no journal entry matches %r" % (pattern,))

        runtime_globals["search_journal"] = search_journal
''', "sandbox search")
sb.write_text(s)
print("journal applied")
