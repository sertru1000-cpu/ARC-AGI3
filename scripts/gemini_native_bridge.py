"""Переходник к родному API Gemini: он отдаёт РАССУЖДЕНИЕ, совместимый эндпоинт — нет (23.09).

Зачем. Учитель нужен нам не только ради ходов, но и ради мысли: у боевой модели рассуждение занимает 32%
промпта, и без него балл падает с 9.43 до 2.91. Проверено запросами: через `/v1beta/openai/chat/completions`
Gemini рассуждение НЕ возвращает (поле пустое даже при явном запросе), а через родной `:generateContent`
с `thinkingConfig.includeThoughts` — возвращает связный текст мысли отдельной частью ответа.

Что делает модуль: переводит запрос обвязки (формат OpenAI) в родной формат и ответ обратно, подкладывая
мысль в поле `reasoning`, которое обвязка уже умеет читать (`_extract_reasoning_text`). Больше ничего в
обвязке менять не нужно — стенд подменяет только транспорт.

Ловушки, заложенные в перевод:
  * аргументы вызова инструмента в родном API — объект, а в формате OpenAI — СТРОКА с JSON;
  * Gemini не принимает часть полей JSON-схемы (`additionalProperties`, `$schema`, `default`), их вырезаем;
  * ответ инструмента уходит частью `functionResponse` в сообщении роли `user`;
  * системное сообщение идёт отдельным полем `systemInstruction`, а не первым сообщением.
"""
import json
from typing import Any

_DROP_SCHEMA_KEYS = {"additionalProperties", "$schema", "default", "examples", "title"}


def _clean_schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k not in _DROP_SCHEMA_KEYS}
    if isinstance(node, list):
        return [_clean_schema(x) for x in node]
    return node


def to_native(payload: dict) -> dict:
    """запрос обвязки (OpenAI) -> тело родного запроса Gemini"""
    contents: list[dict] = []
    system_parts: list[dict] = []
    for m in payload.get("messages", []):
        role = m.get("role")
        text = m.get("content")
        if isinstance(text, list):                      # мультимодальные части оставляем текстом
            text = "\n".join(p.get("text", "") for p in text if isinstance(p, dict))
        if role == "system":
            system_parts.append({"text": text or ""})
        elif role == "tool":
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": m.get("name") or "python",
                "response": {"output": text or ""}}}]})
        elif role == "assistant":
            parts: list[dict] = []
            if text:
                parts.append({"text": text})
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"code": args}
                part = {"functionCall": {"name": fn.get("name") or "python", "args": args or {}}}
                sig = ((call.get("extra_content") or {}).get("google") or {}).get("thought_signature")
                if sig:
                    part["thoughtSignature"] = sig
                parts.append(part)
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text or ""}]})

    body: dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    gen: dict[str, Any] = {"thinkingConfig": {"includeThoughts": True}}
    for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK")):
        if payload.get(src) is not None:
            gen[dst] = payload[src]
    if payload.get("max_tokens"):
        gen["maxOutputTokens"] = payload["max_tokens"]
    body["generationConfig"] = gen

    decls = []
    for t in payload.get("tools") or []:
        fn = t.get("function") or {}
        decls.append({"name": fn.get("name"), "description": fn.get("description", ""),
                      "parameters": _clean_schema(fn.get("parameters") or {"type": "object", "properties": {}})})
    if decls:
        body["tools"] = [{"functionDeclarations": decls}]
    return body


def to_openai(native: dict, model: str) -> dict:
    """ответ родного API -> вид, который разбирает обвязка"""
    cands = native.get("candidates") or []
    parts = (cands[0].get("content") or {}).get("parts", []) if cands else []
    reasoning, content, calls = [], [], []
    for i, p in enumerate(parts):
        if "functionCall" in p:
            fc = p["functionCall"]
            call = {"id": "call_%d" % (i + 1), "type": "function",
                    "function": {"name": fc.get("name"), "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False)}}
            if p.get("thoughtSignature"):
                # ЛОВУШКА (23.09, живая проба): Gemini ТРЕБУЕТ вернуть подпись мысли вместе с прошлым вызовом
                # инструмента, иначе следующий запрос падает с 400 "missing a thought_signature". Кладём её
                # туда же, куда кладёт совместимый эндпоинт, -- обвязка хранит это поле в истории как есть.
                call["extra_content"] = {"google": {"thought_signature": p["thoughtSignature"]}}
            calls.append(call)
        elif p.get("thought"):
            reasoning.append(p.get("text", ""))
        elif "text" in p:
            content.append(p["text"])
    msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(content)}
    if reasoning:
        msg["reasoning"] = "\n".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    u = native.get("usageMetadata") or {}
    return {
        "id": "gemini-native", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": u.get("promptTokenCount", 0),
                  # размышление биллится как выход и в candidatesTokenCount НЕ входит -- считаем по total
                  "completion_tokens": max(int(u.get("candidatesTokenCount", 0)),
                                           int(u.get("totalTokenCount", 0)) - int(u.get("promptTokenCount", 0))),
                  "total_tokens": u.get("totalTokenCount", 0),
                  # родной API называет кэш иначе; без этого счётчик стенда думает, что кэша нет вовсе
                  "prompt_tokens_details": {"cached_tokens": u.get("cachedContentTokenCount", 0)}},
    }
