"""Проба «дообучение во время игры»: выгрузка скрытых состояний перед lm_head на диск (для обучения поправки вне сервера).

Правит sglang/srt/layers/logits_processor.py: при переменной NF_DUMP_DIR каждый вызов с запросом логарифмов вероятностей
(extend_return_logprob) сохраняет pruned_states — нормированные скрытые состояния ровно тех позиций, для которых сервер
считает логарифмы вероятностей (от logprob_start_len до конца) — в NF_DUMP_DIR/<номер>.pt (float16). Запросы шлются по одному,
префилл одним куском (окно < chunked-prefill), черновик выключен — один вызов = один запрос.
usage: python apply_dump.py <путь к пакету sglang>
"""
import sys
from pathlib import Path

MARK = "nextfork ttt dump"
ANCHOR = """        hidden_states_to_store = self._get_hidden_states_to_store(
            hidden_states,"""
ADD = f"""        if logits_metadata.extend_return_logprob and __import__("os").environ.get("NF_DUMP_DIR"):  # {MARK}
            import os as _o, torch as _t
            _d = _o.environ["NF_DUMP_DIR"]
            _n = len([f for f in _o.listdir(_d) if f.endswith(".pt")])
            _t.save(pruned_states.detach().to(_t.float16).cpu(), _o.path.join(_d, f"{{_n:05d}}.pt"))
"""


def main(root: Path):
    p = root / "srt/layers/logits_processor.py"
    s = p.read_text()
    if MARK in s:
        print("уже применено"); return
    assert s.count(ANCHOR) == 1, "якорь не найден ровно один раз"
    p.write_text(s.replace(ANCHOR, ADD + ANCHOR))
    print("применено:", p)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
