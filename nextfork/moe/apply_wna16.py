"""Ядро MoE для Intel W4A16 AutoRound: вместо Marlin — Triton int4 (MoeWNA16Method), при NF_MOE_WNA16=1.
Профиль пода 04.10: Marlin MoE = 43% времени карты; сервер Франзена жёстко держит Marlin (assert is_auto),
а ветка MoeWNA16 в auto_round.py уже есть, но включается только без Marlin. Без переменной поведение прежнее.
usage: python apply_wna16.py <путь к пакету sglang>"""
import sys
from pathlib import Path

MARK = "nextfork moe wna16"
A = """        if isinstance(layer, FusedMoE):
            if use_marlin:
                return GPTQMarlinMoEMethod(quant_args_marlin)
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
"""
B = f"""        if isinstance(layer, FusedMoE):
            if use_marlin and __import__("os").environ.get("NF_MOE_WNA16") != "1":  # {MARK}
                return GPTQMarlinMoEMethod(quant_args_marlin)
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
"""
p = Path(sys.argv[1]) / "srt/layers/quantization/auto_round.py"
s = p.read_text()
if MARK in s:
    print("уже применено")
else:
    assert s.count(A) == 1, "якорь не найден ровно один раз"
    p.write_text(s.replace(A, B)); print("применено:", p)
