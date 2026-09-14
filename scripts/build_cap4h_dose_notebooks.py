"""Два контрольных прогона по раунду 4 критиков (14.09, слово владельца «делаем уже всё»), оба с поиском колёс
соревнования по /kaggle/input (сборка через build() из build_lvfact_reset_notebook):
  1. arc3-stock-flash-cap4h -- потолок игры 14400 с вместо 7920 при конкурентности 28: продолжается ли
     линейный рост балла по времени (усечение базы: 1.36/3.32/4.74/6.11/7.78/10.25 на 15..132 мин).
     ПОРОГИ до пуска: балл на 132-й минуте усечением (h115) должен лечь в разброс базы 6.76-10.25 (контроль);
     балл на 240-й минуте > балл на 132-й минуте того же прогона на >= +4 -- рост продолжается;
     +1..+4 -- замедление; <= +1 -- насыщение. ~4.4 ч квоты.
  2. arc3-dose-conc13 v2 -- повтор дозы (конкурентность 13, вызовов на игру x2): второй дозовый уровень
     для результата «больше вычислений -- не больше прогресса». ПОРОГИ те же, что 12.09: против базы 10.25
     польза при (побед-поражений) >= +8 и медиане >= +8, вред при <= -6; ворота механизма: вызовов на игру
     >= 95. Первый замер: 8.58, уровней 41 против 40, знаки 7/12. ~4.4 ч квоты.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lvfact_reset_notebook import build  # noqa: E402

CAP4_CELL = '''
# =====================================================================
# 4-ЧАСОВОЙ ПОТОЛОК (14.09, раунд 4 критиков): те же 25 игр, конкурентность 28, потолок игры 14400 с.
# Вопрос: продолжается ли линейный рост балла по времени за 132-й минутой. Только оффлайн.
# =====================================================================
_CAP4_S = 14400.0
if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = _CAP4_S
    print("CAP4H: потолок игры %s с (штатный 7920), конкурентность %s. ПОРОГИ: балл на 132-й минуте усечением "
          "в разбросе базы 6.76-10.25; прирост 132->240 мин >= +4 -- рост продолжается, +1..+4 -- замедление, "
          "<= +1 -- насыщение." % (bm.solver.max_runtime_s_per_game, getattr(bm.solver, "concurrency", "?")), flush=True)
'''
dose_lines = open("kernels/notebooks_stockflash_dose/cell15.py", encoding="utf-8").read()
assert "_DOSE_CONC = 13" in dose_lines and "bm.solver.concurrency = _DOSE_CONC" in dose_lines
DOSE_CELL = "\n" + dose_lines + "\n"

if __name__ == "__main__":
    build(CAP4_CELL, "kernels/notebooks_stockflash_cap4h", "sergueimakarov/arc3-stock-flash-cap4h", "arc3 stock flash cap4h", "_CAP4_S = ")
    build(DOSE_CELL, "kernels/notebooks_stockflash_dose_v2", "sergueimakarov/arc3-dose-conc13", "arc3 dose conc13", "_DOSE_CONC = ")
