"""Паспорт машины (30.09): процессор, скорость одного ядра (цикл Python — как главный цикл vLLM), GPU и её предел мощности.
Запускать ПЕРВЫМ на любом новом поде и сравнивать с Kaggle (EPYC 9B45, PYLOOP 0.162 с, 600 Вт) — см. память feedback-machine-probe-before-comparing.
usage: python3 machine_probe.py   (на поде: scp и запуск; вывод сохранить в runs/<под>/machine_probe.txt)
"""
import subprocess, time, os, platform, json
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return repr(e)
print("=== lscpu"); print(sh("lscpu"))
print("=== nproc", os.cpu_count(), "| affinity", len(getattr(os, "sched_getaffinity", lambda p: [0])(0)))
print("=== cpufreq"); print(sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null"))
print("=== free"); print(sh("free -g"))
print("=== nvidia-smi"); print(sh("nvidia-smi --query-gpu=name,power.limit,clocks.max.sm,driver_version --format=csv"))
print("=== python", platform.python_version())
# одно ядро: чистый Python (как цикл планировщика vLLM) — лучшее из 5
def loop(n=3_000_000):
    s = 0; d = {}
    for i in range(n):
        s += i * 3 % 7; d[i & 1023] = s
    return s
best = min((lambda t0: (loop(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(5))
print("=== PYLOOP_SECONDS %.3f  (3 млн итераций, лучшее из 5; меньше — быстрее ядро)" % best)
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
a = np.random.rand(512, 512)
t0 = time.perf_counter()
for _ in range(20): a @ a
print("=== NUMPY_MATMUL_SECONDS %.3f" % (time.perf_counter() - t0))
