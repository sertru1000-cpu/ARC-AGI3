#!/usr/bin/env bash
# Матрица вариантов сервера на поде: для каждого — перезапуск сервера с переменными, нагрузка, (опц.) проба качества кэша.
# usage: nohup bash /workspace/pod_franzen/matrix.sh <вариант> [<вариант> ...] > /workspace/run/matrix.log 2>&1 &
W=/workspace; cd $W
export MODEL_DIR=$W/models/intel DRAFT_MODEL_DIR=$W/models/albucino WHEELHOUSE_DIR=$W/wheels/pennyroyal

stop_server() {
  pkill -f "[p]od_franzen/serve" ; pkill -f "[s]glang serve" ; pkill -f "[s]glang.launch_server"
  for i in $(seq 1 60); do ss -ltn | grep -q ":8101 " || break; sleep 2; done
  pkill -9 -f "[s]glang" ; sleep 10
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

start_server() {  # $1 = имя, остальное — переменные окружения
  local name=$1; shift
  mkdir -p $W/run/$name
  env "$@" WORKING_DIR=$W/run/$name setsid nohup python3 ${SERVE_SCRIPT:-$W/pod_franzen/serve.py} > $W/serve_$name.log 2>&1 < /dev/null &
  for i in $(seq 1 240); do
    curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8101/health | grep -q 200 && { echo "$name READY $(date -u +%T)"; return 0; }
    if grep -q "Traceback\|СЕРВЕР УПАЛ\|Error\|error:" $W/serve_$name.log 2>/dev/null && ! pgrep -f "sglang serve" >/dev/null; then echo "$name FAILED"; tail -30 $W/serve_$name.log; return 1; fi
    sleep 5
  done
  echo "$name TIMEOUT"; return 1
}

variant() {  # имя streams probe(0/1) переменные...
  local name=$1 streams=$2 probe=$3; shift 3
  echo "===== $name ($(date -u +%T)) streams=$streams probe=$probe env: $*"
  stop_server
  start_server $name "$@" || return
  grep -E "KV Cache is allocated|Load weight end" $W/run/$name/serve.log | head -3 | cut -c1-200
  [ "$probe" = "1" ] && python3 $W/pod_franzen/kvprobe.py --tag $name --port 8101 --out $W/run/kvprobe_$name.json 2>&1 | tail -3
  nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm --format=csv,noheader,nounits -l 2 > $W/run/smi_$name.csv 2>/dev/null & SMI=$!
  [ "${PROFILE:-0}" = "1" ] && { python3 $W/pod_franzen/profile_trace.py --port 8101 --out $W/run/prof_$name > $W/run/prof_$name.log 2>&1 & }
  [ "$streams" != "0" ] && python3 $W/replay_load.py --run $W/replay --url http://127.0.0.1:8101/v1 --streams $streams ${REPLAY_ARGS:-} --out $W/run/replay_$name.json 2>&1 | tail -14
  kill $SMI 2>/dev/null
  grep -E "#running-req" $W/run/$name/serve.log | grep -o "full token usage: [0-9.]*" | sort -t: -k2 -n | tail -1
  python3 - "$W/run/$name/serve.log" "$W/run/smi_$name.csv" <<'PY'
import re,sys,statistics as st
L=open(sys.argv[1],errors="ignore").read().splitlines()
d=[l for l in L if "] Decode batch" in l]
g=[float(x) for l in d for x in re.findall(r"gen throughput \(token/s\): ([0-9.]+)",l)]
acc=[float(x) for l in d for x in re.findall(r"accept len: ([0-9.]+)",l)]
cg=[l for l in d if "cuda graph: False" in l]
r=[int(x) for l in d for x in re.findall(r"#running-req: (\d+)",l)]
try: smi=[list(map(float,x.split(","))) for x in open(sys.argv[2]) if x.strip()]
except Exception: smi=[]
print("МЕТРИКИ: декод-шагов %d, без CUDA-графа %.1f%%, средняя генерация %.0f т/с, принято %.2f, активных %.1f, мощность %.0f Вт, загрузка %.0f%%"%(
 len(d),100*len(cg)/max(1,len(d)),st.mean(g) if g else 0,st.mean(acc) if acc else 0,st.mean(r) if r else 0,
 st.mean(x[0] for x in smi) if smi else 0, st.mean(x[1] for x in smi) if smi else 0))
PY
  free -g | sed -n 2p
}

for v in "$@"; do
  case $v in
    fp8)       variant fp8 0 1 POD_KVDTYPE=fp8_e4m3 ;;
    s8)        variant s8 8 0 POD_MAXREQ=8 ;;
    s14)       variant s14 14 0 POD_MAXREQ=14 ;;
    fp4)       variant fp4 10 1 POD_KVDTYPE=nvfp4 ;;
    fp4s14)    variant fp4s14 14 0 POD_KVDTYPE=nvfp4 POD_MAXREQ=14 ;;
    bf16)      variant bf16 0 1 POD_KVDTYPE=bf16 POD_SPEC=0 ;;
    hicache)   variant hicache 10 0 POD_HICACHE_GB=30 ;;
    w250)      variant w250 4 0 POD_CTX_K=258 POD_MAXREQ=4 POD_HICACHE_GB=30 ;;
    fp4q)      python3 $W/fp4qsa/apply_fp4qsa.py /tmp/sgl-intel/venv/lib/python3.12/site-packages/sglang
               variant fp4q 10 1 POD_KVDTYPE=nvfp4 NEXTFORK_FP4_QSA=1 NEXTFORK_FP4_NOWS=1 ;;   # свой патч QSA для nvfp4
    # 20 одновременных игр: записанные 10 игр дважды (--dup 2), слотов Mamba 100 (по 5 на запрос)
    s20)       REPLAY_ARGS='--dup 2' variant s20 20 0 POD_MAXREQ=20 POD_MAMBA=100 ;;
    s20hc)     REPLAY_ARGS='--dup 2' variant s20hc 20 0 POD_MAXREQ=20 POD_MAMBA=100 POD_HICACHE_GB=30 ;;
    s20fp4q)   python3 $W/fp4qsa/apply_fp4qsa.py /tmp/sgl-intel/venv/lib/python3.12/site-packages/sglang
               REPLAY_ARGS='--dup 2' variant s20fp4q 20 0 POD_MAXREQ=20 POD_MAMBA=100 POD_KVDTYPE=nvfp4 NEXTFORK_FP4_QSA=1 NEXTFORK_FP4_NOWS=1 ;;
    # 04.10, ответ критика: черновик и CUDA-графы (10 потоков, та же нагрузка)
    spec0)     variant spec0 10 0 POD_SPEC=0 ;;
    spec1)     variant spec1 10 0 POD_SPEC_STEPS=1 ;;
    spec2)     variant spec2 10 0 POD_SPEC_STEPS=2 ;;
    spec4)     variant spec4 10 0 POD_SPEC_STEPS=4 ;;
    cgall)     variant cgall 10 0 POD_CG_ALL=1 ;;
    base10)    variant base10 10 0 ;;
    prof)      PROFILE=1 variant prof 10 0 ;;   # профиль декодирования на боевой настройке
    # 04.10, второй ответ критика
    s11)       REPLAY_ARGS='--dup 2' variant s11 11 0 POD_MAXREQ=11 POD_MAMBA=60 POD_CG_ALL=1 ;;   # 11 мест, графы 1..11, слотов 60>=55
    s12)       REPLAY_ARGS='--dup 2' variant s12 12 0 POD_MAXREQ=12 POD_MAMBA=60 POD_CG_ALL=1 ;;
    s10d)      REPLAY_ARGS='--dup 2' variant s10d 10 0 ;;   # контроль на той же удвоенной нагрузке, что s11/s12
    moetri)    variant moetri 10 0 POD_MOE=triton ;;
    nooverlap) variant nooverlap 10 0 POD_NO_OVERLAP=1 ;;
    sgl0521)   SERVE_SCRIPT=$W/pod_franzen/serve0521.py variant sgl0521 10 1 DELTA_DIR=$W/wheels/sgl0521 ;;
    *) echo "неизвестный вариант $v" ;;
  esac
done
echo "===== МАТРИЦА ГОТОВА $(date -u +%T)"
