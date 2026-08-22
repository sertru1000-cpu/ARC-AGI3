"""Build `notebooks/submission.ipynb` from agent/ sources.

Notebook layout (LLM-brain submission):
  1. Install `arc-agi` from the offline competition wheelhouse.
  2. Unpack agent sources (my_agent.py + harness/*) from an embedded bundle.
  3. Prepare the agents framework copy in /kaggle/working (slim __init__,
     drop in MyAgent + harness).
  4. Start a vLLM OpenAI server with Qwen3.6-27B-FP8 from the attached
     datasets (Duck's wheelhouse + model snapshot) and smoke-test it.
  5a. Phase A (commit): play two offline games with small budgets — a free
      full-stack dress rehearsal — then write the dummy submission.parquet.
  5b. Phase B (competition rerun): wait for the gateway and run the framework
      against the hidden games.

Env knobs at build time:
  KAGGLE_ACCEL=cpu|t4|p100|rtx6000  (default rtx6000 — required for the FP8 model)
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from textwrap import dedent

ACCELERATOR = os.getenv("KAGGLE_ACCEL", "rtx6000")

# Which brain the notebook ships: "qwen" (Duck's Qwen3.6-27B-FP8 via wheelhouse
# datasets) or "gptoss" (gpt-oss-120b from Kaggle Models, image-bundled vLLM —
# the official ARC template config, MoE => ~3-4x faster decoding).
BRAIN_MODEL = os.getenv("BRAIN_MODEL", "qwen").strip().lower()
GPTOSS_MODEL_REF = "danielhanchen/gpt-oss-120b/Transformers/default/1"
GPTOSS_MODEL_PATH = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"
# vLLM arrives via this utility-script kernel (adds itself to PYTHONPATH);
# the docker image is pinned to the one the official gpt-oss template uses.
GPTOSS_VLLM_DEPS_KERNEL = "philipvonderlind/vllm-deps"
GPTOSS_DOCKER_IMAGE = ("gcr.io/kaggle-images/python@sha256:"
                       "e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c")

_ACCELERATORS = {
    # machine_shape values are the server's canonical names (verified by
    # setting the accelerator in the Kaggle UI and pulling metadata back).
    "cpu":     {"name": "none",            "gpu": False, "shape": None},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True,  "shape": "NvidiaTeslaT4"},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True,  "shape": "NvidiaTeslaP100"},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True,  "shape": "NvidiaRtxPro6000"},
}

# Attached Kaggle datasets: Duck's public vLLM wheelhouse + Qwen snapshot.
WHEELHOUSE_REF = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
# Model dataset: Duck's base Qwen by default; our distilled student via env
# (BRAIN_MODEL=qwen MODEL_REF=sergueimakarov/arc3-student-v1-fp8 -> v23+).
MODEL_REF = os.getenv("MODEL_REF", "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "vrfai/Qwen3.6-27B-FP8")
# vLLM context window: Duck's VL base carries 64k; our text-only student's
# native max_position_embeddings is 32k (config-derived, vLLM refuses more).
model_max_len = os.getenv("MODEL_MAX_LEN", "65536")
# Runtime LoRA (Duck's own deployment pattern): serve the proven base and
# attach our student as an adapter. The wheelhouse vLLM has NO text-only
# qwen3_5 class, so a merged text checkpoint cannot be served at all —
# base+LoRA is the ONLY student path on this stack (learned via v23-v25).
LORA_REF = os.getenv("LORA_REF", "")
LORA_NAME = os.getenv("LORA_NAME", "student")
lora_ref, lora_name = LORA_REF, LORA_NAME  # f-string placeholders in the cell
# Vision (21.08): the model sees the current board as a 512px nearest-neighbor
# PNG each turn (VisionLLM). Teacher data was collected this way and the
# VL student was trained this way -- the image MUST be on at runtime or the
# student plays blind. VISION=0 reproduces the old text-only runs.
VISION = os.getenv("VISION", "1") == "1"
vision_flag = "1" if VISION else "0"
# Thinking: "on" = Duck/base behaviour (preserve_thinking); "off" = empty
# <think> block, exactly how the VL student was trained (stand 21.08).
THINKING = os.getenv("THINKING", "on").strip().lower()
chat_kwargs = '{"preserve_thinking": true}' if THINKING == "on" else '{"enable_thinking": false}'
# LLM_TEMPERATURE passthrough (harness default 0.6; first student runs used 0.0).
TEMPERATURE = os.getenv("TEMPERATURE", "")
# Phase A rehearsal game list (comma-separated ids).
SMOKE_GAMES = os.getenv("SMOKE_GAMES", "sk48,tn36,m0r0,bp35,ls20,ft09,sp80,vc33")
smoke_games = ", ".join(f'"{g.strip()}"' for g in SMOKE_GAMES.split(",") if g.strip())
# Phase A concurrency (22.08): games used to run one after another (8 x 720s
# = 96 min of quota per commit). vLLM batches requests and Phase B already
# runs 30 threads against it, so the rehearsal plays N games at once too —
# same pattern as scripts/run_stand.py. 1 = old sequential behaviour.
PHASE_A_PARALLEL = int(os.getenv("PHASE_A_PARALLEL", "8"))
# Wall-clock cap per rehearsal game. 720 mirrored the OLD sequential Phase B;
# with N games sharing one GPU a turn takes far longer (25-way, no thinking:
# 36 s/turn -> 9-49 turns in 720 s), so "did it reach 60 turns" needs more.
PHASE_A_GAME_SECONDS = int(os.getenv("PHASE_A_GAME_SECONDS", "720"))
# Turn/action caps. 60/400 was the old rehearsal budget; Phase B itself runs
# 250/800, so a "what would it score at submission budget" probe sets those.
# Whichever cap binds first ends the game -- keep GAME_SECONDS above
# MAX_TURNS x measured sec/turn or time silently truncates the run.
PHASE_A_MAX_TURNS = int(os.getenv("PHASE_A_MAX_TURNS", "60"))
PHASE_A_MAX_ACTIONS = int(os.getenv("PHASE_A_MAX_ACTIONS", "400"))

# ── Phase B (the real submission rerun) ──────────────────────────────────────
# Kaggle's hard limit for this competition is 9 h of notebook run-time (quoted
# from Code Requirements 22.08 -- third-party summaries claiming 6 h are wrong).
# TOTAL_SECONDS is the agent's own window and starts AFTER the stack is up, so
# the budget is: 9 h - startup (~15 min) - final write (~5 min) - margin.
# 28800 (8 h) leaves ~40 min, of which one stuck turn can eat LLM_TIMEOUT_S.
PHASE_B_TOTAL_SECONDS = int(os.getenv("PHASE_B_TOTAL_SECONDS", "28800"))
# Per-game cap. Must NOT be below TOTAL: games run concurrently and each may
# use the whole remaining window, so a lower value silently truncates long
# games (at 21600 vs a 28800 window every game died 2 h early).
PHASE_B_GAME_SECONDS = int(os.getenv("PHASE_B_GAME_SECONDS", str(PHASE_B_TOTAL_SECONDS)))
PHASE_B_MAX_TURNS = int(os.getenv("PHASE_B_MAX_TURNS", "250"))
PHASE_B_MAX_ACTIONS = int(os.getenv("PHASE_B_MAX_ACTIONS", "800"))
# Scorecards idle-expire after this; must outlive the longest game.
PHASE_A_STALE_MINUTES = max(60, PHASE_A_GAME_SECONDS // 60 + 30)

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"

COMP_INPUT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
FRAMEWORK_DST = "/kaggle/working/ARC-AGI-3-Agents"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def collect_sources() -> dict[str, str]:
    """my_agent.py + every harness module, keyed by target-relative path."""
    files = {"my_agent.py": (AGENT_DIR / "my_agent.py").read_text(encoding="utf-8")}
    for p in sorted((AGENT_DIR / "harness").glob("*.py")):
        files[f"harness/{p.name}"] = p.read_text(encoding="utf-8")
    return files


def build() -> dict:
    files = collect_sources()
    bundle_b64 = base64.b64encode(json.dumps(files).encode("utf-8")).decode("ascii")

    install_cell = code_cell(
        "!pip install --quiet --no-index --find-links \\\n"
        f"    {COMP_INPUT}/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    unpack_cell = code_cell(dedent(
        f"""\
        # Unpack agent sources (my_agent.py + harness/*) bundled at build time.
        import base64, json
        from pathlib import Path

        FILES = json.loads(base64.b64decode("{bundle_b64}").decode("utf-8"))
        root = Path("/tmp/agent_src")
        for rel, src in FILES.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src, encoding="utf-8")
        print(f"unpacked {{len(FILES)}} source files under {{root}}")
        """
    ))

    prepare_cell = code_cell(dedent(
        f"""\
        # Framework copy in a writable dir + our agent and harness dropped in.
        import shutil
        from pathlib import Path

        src = Path("{COMP_INPUT}/ARC-AGI-3-Agents")
        dst = Path("{FRAMEWORK_DST}")
        if not dst.exists():
            shutil.copytree(src, dst)
        shutil.copy("/tmp/agent_src/my_agent.py", dst / "agents/templates/my_agent.py")
        shutil.copytree("/tmp/agent_src/harness", dst / "harness", dirs_exist_ok=True)

        # Slim registry: upstream __init__ eagerly imports heavy templates.
        (dst / "agents/__init__.py").write_text('''from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{
            'random': Random,
            'myagent': MyAgent,
        }}
        '''.replace('\\n        ', '\\n'), encoding="utf-8")

        (dst / ".env").write_text('''SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        '''.replace('\\n        ', '\\n'), encoding="utf-8")
        print("framework prepared at", dst)
        """
    ))

    gptoss_vllm_cell = code_cell(dedent(
        f"""\
        # Start vLLM with gpt-oss-120b from Kaggle Models. vllm itself is NOT
        # bundled in the image -- the attached vllm-deps kernel only downloaded
        # the wheels (pip download, no install) into /kaggle/input/vllm-deps/wheels;
        # install them into a local target dir before importing vllm anywhere.
        import json, os, subprocess, sys, time, urllib.request
        from pathlib import Path

        MODEL_PATH = "{GPTOSS_MODEL_PATH}"
        SERVED = "gpt-oss-120b"
        VLLM_BASE_URL = "http://127.0.0.1:1234/v1"
        LOG = Path("/kaggle/working/vllm-server.log")
        SITE_PACKAGES = Path("/kaggle/working/vllm-site-packages")

        wheel_dir = None
        for root in (Path("/kaggle/input"),):
            for cand in root.rglob("*.whl"):
                if cand.name.startswith("vllm-"):
                    wheel_dir = cand.parent
                    break
        if wheel_dir is None:
            raise FileNotFoundError("no vllm-*.whl found under /kaggle/input — is vllm-deps attached?")
        if not (SITE_PACKAGES / ".installed").exists():
            SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                            "--find-links", str(wheel_dir), "--target", str(SITE_PACKAGES),
                            "--upgrade", "--ignore-installed", "--only-binary", ":all:",
                            "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts",
                            "vllm", "openai", "openai-harmony"], check=True)
            (SITE_PACKAGES / ".installed").write_text("ok")
        print("vllm wheels installed from", wheel_dir)

        # Offline tiktoken encodings for the harmony tokenizer.
        enc_dir = None
        for root in (Path("/kaggle/input"),):
            for cand in root.rglob("cl100k_base.tiktoken"):
                if (cand.parent / "o200k_base.tiktoken").exists():
                    enc_dir = cand.parent
                    break
        if enc_dir:
            os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(enc_dir)
            print("tiktoken encodings:", enc_dir)

        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", MODEL_PATH, "--served-model-name", SERVED,
               "--host", "127.0.0.1", "--port", "1234",
               "--max-num-seqs", "12", "--max-model-len", "64000",
               "--kv-cache-dtype", "fp8", "--tensor-parallel-size", "1",
               "--enforce-eager"]
        env = os.environ.copy()
        env["LIBRARY_PATH"] = "/usr/local/nvidia/lib64" + os.pathsep + env.get("LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64" + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        env["PYTHONPATH"] = str(SITE_PACKAGES) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(cmd, env=env, stdout=LOG.open("w"), stderr=subprocess.STDOUT)
        print("vLLM starting, pid", proc.pid)

        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(LOG.read_text()[-4000:])
                raise RuntimeError("vLLM died during startup")
            try:
                with urllib.request.urlopen(VLLM_BASE_URL + "/models", timeout=5) as r:
                    print("vLLM ready:", json.loads(r.read())["data"][0]["id"]); break
            except Exception:
                time.sleep(5)
        else:
            print(LOG.read_text()[-4000:])
            raise TimeoutError("vLLM not ready in 1800s")

        payload = {{"model": SERVED, "max_tokens": 200, "temperature": 0.0,
                   "messages": [{{"role": "user", "content": "Reply with one word: ready?"}}]}}
        reqo = urllib.request.Request(VLLM_BASE_URL + "/chat/completions",
                                      data=json.dumps(payload).encode(),
                                      headers={{"Content-Type": "application/json"}})
        with urllib.request.urlopen(reqo, timeout=180) as r:
            print("smoke reply:", json.loads(r.read())["choices"][0]["message"]["content"])

        os.environ.update({{
            "AGENT_BRAIN": "llm",
            "LLM_BASE_URL": VLLM_BASE_URL,
            "LLM_MODEL": SERVED,
            "LLM_TIMEOUT_S": "300",
            # Reasoning model: thinking + reply share this budget. 4096 left
            # ~350-token replies with zero verify_theory calls (v18); give
            # thinking real room and ask for deeper effort explicitly.
            "LLM_MAX_TOKENS": "16384",
            "LLM_REASONING_EFFORT": "high",
            "ONLY_RESET_LEVELS": "true",
        }})
        """
    ))

    vllm_cell = code_cell(dedent(
        f"""\
        # Start a vLLM OpenAI server with Qwen3.6-27B-FP8 (offline wheels + weights).
        import json, os, shutil, subprocess, sys, time, urllib.request
        from pathlib import Path

        WHEELHOUSE_REF = "{WHEELHOUSE_REF}"
        MODEL_REF = "{MODEL_REF}"
        SERVED_MODEL_NAME = "{SERVED_MODEL_NAME}"
        VLLM_BASE_URL = "http://127.0.0.1:1234/v1"
        WORKING = Path("/kaggle/working")
        SITE_PACKAGES = WORKING / "vllm-site-packages"
        LOG = WORKING / "vllm-server.log"

        def dataset_path(ref):
            owner, slug = ref.split("/", 1)
            for c in (Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug):
                if c.exists():
                    return c
            raise FileNotFoundError(f"dataset {{ref}} not mounted — attach it to the notebook")

        WHEELHOUSE = dataset_path(WHEELHOUSE_REF)
        MODEL_PATH = dataset_path(MODEL_REF)
        print("wheelhouse:", WHEELHOUSE)
        print("model:", MODEL_PATH)

        # Compat staging: checkpoints exported by transformers 5.x carry the
        # NEW arch name (model_type=qwen3_5_text) and keep the chat template
        # in a separate .jinja — the wheelhouse stack predates both. Build a
        # symlink dir with a patched config instead of re-uploading 29GB.
        import json as _json
        cfg = _json.loads((MODEL_PATH / "config.json").read_text())
        if cfg.get("model_type") == "qwen3_5_text":
            staged = WORKING / "model-staged"
            staged.mkdir(exist_ok=True)
            for f in MODEL_PATH.iterdir():
                dst = staged / f.name
                if not dst.exists() and f.name not in ("config.json", "tokenizer_config.json"):
                    os.symlink(f, dst)
            cfg["model_type"] = "qwen3_5"
            (staged / "config.json").write_text(_json.dumps(cfg, indent=2))
            tokcfg = _json.loads((MODEL_PATH / "tokenizer_config.json").read_text())
            tpl = MODEL_PATH / "chat_template.jinja"
            if "chat_template" not in tokcfg and tpl.exists():
                tokcfg["chat_template"] = tpl.read_text()
            (staged / "tokenizer_config.json").write_text(_json.dumps(tokcfg, indent=2))
            MODEL_PATH = staged
            print("compat-staged model at", MODEL_PATH)

        def vllm_env():
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SITE_PACKAGES) + os.pathsep + env.get("PYTHONPATH", "")
            # Kaggle GPU images keep libcuda off the default linker path; without
            # this, flashinfer's JIT link step dies with "cannot find -lcuda".
            env["LIBRARY_PATH"] = "/usr/local/nvidia/lib64" + os.pathsep + env.get("LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64" + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            env.update({{"USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                        "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1"}})
            return env

        req = WHEELHOUSE / "requirements.lock"
        if not (SITE_PACKAGES / ".installed").exists():
            SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                            "--find-links", str(WHEELHOUSE), "--requirement", str(req),
                            "--target", str(SITE_PACKAGES), "--upgrade", "--ignore-installed",
                            "--only-binary", ":all:", "--no-compile",
                            "--disable-pip-version-check", "--no-warn-conflicts"], check=True)
            (SITE_PACKAGES / ".installed").write_text("ok")

        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(MODEL_PATH), "--served-model-name", SERVED_MODEL_NAME,
               "--host", "127.0.0.1", "--port", "1234",
               "--tensor-parallel-size", "1", "--generation-config", "vllm",
               "--enable-prefix-caching", "--reasoning-parser", "qwen3",
               "--default-chat-template-kwargs", '{chat_kwargs}',
               "--max-model-len", "{model_max_len}"]
        if "{vision_flag}" == "1":
            cmd += ["--limit-mm-per-prompt", '{{"image": 1}}']
        RUNTIME_MODEL = SERVED_MODEL_NAME
        LORA_REF = "{lora_ref}"
        if LORA_REF:
            LORA_PATH = dataset_path(LORA_REF)
            cmd += ["--enable-lora", "--max-loras", "1", "--max-lora-rank", "16",
                    "--lora-modules", "{lora_name}=" + str(LORA_PATH)]
            RUNTIME_MODEL = "{lora_name}"
            print("runtime LoRA:", LORA_PATH, "->", RUNTIME_MODEL)
        proc = subprocess.Popen(cmd, env=vllm_env(),
                                stdout=LOG.open("w"), stderr=subprocess.STDOUT)
        print("vLLM starting, pid", proc.pid)

        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(LOG.read_text()[-4000:])
                raise RuntimeError("vLLM server died during startup")
            try:
                with urllib.request.urlopen(VLLM_BASE_URL + "/models", timeout=5) as r:
                    print("vLLM ready:", json.loads(r.read())["data"][0]["id"]); break
            except Exception:
                time.sleep(5)
        else:
            print(LOG.read_text()[-4000:])
            raise TimeoutError("vLLM server not ready in 900s")

        # Smoke completion.
        payload = {{"model": RUNTIME_MODEL, "max_tokens": 64, "temperature": 0.0,
                   "messages": [{{"role": "user", "content": "Reply with one word: ready?"}}]}}
        reqo = urllib.request.Request(VLLM_BASE_URL + "/chat/completions",
                                      data=json.dumps(payload).encode(),
                                      headers={{"Content-Type": "application/json"}})
        with urllib.request.urlopen(reqo, timeout=120) as r:
            print("smoke reply:", json.loads(r.read())["choices"][0]["message"]["content"])

        if "{vision_flag}" == "1":
            # Vision smoke: the server must SEE before we spend a game on it.
            import base64, io
            from PIL import Image
            img = Image.new("RGB", (512, 512), (255, 255, 255))
            for x in range(32, 96):
                for y in range(32, 96):
                    img.putpixel((x, y), (249, 60, 49))
            buf = io.BytesIO(); img.save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            payload = {{"model": RUNTIME_MODEL, "max_tokens": 64, "temperature": 0.0,
                       "messages": [{{"role": "user", "content": [
                           {{"type": "image_url", "image_url": {{"url": url}}}},
                           {{"type": "text", "text": "One sentence: what do you see and where?"}}]}}]}}
            reqo = urllib.request.Request(VLLM_BASE_URL + "/chat/completions",
                                          data=json.dumps(payload).encode(),
                                          headers={{"Content-Type": "application/json"}})
            with urllib.request.urlopen(reqo, timeout=180) as r:
                print("vision smoke:", json.loads(r.read())["choices"][0]["message"]["content"])

        # Environment for the agent (inherited by ! subprocesses too).
        os.environ.update({{
            "AGENT_BRAIN": "llm",
            "LLM_BASE_URL": VLLM_BASE_URL,
            "LLM_MODEL": RUNTIME_MODEL,
            "LLM_TIMEOUT_S": "300",
            # Qwen thinking + reply share this budget; at 4096 the verifier
            # checkpoint pressure made thinking eat it all -> empty replies
            # (v21: 4-5 empty turns per game, episodes died to strikes).
            "LLM_MAX_TOKENS": "16384",
            "ONLY_RESET_LEVELS": "true",
            "MY_AGENT_VISION": "{vision_flag}",
            "MY_AGENT_VISION_SCALE": "8",
        }})
        if "{TEMPERATURE}":
            os.environ["LLM_TEMPERATURE"] = "{TEMPERATURE}"
        """
    ))

    smoke_cell = code_cell(dedent(
        f"""\
        # Phase A (commit) only: dress rehearsal on offline games + dummy parquet.
        import os, sys, time
        from concurrent.futures import ThreadPoolExecutor

        if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            PARALLEL = {PHASE_A_PARALLEL}
            os.environ.update({{"MY_AGENT_MAX_ACTIONS": "{PHASE_A_MAX_ACTIONS}",
                               "MY_AGENT_MAX_TURNS": "{PHASE_A_MAX_TURNS}",
                               "MY_AGENT_GAME_SECONDS": "{PHASE_A_GAME_SECONDS}",
                               "STALE_MINUTES": "{PHASE_A_STALE_MINUTES}",  # keep scorecards alive for the whole rehearsal
                               "MY_AGENT_CROSS_MEMORY_PATH": "/kaggle/working/cross_memory.json",
                               "MY_AGENT_TRACE_DIR": "/kaggle/working/traces"}})
            if PARALLEL > 1:
                # Per-thread budgets against a shared vLLM, exactly like Phase B;
                # the LLM timeout gets Phase B's queueing headroom as well.
                os.environ.update({{"MY_AGENT_PARALLEL": "1", "LLM_TIMEOUT_S": "600"}})
            sys.path.insert(0, "{FRAMEWORK_DST}")
            import arc_agi
            from arc_agi import OperationMode
            from agents import MyAgent

            arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                                 environments_dir="{COMP_INPUT}/environment_files")
            # Rehearsal set (build-time override via SMOKE_GAMES env).
            # Historic dev set: sk48,tn36,m0r0,bp35,ls20,ft09,sp80,vc33.
            # Student exams add lp85/sb26/wa30 — solvable games excluded
            # from the SFT dataset, so they measure method transfer.
            DEV = {{{smoke_games}}}
            games = [e.game_id for e in arc.available_environments
                     if e.game_id.split("-")[0] in DEV]
            print(f"offline smoke on: {{games}} ({{PARALLEL}} at a time)")

            def play(gid):
                env = arc.make(gid)
                agent = MyAgent(card_id="phaseA", game_id=gid, agent_name=f"smoke.{{gid}}",
                                ROOT_URL="http://localhost", record=False, arc_env=env,
                                tags=["phaseA-smoke"])
                t0 = time.time()
                try:
                    agent.main()
                except Exception as exc:  # one broken game must not sink the rehearsal
                    print(f"{{gid}}: CRASHED {{exc!r}} ({{time.time()-t0:.0f}}s)")
                    return gid, agent
                f = agent.frames[-1]
                print(f"{{gid}}: levels={{f.levels_completed}}/{{f.win_levels}} "
                      f"actions={{agent.action_counter}} state={{f.state}} "
                      f"({{time.time()-t0:.0f}}s)", flush=True)
                return gid, agent

            t_all = time.time()
            with ThreadPoolExecutor(max_workers=max(1, PARALLEL)) as pool:
                played = list(pool.map(play, games))
            print(f"rehearsal wall time: {{time.time()-t_all:.0f}}s")

            for gid, agent in played:
                # Self-computed RHAE from the agent's own frame trace — no
                # dependence on the Arcade's internal scorecard plumbing.
                try:
                    from arc_agi.scorecard import Card, Scorecard, EnvironmentScorecard
                    from arcengine import GameState as GS
                    card = Card(game_id=gid)
                    run = -1
                    prev_state = ""
                    for e_ in agent.replay_log:
                        if e_["id"] == 0:
                            # Full reset only at session start or from WIN
                            if run < 0 or prev_state.endswith("WIN"):
                                run += 1
                                card.inc_play_count(f"r{{run}}")
                            else:
                                card.inc_reset_count(f"r{{run}}")
                        else:
                            if run < 0:
                                run = 0
                                card.inc_play_count("r0")
                            card.inc_action_count(f"r{{run}}")
                        gd = f"r{{run}}"
                        card.set_levels_completed(gd, int(e_["level"]))
                        st = e_["state"].split(".")[-1]
                        if st == "WIN":
                            card.set_state(gd, GS.WIN)
                        elif st == "GAME_OVER":
                            card.set_state(gd, GS.GAME_OVER)
                        prev_state = e_["state"]
                    sc2 = Scorecard(card_id="x", api_key="k", cards={{gid: card}})
                    esc = EnvironmentScorecard.from_scorecard(sc2, arc.available_environments)
                    if esc.environments:
                        e = esc.environments[0]
                        msgs = {{r.message for r in e.runs if r.message}}
                        print(f"  RHAE {{gid}}: {{e.score:.2f}} runs={{[round(r.score, 2) for r in e.runs]}}"
                              + (f" note={{msgs}}" if msgs else ""))
                    else:
                        print(f"  RHAE {{gid}}: no card built (frames={{len(agent.frames)}})")
                except Exception as exc:
                    print("  RHAE self-compute failed:", repr(exc))

            # Local RHAE scorecard: the engine scores public games against real
            # human baselines — the same math the leaderboard uses.
            try:
                sc = arc.get_scorecard()
                for e in sc.environments:
                    print(f"RHAE score {{e.id}}: {{e.score:.2f}} "
                          f"(runs: {{[round(r.score, 2) for r in e.runs]}})")
                print(f"RHAE aggregate over played games: {{sc.score:.3f}}")
            except Exception as exc:
                print("scorecard unavailable:", exc)

            import pandas as pd
            pd.DataFrame([["1_0", "1", True, 1]],
                         columns=["row_id", "game_id", "end_of_game", "score"]
                         ).to_parquet("/kaggle/working/submission.parquet", index=False)
            print("dummy submission.parquet written")
        """
    ))

    rerun_cell = code_cell(dedent(
        f"""\
        # Phase B (competition rerun): play the hidden games via the gateway.
        import os

        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            # Swarm runs every hidden game in its OWN THREAD, concurrently.
            # Budgets are therefore per-thread against a SHARED wall window:
            # the old 720s/game cap silently ended the whole run in ~12 min.
            os.environ.update({{"MY_AGENT_PARALLEL": "1",
                               "MY_AGENT_MAX_ACTIONS": "{PHASE_B_MAX_ACTIONS}",
                               "MY_AGENT_MAX_TURNS": "{PHASE_B_MAX_TURNS}",
                               "MY_AGENT_GAME_SECONDS": "{PHASE_B_GAME_SECONDS}",
                               "MY_AGENT_TOTAL_SECONDS": "{PHASE_B_TOTAL_SECONDS}",  # shared window, 9h Kaggle cap
                               "MY_AGENT_EXPECTED_GAMES": "30",
                               "LLM_TIMEOUT_S": "600",  # 30-way queueing headroom
                               "MY_AGENT_CROSS_MEMORY_PATH": "/kaggle/working/cross_memory.json",
                               "MY_AGENT_TRACE_DIR": "/kaggle/working/traces"}})
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games
            !cd {FRAMEWORK_DST} && MPLBACKEND=agg python main.py --agent myagent
        """
    ))

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(f"Unknown ACCELERATOR={ACCELERATOR!r}")
    accel = _ACCELERATORS[ACCELERATOR]

    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python",
                              "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                f"# ARC Prize 2026 — ARC-AGI-3 Submission (brain: "
                f"{'gpt-oss-120b' if BRAIN_MODEL == 'gptoss' else 'Qwen3.6-27B'} via vLLM)\n\n"
                "Built from `agent/` via `scripts/build_notebook.py`. Do not edit cells "
                "directly — edit the sources and rebuild."
            ),
            install_cell,
            unpack_cell,
            prepare_cell,
            gptoss_vllm_cell if BRAIN_MODEL == "gptoss" else vllm_cell,
            smoke_cell,
            rerun_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  (accelerator: {ACCELERATOR})")

    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        changed = False
        wanted_gpu = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted_gpu:
            meta["enable_gpu"] = wanted_gpu
            changed = True
        if BRAIN_MODEL == "gptoss":
            wanted_ds, wanted_models = [], [GPTOSS_MODEL_REF]
            wanted_kernels = [GPTOSS_VLLM_DEPS_KERNEL]
            wanted_docker = GPTOSS_DOCKER_IMAGE
        else:
            wanted_ds, wanted_models = [WHEELHOUSE_REF, MODEL_REF], []
            if LORA_REF:
                wanted_ds.append(LORA_REF)
            wanted_kernels = []
            wanted_docker = None
        if meta.get("dataset_sources") != wanted_ds:
            meta["dataset_sources"] = wanted_ds
            changed = True
        if meta.get("model_sources") != wanted_models:
            meta["model_sources"] = wanted_models
            changed = True
        if meta.get("kernel_sources") != wanted_kernels:
            meta["kernel_sources"] = wanted_kernels
            changed = True
        # None must REMOVE a stale pin, not keep it: the gptoss template image
        # lacks nvcc, and the qwen path needs it for flashinfer's JIT (v20
        # died on exactly this — 'nvcc: not found' during gemm_sm120 build).
        if wanted_docker is None:
            if "docker_image" in meta:
                del meta["docker_image"]
                changed = True
        elif meta.get("docker_image") != wanted_docker:
            meta["docker_image"] = wanted_docker
            changed = True
        wanted_shape = _ACCELERATORS[ACCELERATOR]["shape"]
        if meta.get("machine_shape") != wanted_shape:
            meta["machine_shape"] = wanted_shape
            changed = True
        if changed:
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"[build_notebook] Synced kernel metadata (gpu={wanted_gpu}, datasets={wanted_ds})")


if __name__ == "__main__":
    main()
