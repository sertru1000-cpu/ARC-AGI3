@echo off
rem Detached multimodal teacher collection (vision round 1, 21.08.2026).
rem 4 resistant games (pure-Gemini ceiling = 0 levels), no human hints,
rem production-like budget: 200 turns / 800 actions, 4 played concurrently.
cd /d C:\Users\sertru1000\Projects\ARC-AGI-3
set PYTHONUTF8=1
set LLM_BASE_URL=https://aiplatform.googleapis.com/v1/projects/gen-lang-client-0415838739/locations/global/endpoints/openapi
set LLM_MODEL=google/gemini-3.1-pro-preview
set LLM_AUTH=gcloud-adc
set LLM_GCLOUD_PATH=C:/Users/sertru1000/.local/google-cloud-sdk/bin/gcloud.cmd
set LLM_API_KEY=vertex
set LLM_TIMEOUT_S=300
set LLM_RETRIES=4
set LLM_MAX_TOKENS=8192
.venv\Scripts\python.exe scripts\collect_teacher.py --games sk48,tn36,bp35,cd82 --vision --parallel 4 --label round1 --max-turns 200 --max-actions 800 --game-seconds 14400 > data\teacher\vision_round1.log 2>&1
