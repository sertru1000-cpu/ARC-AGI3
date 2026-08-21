@echo off
rem Multimodal teacher collection, AI-Studio-first with automatic Vertex fallback.
rem Primary: Google AI Studio prepaid key (LLM_API_KEY from .env LLM_API_KEY_AISTUDIO).
rem When it returns billing/quota errors (403 / repeated quota-429) the backend
rem flips process-wide to Vertex (ADC auth) and episodes continue uninterrupted.
rem Usage: scripts\run_teacher_vision_v2.cmd <label> <games> [max-turns] [extra args]
cd /d C:\Users\sertru1000\Projects\ARC-AGI-3
set PYTHONUTF8=1
for /f "tokens=2 delims==" %%k in ('findstr /b LLM_API_KEY_AISTUDIO .env') do set LLM_API_KEY=%%k
set LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
if "%MODEL%"=="" set MODEL=gemini-3.1-pro-preview
set LLM_MODEL=%MODEL%
set LLM_AUTH=key
set LLM_FALLBACK_BASE_URL=https://aiplatform.googleapis.com/v1/projects/gen-lang-client-0415838739/locations/global/endpoints/openapi
set LLM_FALLBACK_MODEL=google/%MODEL%
set LLM_FALLBACK_AUTH=gcloud-adc
set LLM_GCLOUD_PATH=C:/Users/sertru1000/.local/google-cloud-sdk/bin/gcloud.cmd
set LLM_TIMEOUT_S=300
set LLM_RETRIES=4
set LLM_MAX_TOKENS=8192
set LABEL=%~1
set GAMES=%~2
set TURNS=%~3
if "%PARALLEL%"=="" set PARALLEL=4
if "%MAX_ACTIONS%"=="" set MAX_ACTIONS=800
if "%TURNS%"=="" set TURNS=200
.venv\Scripts\python.exe scripts\collect_teacher.py --games %GAMES% --vision --parallel %PARALLEL% --label %LABEL% --max-turns %TURNS% --max-actions %MAX_ACTIONS% --game-seconds 14400 %4 %5 %6 %7 %8 > data\teacher\vision_%LABEL%.log 2>&1
