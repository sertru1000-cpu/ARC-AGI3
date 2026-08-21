@echo off
rem Detached teacher collection — survives Claude Code session restarts.
cd /d C:\Users\sertru1000\Projects\ARC-AGI-3
set PYTHONUTF8=1
set LLM_BASE_URL=https://aiplatform.googleapis.com/v1/projects/gen-lang-client-0415838739/locations/global/endpoints/openapi
set LLM_MODEL=google/gemini-3.1-pro-preview
set LLM_AUTH=gcloud-adc
set LLM_GCLOUD_PATH=C:/Users/sertru1000/.local/google-cloud-sdk/bin/gcloud.cmd
set LLM_API_KEY=vertex
set LLM_TIMEOUT_S=240
set LLM_RETRIES=4
set MY_AGENT_MAX_ACTIONS=400
.venv\Scripts\python.exe scripts\collect_teacher.py --games lp85,ar25,ft09,m0r0,sc25,tu93,ka59,ls20,re86,s5i5,sb26,su15,vc33,wa30,dc22,lf52 --max-turns 40 --game-seconds 1800 > data\teacher\verifier_round.log 2>&1
