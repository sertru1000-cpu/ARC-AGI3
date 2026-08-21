#!/usr/bin/env bash
# Build a slim data/stand-kit.tgz for the RunPod experiment stand.
#
# Ships only what run_stand.py + my_agent.py actually import: no vendor
# .git, no tests/templates, no __pycache__. Vendor is trimmed to the one
# class we use (agents.agent.Agent) plus its two direct dependencies.
#
#   bash scripts/stand_kit/build_kit.sh                # all games
#   GAMES=lp85,sb26,wa30 bash scripts/stand_kit/build_kit.sh   # subset
set -euo pipefail
cd "$(dirname "$0")/../.."

STAGE="$(mktemp -d)"
KIT="$STAGE/stand-kit"
mkdir -p "$KIT/agent/harness" "$KIT/scripts" \
         "$KIT/vendor/ARC-AGI-3-Agents/agents" "$KIT/environment_files"

cp agent/__init__.py agent/my_agent.py "$KIT/agent/"
cp agent/harness/*.py "$KIT/agent/harness/"

cp scripts/run_stand.py scripts/run_oracle_probe.py "$KIT/scripts/"
cp scripts/stand_kit/serve_and_run.sh scripts/stand_kit/serve_vl_and_run.sh "$KIT/"

# Oracle-probe fuel: real successful teacher traces for our zero-level dead
# games (20.08 diagnostic) -- just the specific jsonl files, not whole run
# dirs (those also carry other games + a summary, pure bloat here).
mkdir -p "$KIT/data/teacher"
cp "data/teacher/google_gemini-3.1-pro-preview_20260817_192239/sb26.jsonl" "$KIT/data/teacher/sb26.jsonl"
cp "data/teacher/models_gemini-3.1-pro-preview_20260819_063307/ft09.jsonl" "$KIT/data/teacher/ft09.jsonl"
cp "data/teacher/google_gemini-3.1-pro-preview_20260817_192239/m0r0.jsonl" "$KIT/data/teacher/m0r0.jsonl"

# Import closure of agents/__init__.py: agent (+recorder, tracing), swarm,
# templates.random_agent. Verified 20.08 — missing swarm.py broke the pod run.
mkdir -p "$KIT/vendor/ARC-AGI-3-Agents/agents/templates"
cp vendor/ARC-AGI-3-Agents/agents/__init__.py \
   vendor/ARC-AGI-3-Agents/agents/agent.py \
   vendor/ARC-AGI-3-Agents/agents/recorder.py \
   vendor/ARC-AGI-3-Agents/agents/tracing.py \
   vendor/ARC-AGI-3-Agents/agents/swarm.py \
   "$KIT/vendor/ARC-AGI-3-Agents/agents/"
cp vendor/ARC-AGI-3-Agents/agents/templates/random_agent.py \
   "$KIT/vendor/ARC-AGI-3-Agents/agents/templates/"

if [ -n "${GAMES:-}" ]; then
  IFS=',' read -ra ids <<< "$GAMES"
  for g in "${ids[@]}"; do
    cp -r "environment_files/$g" "$KIT/environment_files/"
  done
else
  cp -r environment_files/. "$KIT/environment_files/"
fi

tar czf data/stand-kit.tgz --owner=0 --group=0 -C "$STAGE" stand-kit
rm -rf "$STAGE"

echo "built data/stand-kit.tgz:"
ls -la data/stand-kit.tgz
echo "entries: $(tar tzf data/stand-kit.tgz | wc -l)"
