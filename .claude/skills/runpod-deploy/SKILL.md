---
name: runpod-deploy
description: "Before renting/creating ANY RunPod GPU pod for this project, or deploying our harness/vLLM onto one. Load this FIRST, before touching the RunPod web UI or API. Prevents repeating 27.08's ~2 hours of wasted A100 time across 5 failed pod-creation attempts in a row."
metadata:
  author: user
  version: "1.0.0"
---

# RunPod deploy — do this, not that

**Why this exists:** 27.08, one afternoon, five pod-creation attempts in a row failed for five different infrastructure reasons before one finally worked, burning ~2 hours of paid A100 time on pure setup friction, zero harness testing accomplished. User's own words twice: "каждый раз мы теряем по 2 часа" and "Деньги только тратим." This skill exists so the NEXT pod is right on the first try.

## The one decision that matters: how to create the pod

**Use the RunPod WEB UI with a plain "RunPod Pytorch 2.8.0" template. Do NOT use the official `vllm/vllm-openai` Docker image, via ANY method, until someone fixes the SSH problem below outside a paid session.**

This is a reversal of earlier same-day advice ("switch to the vllm-openai image, it avoids the whole slow-install class of bug") — that diagnosis was CORRECT (the official image really does avoid the install-speed problem) but the SSH access story around it is broken in every way tried on 27.08:
- Created via raw REST API (`imageName` only, no `templateId`): SSH refused, both direct IP:port and the `ssh.runpod.io` proxy ("container not found"). RunPod's SSH-enabling agent appears to be injected by a Pod TEMPLATE, not by the platform independent of the image — a bare `imageName`+API call skips it entirely.
- Created via API with `dockerEntrypoint`/`dockerStartCmd` overridden to neutralize the image's own crash-on-no-model default: SAME failure, entirely broken SSH. Do not try this "fix" again.
- Created via the web UI's own vLLM-shaped deploy template (found by exploring the UI, not a raw custom image): SSH worked via the proxy the FIRST time, but (a) it used `vllm/vllm-openai:latest` (v0.28.0) with no visible way to pin an exact version tag, and (b) the container later crashed and would not come back even after an API-triggered restart.
- Net effect: the official image path is not currently a reliable choice for this project's workflow at all, regardless of method. Revisit it only as a deliberate, NOT-on-the-meter investigation (find/build a proper custom Template with SSH bootstrap baked in) — see the bottom of this file for what that would take.

**What DOES reliably work, confirmed multiple times 26-27.08:** a plain "RunPod Pytorch 2.8.0" template pod (1x A100 SXM 80GB, via the web UI) always gave working SSH, both times it was tried. Use this.

## Steps, in order

1. **Tell the user exactly what to pick in the web UI**: template "RunPod Pytorch 2.8.0" (or whatever the current equivalent default PyTorch template is called), GPU "1x A100 SXM" / 80GB VRAM. Do not attempt to specify a Docker image or override any start command — the default template already works.
2. **Get the SSH connection string from the user** (`ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519` — a direct IP:port, not the `ssh.runpod.io` proxy form; the PyTorch template exposes a direct port). Verify: `nvidia-smi --query-gpu=name,memory.total --format=csv` and `df -h / /workspace`.
3. **scp the deploy script and the Kaggle access token** — plain `scp` works fine on this template (unlike the vllm-openai proxy path, which rejects the SFTP subsystem entirely):
   ```
   scp -P <port> -i ~/.ssh/id_ed25519 scripts/runpod_a100_20x3h_run.sh root@<ip>:/root/
   scp -P <port> -i ~/.ssh/id_ed25519 ~/.kaggle/access_token root@<ip>:/root/.kaggle/access_token
   ```
   Use `scripts/runpod_a100_20x3h_run.sh` (or its current equivalent) — it already contains the `UV_LINK_MODE=symlink` fix (see below). **Do not use `scripts/runpod_a100_vllmimg_20x3h_run.sh`** — it targets the broken image-based path above.
4. **Launch it backgrounded and poll the log** — `nohup bash /root/runpod_a100_20x3h_run.sh > /root/deploy.log 2>&1 &`. Confirm the install step specifically completed FAST (`grep -n 'Installed .* packages' /root/deploy.log` should show a number of seconds, not tens of minutes — 26.08's numbers were ~30+ min before the symlink fix, 15.81s after it). If you see `warning: Failed to hardlink files; falling back to full copy` in the log, the symlink fix did not take — STOP and check the script actually has `export UV_LINK_MODE=symlink` before the `uv pip install torch==... vllm==...` line.
5. **When done: Terminate the pod (not Stop)**, then check `GET https://rest.runpod.io/v1/networkvolumes` for anything orphaned and ask the user before deleting (real per-GB-per-month billing regardless of pod state).

## The `UV_LINK_MODE=symlink` fix (already in the current script — verify, don't re-derive)

uv's cache (`/root/.cache/uv`, local disk) and the venv (`/workspace/venv312`, network volume) are on different filesystems. uv's default hardlink install mode silently falls back to a full byte-for-byte copy across them — and torch alone ships thousands of tiny header files under `include/ATen/ops/`, so the fallback crawls at ~7KB/s (confirmed live via `lsof`/`/proc/<pid>/io`, not a hang, just genuinely that slow on small files over network storage). The fix is `export UV_LINK_MODE=symlink` right before the `uv pip install` calls — symlinks work across filesystems (unlike hardlink) and are metadata-only (unlike copy). **Do NOT "fix" this by moving `UV_CACHE_DIR` onto the network volume instead** — that reintroduces the unkillable-D-state risk `PIP_CACHE_DIR` is deliberately kept off the network volume to avoid (see the comment already in the script).

## If revisiting the vllm-openai image later (not on a paid pod)

The actual fix is almost certainly: build/save a proper RunPod Template (not a bare `imageName`) that wraps `vllm/vllm-openai:v0.19.0` with an init step that installs+starts sshd and injects the `PUBLIC_KEY` env into `authorized_keys` — i.e., reproduce what RunPod's own PyTorch templates do automatically. `POST /v1/templates` exists in the REST API for this. Do this kind of exploration in a free/idle moment, verified end-to-end, BEFORE ever proposing it again as the plan for a paid run.

## Related memory (fuller narrative + the exact API field reference)

[[arc-agi-3-runpod-pod-creation-checklist]] (REST API field-by-field reference, if attempting the API path again), [[arc-agi-3-runpod-lessons]] (full incident history), [[feedback-audit-deploy-scripts-fully]] (the "audit the whole risk category" lesson).
