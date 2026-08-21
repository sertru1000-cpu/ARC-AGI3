"""Merge the trained adapter into bf16 and quantize to FP8 for vLLM/Kaggle.

Steps: bf16 base + LoRA -> merged bf16 -> FP8-dynamic (llmcompressor) ->
optional push of BOTH artifacts to a private HF repo.

Env knobs:
  BASE_MODEL (default Qwen/Qwen3.6-27B)
  ADAPTER    (default /workspace/out/adapter_final)
  MERGED     (default /workspace/student-bf16)
  FP8        (default /workspace/student-fp8)
  HF_REPO    (e.g. username/arc3-student-v1; empty = skip upload)
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen3.6-27B")
ADAPTER = os.getenv("ADAPTER", "/workspace/out/adapter_final")
MERGED = os.getenv("MERGED", "/workspace/student-bf16")
FP8 = os.getenv("FP8", "/workspace/student-fp8")

print("[1/3] merging adapter into bf16 base...")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(MERGED)
tok.save_pretrained(MERGED)
print("merged ->", MERGED)

print("[2/3] FP8-dynamic quantization (llmcompressor)...")
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=MERGED, recipe=recipe, output_dir=FP8)
tok.save_pretrained(FP8)
print("fp8 ->", FP8)

repo = os.getenv("HF_REPO", "").strip()
if repo:
    print("[3/3] uploading to HF:", repo)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=FP8, repo_id=repo, path_in_repo="fp8")
    api.upload_folder(folder_path=ADAPTER, repo_id=repo, path_in_repo="adapter")
    print("uploaded: fp8/ and adapter/ ->", repo)
else:
    print("[3/3] HF_REPO not set - skipping upload (artifacts stay on /workspace)")
