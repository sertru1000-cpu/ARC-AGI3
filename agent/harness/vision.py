"""Multimodal (vision) input for the teacher: the current board as a PNG.

Why: the text-only teacher (Gemini 3.1 Pro) plateaued at 0 levels on a set
of public games even with the verifier harness; Duck's 27B SEES the board
(vision_context.py) and that is part of its edge. This module renders the
current 64x64 frame to an image and a `VisionLLM` wrapper attaches it to
the LLM request **at call time only**:

  - the policy's message history stays pure text (what the student will see
    at inference -- the student is trained text-only, no VL base);
  - the JSONL trace never contains the image, so `build_sft_dataset.py`
    reconstructs the exact text prompt without any change;
  - only the CURRENT frame is attached (Duck does the same) -- history frames
    would multiply cost for little value.

Rendering rules (matter for a 16-color discrete palette):
  - palette = Duck's ARC_COLOR_MAP, which matches three.arcprize.org, so the
    teacher sees the same colors a human player does;
  - upscale by an integer factor with NEAREST-NEIGHBOR only. Bilinear/bicubic
    would blend neighbouring colors into gradients that exist nowhere in the
    game and blur object boundaries. Default x8 -> 512x512: Gemini bills one
    tile (258 tokens) for anything up to 768px on a side, so 512 costs the
    same as 256 but is crisper.

Wire format: OpenAI-compatible multipart content
  [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
   {"type": "text", "text": ...}]
which Vertex AI's OpenAI-compatible endpoint accepts for Gemini (equivalent
of `types.Part.from_bytes` in the google-genai SDK, without a new dependency).

Env knobs (read by my_agent.py, not here):
  MY_AGENT_VISION=1          attach the image (default off)
  MY_AGENT_VISION_SCALE=8    integer upscale factor
"""
from __future__ import annotations

import base64
import copy
import io
from typing import Callable

import numpy as np

from .llm import LLMBackend, Message

# Same palette Duck uses (reference/duck-source/.../vision_context.py) --
# matches the official web player, index = grid value 0..15.
ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}

_PALETTE = np.array([ARC_COLOR_MAP[i] for i in range(16)], dtype=np.uint8)


def grid_to_png_bytes(grid: np.ndarray, scale: int = 8) -> bytes:
    """Render a HxW int grid (values 0..15) to PNG bytes, upscaled by an
    integer factor with nearest-neighbor (block replication, no blending)."""
    from PIL import Image

    g = np.asarray(grid, dtype=np.int64) % 16
    rgb = _PALETTE[g]  # (H, W, 3)
    scale = max(1, int(scale))
    if scale > 1:
        # np.repeat on both axes == exact nearest-neighbor block upscale.
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def grid_to_data_url(grid: np.ndarray, scale: int = 8) -> str:
    b64 = base64.b64encode(grid_to_png_bytes(grid, scale)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_caption(grid: np.ndarray, scale: int) -> str:
    h, w = grid.shape
    return (
        f"[image] The PNG above is the CURRENT board ({h}x{w} cells, each cell "
        f"drawn as a {scale}x{scale} pixel block, same 16-color palette as the "
        "ascii symbols 0-F in current_frame.ascii; pixel (px,py) -> cell "
        f"(x=px//{scale}, y=py//{scale})). Use it to see shapes, zones and "
        "layout at a glance; use the grid/ascii/segmentation in code for exact "
        "coordinates."
    )


class VisionLLM(LLMBackend):
    """Wraps any backend; attaches the current board image to each request.

    `grid_provider()` returns the current grid (or None -> plain text call).
    The caller's `messages` list is never mutated -- a shallow copy with a
    rebuilt last message is sent instead, so history/trace stay text-only.
    """

    def __init__(self, inner: LLMBackend, grid_provider: Callable[[], np.ndarray | None],
                 scale: int = 8):
        self.inner = inner
        self.grid_provider = grid_provider
        self.scale = max(1, int(scale))
        self.images_sent = 0

    def _augment(self, messages: list[Message]) -> list[Message]:
        grid = self.grid_provider()
        if grid is None:
            return messages
        part_img = {"type": "image_url", "image_url": {"url": grid_to_data_url(grid, self.scale)}}
        caption = image_caption(grid, self.scale)
        out = list(messages)  # shallow: untouched messages are shared, not copied
        last = out[-1] if out else None
        if last is not None and last.get("role") == "user" and isinstance(last.get("content"), str):
            out[-1] = {
                **copy.copy(last),
                "content": [part_img, {"type": "text", "text": caption + "\n\n" + last["content"]}],
            }
        else:
            # Last message is assistant/tool (tool-loop): tool messages can't
            # carry images, so add a separate user turn with the picture.
            out.append({"role": "user", "content": [part_img, {"type": "text", "text": caption}]})
        self.images_sent += 1
        return out

    def chat(self, messages: list[Message], max_tokens: int = 2048, temperature: float = 0.6) -> str:
        return self.inner.chat(self._augment(messages), max_tokens=max_tokens, temperature=temperature)

    def chat_tools(self, messages: list[Message], tools: list[dict], max_tokens: int = 2048,
                   temperature: float = 0.6, tool_choice: str = "auto") -> dict:
        return self.inner.chat_tools(self._augment(messages), tools, max_tokens=max_tokens,
                                     temperature=temperature, tool_choice=tool_choice)

    @property
    def name(self) -> str:
        return f"Vision(x{self.scale})+{self.inner.name}"
