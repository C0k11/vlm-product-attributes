"""FastAPI service: product image (+ optional title) -> attribute JSON.

Runs the base model under vLLM with the LoRA adapters loaded in-process and
decodes under the label JSON schema, so every response is valid JSON with
labels from the closed sets. Run inside WSL:

    BASE_MODEL=~/models/Qwen__Qwen3.5-4B \
    LORA_IMAGE=outputs/train/lora_r16_short/final \
    LORA_TITLE=outputs/train/lora_title/final \
    uvicorn demo.api:app --host 0.0.0.0 --port 8000

LORA_TITLE is optional; without it, requests that include a title are answered
by the image-only adapter and the title is ignored.
"""
import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vpa.prompting import ATTRIBUTES, build_prompt, json_schema, parse_output  # noqa: E402

LONG_SIDE = 256  # training resolution

app = FastAPI(title="Product attribute extraction")
_state = {}
_lock = threading.Lock()


def _engine():
    if "llm" in _state:
        return _state
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    base = os.path.expanduser(os.environ["BASE_MODEL"])
    adapters = {}
    for i, key in enumerate(["LORA_IMAGE", "LORA_TITLE"], start=1):
        path = os.environ.get(key)
        if path and (ROOT / path / "adapter_config.json").exists():
            adapters[key] = LoRARequest(key.lower(), i, str((ROOT / path).resolve()))
    if "LORA_IMAGE" not in adapters:
        raise RuntimeError("LORA_IMAGE adapter not found")
    rank = max(json.load(open(Path(r.lora_path) / "adapter_config.json"))["r"] for r in adapters.values())
    llm = LLM(model=base, max_model_len=2048, gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.5")),
              limit_mm_per_prompt={"image": 1, "video": 0}, enable_lora=True, max_lora_rank=rank,
              max_loras=len(adapters), max_num_seqs=16, seed=0)
    product_types = json.load(open(ROOT / "data/processed/manifest.json"))["product_types"]
    _state.update(
        llm=llm, adapters=adapters, product_types=product_types,
        params=SamplingParams(temperature=0, max_tokens=64,
                              structured_outputs=StructuredOutputsParams(json=json_schema(product_types))),
    )
    return _state


@app.on_event("startup")
def _warm():
    _engine()


@app.get("/health")
def health():
    st = _engine()
    return {"status": "ok", "adapters": sorted(st["adapters"])}


@app.post("/extract")
async def extract(image: UploadFile = File(...), title: str | None = Form(None)):
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="not a readable image")
    img.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    st = _engine()
    title = (title or "").strip() or None
    use_title = title is not None and "LORA_TITLE" in st["adapters"]
    adapter = st["adapters"]["LORA_TITLE" if use_title else "LORA_IMAGE"]
    prompt = build_prompt("short", st["product_types"], title if use_title else None)
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}},
                                              {"type": "text", "text": prompt}]}]
    t0 = time.perf_counter()
    with _lock:
        out = st["llm"].chat([messages], st["params"], chat_template_kwargs={"enable_thinking": False},
                             use_tqdm=False, lora_request=adapter)
    latency_ms = (time.perf_counter() - t0) * 1000
    text = out[0].outputs[0].text
    obj, status = parse_output(text)
    if obj is None:
        raise HTTPException(status_code=500, detail=f"model output was not JSON: {text!r}")
    return {
        "attributes": {a: obj.get(a) for a in ATTRIBUTES},
        "adapter": adapter.lora_name,
        "used_title": use_title,
        "latency_ms": round(latency_ms, 1),
    }
