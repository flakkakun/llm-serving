"""Serve an open-source LLM with vLLM on a Modal GPU.

Exposes vLLM's OpenAI-compatible API (/v1/...) and its Prometheus metrics (/metrics).

    modal serve serve.py    # dev: live URL, stops when you Ctrl+C
    modal deploy serve.py   # stays deployed; scales to zero when idle
"""

import os
import subprocess

import modal

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
SERVED_NAME = "llm"
GPU = os.environ.get("GPU", "L4")  # 24 GB, ~$0.80/hr on Modal
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM 0.10.2 predates transformers 5 / huggingface_hub 1.x, so pin below them.
    .uv_pip_install("vllm==0.10.2", "transformers<5", "huggingface_hub[hf_transfer]<1.0")
    # Bake the settings into the image so the container sees what you set locally.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "MODEL": MODEL, "MAX_MODEL_LEN": str(MAX_MODEL_LEN)})
)

# Cache model weights and vLLM's compile artifacts so cold starts skip the download.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("llm-serving")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("vllm-api-key")],
    max_containers=1,  # one GPU at most, so a traffic spike can't burn credits
    scaledown_window=3 * 60,  # shut the GPU down after 3 idle minutes
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=256)  # vLLM batches concurrent requests itself
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--api-key", os.environ["VLLM_API_KEY"],  # guards /v1; /metrics stays open
    ]
    subprocess.Popen(cmd)
