# LLM Serving: vLLM on a Cloud GPU, with Latency Monitoring

This project deploys an open-source LLM (Qwen2.5-7B-Instruct) behind an OpenAI-compatible API using **vLLM** on a **Modal** GPU. It then measures the three numbers that matter in production:

| Metric | What it means | Why it matters |
|---|---|---|
| **TTFT** (time to first token) | Time from sending a request to the first word coming back | How fast the app *feels* |
| **ITL** (inter-token latency) | Gap between each word as it streams | How smoothly the answer reads |
| **Throughput** | Total tokens per second across all users | How many users one GPU can serve, which sets the cost |

These numbers pull against each other. Adding concurrent users raises throughput, because vLLM batches requests together, but it also slows each user's TTFT and ITL. The benchmark below finds where that trade-off lands on one GPU.

## How it works

```
bench.py (your laptop) ──HTTPS──▶ Modal L4 GPU: vLLM server ──▶ /v1/chat/completions
                                                            └─▶ /metrics (Prometheus format)
Prometheus + Grafana (Docker, your laptop) ─────scrapes─────────┘
```

- `serve.py`: the Modal app. It runs `vllm serve` on an L4 (24 GB) with an API key, caches model weights in a volume so cold starts are faster, and allows at most one GPU, which scales to zero after 3 idle minutes.
- `bench.py`: an async load tester. It streams requests at increasing concurrency and records TTFT, ITL and throughput from the client side (p50/p90/p99).
- `monitoring/`: Prometheus scrapes vLLM's server-side metrics, and Grafana shows a ready-made dashboard: TTFT, ITL, throughput, running and waiting requests, KV-cache use, and end-to-end latency.

## Run it

You need a Modal account (the free plan includes $30/month of credit) and Docker for the dashboard.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
modal setup                      # log in to Modal in the browser

# API key: keep it in .env locally, and in a Modal secret for the server
cp .env.example .env             # then put a random key in VLLM_API_KEY (e.g. from: openssl rand -hex 16)
modal secret create vllm-api-key VLLM_API_KEY=<the same key>

modal deploy serve.py            # prints your URL; put it in .env as VLLM_URL
                                 # (GPU=A10 modal deploy serve.py picks another GPU if L4s are busy)
python bench.py --label "L4 qwen2.5-7b fp16"
```

Dashboard: put your Modal host in `monitoring/prometheus.yml`, then run `docker compose -f monitoring/docker-compose.yml up -d` and open http://localhost:3000.

**Keeping costs down:** Prometheus scrapes every 5 seconds, and each scrape keeps the GPU awake. Run `docker compose -f monitoring/docker-compose.yml down` when you finish a benchmark, and use `modal app stop llm-serving` to shut everything down.

## Results

**Setup:** Qwen2.5-7B-Instruct (fp16, about 15 GB), vLLM 0.10.2, one NVIDIA A10 (24 GB) on Modal. Each request generates 256 tokens. The client is a laptop calling Modal's us-east region over the internet, so TTFT includes that round trip and Modal's proxy. Raw data: [`results/20261030-034259.json`](results/20261030-034259.json).

| Concurrent users | TTFT p50 | TTFT p99 | Time per token p50 | Speed per user | Total throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 459 ms | 835 ms | 33.6 ms | 29.8 tok/s | 28 tok/s |
| 4 | 466 ms | 1,746 ms | 34.8 ms | 28.8 tok/s | 106 tok/s |
| 16 | 504 ms | 2,878 ms | 37.6 ms | 26.6 tok/s | 383 tok/s |
| 32 | 542 ms | 4,711 ms | 39.2 ms | 25.5 tok/s | 710 tok/s |

**What it shows:**
- **Batching is almost free here.** Going from 1 to 32 users raised total throughput **25x** (28 → 710 tok/s), while each user got only **14% slower** (29.8 → 25.5 tok/s). That's continuous batching and PagedAttention at work.
- **Decode is memory-bound, as predicted.** The A10 reads memory at about 600 GB/s, so streaming 15 GB of weights sets a floor of about 25 ms per token. We measured 33.6 ms for one user, roughly 75% of the hardware limit.
- **The cost of tail latency.** Median TTFT barely moves, but p99 climbs from 0.8 s to 4.7 s. When 32 requests land at once, some have to wait while other prompts are prefilled. With 32 users we haven't hit the knee of the curve yet, so the next run should push to 64 and 128.
- **Cost:** at about $1.10/hr for an A10 and 710 tok/s, that's about **$0.43 per million output tokens** at 32 users.

**Measurement note:** Modal's proxy sometimes delivers streamed tokens in bursts, so the gaps between individual tokens seen on the client are noisy. The table therefore reports time per output token, which is decode time averaged over each response. vLLM's own `vllm:inter_token_latency_seconds` metric gives the server-side view in Grafana.

## Experiments planned

- [x] Baseline: fp16 on an A10, concurrency 1 → 32
- [ ] Push concurrency to 64 / 128 to find the knee
- [ ] Same run on an L4 (cheaper, half the memory bandwidth)
- [ ] Quantised model (AWQ): speed vs. memory
- [ ] Bigger GPU (A100/H100) for one comparison run
- [ ] SGLang vs. vLLM on the same model and GPU
