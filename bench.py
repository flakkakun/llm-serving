"""Load-test a vLLM server and measure TTFT, inter-token latency and throughput.

    python bench.py --url https://<you>--llm-serving-serve.modal.run --concurrency 1 4 16 32

Every request streams, so the timing is measured from the client side:
  TTFT        time from sending the request to the first generated token
  ITL         gap between consecutive streamed tokens
  throughput  total output tokens per second across all in-flight requests
Results go to results/<timestamp>.json plus a summary table on stdout.
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx
import numpy as np

TOPICS = [
    "how a CPU cache works", "the history of the printing press", "why the sky is blue",
    "how vaccines train the immune system", "the rules of chess", "how a bill becomes law",
    "photosynthesis", "how TCP handles packet loss", "the causes of inflation",
    "how airplanes stay in the air", "plate tectonics", "how a database index works",
]


def load_env():
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


async def one_request(client, url, key, max_tokens):
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": f"Explain {random.choice(TOPICS)} in detail."}],
        "max_tokens": max_tokens,
        "ignore_eos": True,  # vLLM extension: always generate max_tokens, so runs are comparable
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    token_times, output_tokens = [], 0
    async with client.stream(
        "POST", f"{url}/v1/chat/completions", json=payload,
        headers={"Authorization": f"Bearer {key}"},
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                output_tokens = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                token_times.append(time.perf_counter())
    end = time.perf_counter()
    if not token_times:
        raise RuntimeError("no tokens streamed")
    n = max(output_tokens, len(token_times))
    return {
        "ttft": token_times[0] - start,
        "itl": np.diff(token_times).tolist(),
        # Proxies can deliver streamed chunks in bursts, which distorts individual gaps;
        # time per output token averages decode time over the whole response instead.
        "tpot": (token_times[-1] - token_times[0]) / (n - 1) if n > 1 else 0.0,
        "latency": end - start,
        "output_tokens": output_tokens,
    }


async def run_level(url, key, concurrency, num_requests, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    results, errors = [], 0

    async def worker(client):
        nonlocal errors
        async with sem:
            try:
                results.append(await one_request(client, url, key, max_tokens))
            except Exception as e:  # count failures rather than abort the whole level
                errors += 1
                print(f"  request failed: {e}")

    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(num_requests)))
        wall = time.perf_counter() - start

    ttft = np.array([r["ttft"] for r in results]) * 1000
    itl = np.array([g for r in results for g in r["itl"]]) * 1000
    tpot = np.array([r["tpot"] for r in results]) * 1000
    tokens = sum(r["output_tokens"] for r in results)

    def pct(a):
        return {f"p{p}": round(float(np.percentile(a, p)), 1) for p in (50, 90, 99)} if len(a) else {}

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": errors,
        "wall_s": round(wall, 2),
        "ttft_ms": pct(ttft),
        "itl_ms": pct(itl),
        "tpot_ms": pct(tpot),
        "per_user_tok_per_s": round(1000 / float(np.median(tpot)), 1) if len(tpot) else None,
        "output_tok_per_s": round(tokens / wall, 1),
        "req_per_s": round(len(results) / wall, 2),
    }


async def main():
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("VLLM_URL"), help="server base URL (or VLLM_URL)")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--requests-per-level", type=int, default=None, help="default: 4x concurrency, min 8")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--label", default="", help="tag for this run, e.g. 'L4 qwen2.5-7b fp16'")
    args = p.parse_args()
    if not args.url:
        p.error("pass --url or set VLLM_URL in .env")
    url, key = args.url.rstrip("/"), os.environ["VLLM_API_KEY"]

    # Modal answers slow cold starts with a 303 "retry here" redirect, so poll /health
    # (following redirects) until vLLM is up; benchmark requests then never hit the redirect.
    print("waiting for the server (a cold start can take several minutes)...")
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), follow_redirects=True) as client:
        while True:
            try:
                if (await client.get(f"{url}/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - t0 > 1500:
                raise SystemExit("server did not become healthy in 25 minutes")
            await asyncio.sleep(5)
        print(f"server ready after {time.perf_counter() - t0:.0f}s, warming up...")
        await one_request(client, url, key, 8)

    levels = []
    for c in args.concurrency:
        n = args.requests_per_level or max(8, 4 * c)
        print(f"concurrency {c}: {n} requests")
        levels.append(await run_level(url, key, c, n, args.max_tokens))

    print(f"\n{'conc':>5} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'TPOT p99':>9} "
          f"{'user tok/s':>10} {'total tok/s':>11} {'err':>4}")
    for r in levels:
        print(f"{r['concurrency']:>5} {r['ttft_ms'].get('p50', '-'):>9} {r['ttft_ms'].get('p99', '-'):>9} "
              f"{r['tpot_ms'].get('p50', '-'):>9} {r['tpot_ms'].get('p99', '-'):>9} "
              f"{r['per_user_tok_per_s']:>10} {r['output_tok_per_s']:>11} {r['errors']:>4}")

    out = Path(__file__).with_name("results") / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"label": args.label, "url": url, "max_tokens": args.max_tokens,
                               "levels": levels}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
