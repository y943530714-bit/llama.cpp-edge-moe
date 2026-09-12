# Per-token decode latency profile: is the prefill-eviction damage front-loaded
# (first few decode steps slow, then fast) or spread across all steps?
import json
import subprocess
import sys
import time
import urllib.request

MODEL = "D:\\workspace\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PORT = 8899

PROBLEMS = {}
with open("HumanEval.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            PROBLEMS[p["task_id"]] = p["prompt"]


def stream_request(prompt, n_predict=32):
    payload = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
        "top_k": 1, "cache_prompt": False, "stream": True,
        "timings_per_token": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    per_token = []
    prefill_ms = None
    with urllib.request.urlopen(req, timeout=2400) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                c = json.loads(body)
            except json.JSONDecodeError:
                continue
            t = c.get("timings") or {}
            if t.get("predicted_per_token_ms") and t.get("predicted_n", 0) >= 1:
                prefill_ms = t.get("prompt_ms")
                per_token.append((t.get("predicted_n"), round(t["predicted_per_token_ms"], 1)))
            if c.get("stop"):
                break
    return prefill_ms, per_token


def main():
    server = subprocess.Popen([
        "build-win/bin/Release/llama-server.exe",
        "-m", MODEL, "--ctx-size", "2048", "--no-warmup", "-t", "8",
        "--port", str(PORT), "--host", "127.0.0.1", "--parallel", "1",
        "--no-prefetch", "-fit", "off", "--no-repack",
        "--moe-skip-k1", "4", "--moe-skip-k2", "16",
    ], stdout=open("edge-moe-results/server_profile.log", "wb"), stderr=subprocess.STDOUT)
    try:
        ready = False
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                    if json.loads(r.read().decode()).get("status") == "ok":
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(3)
        if not ready:
            print("server failed to start", file=sys.stderr)
            sys.exit(3)
        print("server ready", flush=True)

        out = {}
        for label, task in [("req1_p0", "HumanEval/0"), ("req2_p0_warm", "HumanEval/0"),
                            ("req3_p10_new", "HumanEval/10")]:
            prefill_ms, per_token = stream_request(PROBLEMS[task])
            out[label] = {"prefill_ms": prefill_ms, "per_token": per_token}
            lat = [x[1] for x in per_token]
            tail = lat[6:] or [0]
            print(f"{label}: prefill={prefill_ms:.0f}ms  first6={lat[:6]}  "
                  f"tail_mean={sum(tail)/len(tail):.0f}ms/tok (n={len(tail)})", flush=True)
        with open("edge-moe-results/token_latency_profile.json", "w") as f:
            json.dump(out, f, indent=2)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
