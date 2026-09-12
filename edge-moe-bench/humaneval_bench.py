# HumanEval TPOT benchmark driver for the edge MoE experiments.
# Sends the prompts one by one to a running llama-server and records the
# per-request timings reported by the server. Aborts early when host free
# memory drops below the safety margin (8 GB budget machine).
import argparse
import ctypes
import json
import sys
import time
import urllib.request


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_mem_gb():
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / (1024 ** 3)


def post_completion(base_url, payload, timeout=1800):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--dataset", default="HumanEval.jsonl")
    ap.add_argument("--n-problems", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--predict", type=int, default=128)
    ap.add_argument("--config-name", default="config")
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--min-free-gb", type=float, default=1.2)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    problems = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    picked = []
    i = args.offset
    while len(picked) < args.n_problems and i < len(problems):
        picked.append(problems[i])
        i += args.stride
    if len(picked) < args.n_problems:
        print(f"warning: only {len(picked)} problems available", file=sys.stderr)

    base_url = f"http://{args.host}:{args.port}"
    rows = []
    for n, prob in enumerate(picked):
        free = free_mem_gb()
        if free < args.min_free_gb:
            print(f"ABORT: free memory {free:.2f} GB below {args.min_free_gb} GB", file=sys.stderr)
            sys.exit(2)

        payload = {
            "prompt": prob["prompt"],
            "n_predict": args.predict,
            "temperature": args.temperature,
            "top_k": 1,
            "cache_prompt": False,
            "speculative": bool(args.speculative),
        }
        t0 = time.time()
        try:
            out = post_completion(base_url, payload)
        except Exception as e:
            print(f"request failed on {prob['task_id']}: {e}", file=sys.stderr)
            sys.exit(3)
        wall = time.time() - t0

        timings = out.get("timings", {})
        prompt_n = timings.get("prompt_n", 0)
        prompt_ms = timings.get("prompt_ms", 0.0)
        predicted_n = timings.get("predicted_n", 0)
        predicted_ms = timings.get("predicted_ms", 0.0)
        tpot_ms = predicted_ms / predicted_n if predicted_n else 0.0
        ttft_ms = prompt_ms

        rows.append({
            "config": args.config_name,
            "task_id": prob["task_id"],
            "prompt_n": prompt_n,
            "predicted_n": predicted_n,
            "prompt_ms": round(prompt_ms, 1),
            "predicted_ms": round(predicted_ms, 1),
            "tpot_ms": round(tpot_ms, 2),
            "ttft_ms": round(ttft_ms, 1),
            "wall_s": round(wall, 1),
            "stopped_eos": out.get("stopped_eos", False),
        })
        print(f"[{n + 1}/{len(picked)}] {prob['task_id']}: tpot={tpot_ms:.1f} ms/tok "
              f"({predicted_n} tok, ttft={prompt_ms:.0f} ms, wall={wall:.0f}s)", flush=True)

    tpots = [r["tpot_ms"] for r in rows if r["predicted_n"] > 0]
    gen_total = sum(r["predicted_n"] for r in rows)
    gen_ms_total = sum(r["predicted_ms"] for r in rows)
    summary = {
        "config": args.config_name,
        "n_problems": len(rows),
        "predict": args.predict,
        "mean_tpot_ms": round(sum(tpots) / len(tpots), 2) if tpots else None,
        "median_tpot_ms": round(sorted(tpots)[len(tpots) // 2], 2) if tpots else None,
        "pooled_tpot_ms": round(gen_ms_total / gen_total, 2) if gen_total else None,
        "total_generated": gen_total,
        "total_decode_s": round(gen_ms_total / 1000.0, 1),
    }

    header = "config,task_id,prompt_n,predicted_n,prompt_ms,predicted_ms,tpot_ms,ttft_ms,wall_s,stopped_eos"
    write_header = False
    try:
        with open(args.out, "r", encoding="utf-8") as f:
            write_header = len(f.read()) == 0
    except FileNotFoundError:
        write_header = True
    with open(args.out, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in header.split(",")) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
