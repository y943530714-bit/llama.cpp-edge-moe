# Measure DRAM expert hit rate under a persistent llama-server with warmup.
#
# Protocol (user suggestion): launch llama-server once (hard 6.5 GB working
# set cap), send 2 warmup problems, then measure the same 2 problems again
# (warm same-prompt pass) and 3 further problems. For each measured request
# the decode window is [first streamed chunk, final chunk]; the DRAM expert
# hit rate = 1 - pages_input_during_window*4096 / (31 * 311.3 MB) for the
# opt1 skip config.
import ctypes
import csv
import json
import subprocess
import sys
import threading
import time
import urllib.request

PROCESS_SET_QUOTA = 0x0100
QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x00000004

PER_TOKEN_EXPERT_BYTES = 4 * 40 * 2039808  # opt1: 4 experts x 40 layers
MODEL = "D:\\workspace\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PORT = 8899


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


def free_gb():
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 1024 ** 3


def set_ws_cap(pid, cap_bytes):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_SET_QUOTA, False, pid)
    if not h:
        return False
    try:
        return bool(k32.SetProcessWorkingSetSizeEx(
            h, ctypes.c_size_t(64 * 1024 * 1024), ctypes.c_size_t(cap_bytes),
            ctypes.c_uint(QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE)))
    finally:
        k32.CloseHandle(h)


def load_problems():
    probs = []
    with open("HumanEval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                probs.append(json.loads(line))
    return {p["task_id"]: p for p in probs}


def request_stream(prompt, n_predict=32, timeout=2400):
    """POST a streaming completion; returns (t_first_decode_chunk, t_end, final_timings)."""
    payload = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
        "top_k": 1, "cache_prompt": False, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    t_send = time.time()
    t_first = None
    timings = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if "timings" in chunk and chunk["timings"].get("predicted_n", 0) >= 1:
                if t_first is None:
                    t_first = time.time()
                timings = chunk["timings"]
            if chunk.get("stop"):
                break
    return t_send, t_first, time.time(), timings


def main():
    problems = load_problems()
    seq = [("warmup", "HumanEval/0"), ("warmup", "HumanEval/5"),
           ("warm_same", "HumanEval/0"), ("warm_same", "HumanEval/5"),
           ("new_problem", "HumanEval/10"), ("new_problem", "HumanEval/15"),
           ("new_problem", "HumanEval/20")]

    counter_csv = "edge-moe-results/pages_input_server.csv"
    tp = subprocess.Popen(
        ["typeperf", "\\Memory\\Pages Input/sec", "-si", "1", "-f", "csv",
         "-o", counter_csv, "-y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    t_counter_start = time.time()

    server_cmd = [
        "build-win/bin/Release/llama-server.exe",
        "-m", MODEL, "--ctx-size", "2048", "--no-warmup", "-t", "8",
        "--port", str(PORT), "--host", "127.0.0.1", "--parallel", "1",
        "--moe-skip-k1", "4", "--moe-skip-k2", "16",
    ]
    log = open("edge-moe-results/server_warm.log", "wb")
    server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT)
    capped = set_ws_cap(server.pid, int(6.5 * 1024 ** 3))
    print(f"server pid {server.pid}, ws cap applied: {capped}", flush=True)

    # wait for health
    ready = False
    for _ in range(180):
        if server.poll() is not None:
            print("server exited during load", file=sys.stderr)
            tp.terminate()
            sys.exit(3)
        # the full-file prefetch at load transiently pushes available memory
        # low; only abort on a hard limit (windows trims standby on its own)
        if free_gb() < 0.6:
            print("ABORT: free memory below 0.6 GB during load", file=sys.stderr)
            server.kill()
            tp.terminate()
            sys.exit(2)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(4)
    if not ready:
        print("server never became healthy", file=sys.stderr)
        server.kill()
        tp.terminate()
        sys.exit(3)
    print(f"server ready after {time.time() - t_counter_start:.0f}s", flush=True)

    results = []
    try:
        for phase, task_id in seq:
            if free_gb() < 0.8:
                print("ABORT: low memory", file=sys.stderr)
                break
            t_send, t_first, t_end, timings = request_stream(problems[task_id]["prompt"])
            tpot = timings.get("predicted_per_token_ms")
            pred_n = timings.get("predicted_n", 0)
            row = {
                "phase": phase, "task_id": task_id,
                "wall_s": round(t_end - t_send, 1),
                "decode_wall_s": round(t_end - t_first, 1) if t_first else None,
                "tpot_ms": round(tpot, 1) if tpot else None,
                "predicted_n": pred_n,
            }
            if t_first and phase != "warmup":
                touched = (pred_n - 1) * PER_TOKEN_EXPERT_BYTES
                row["decode_window"] = [t_first, t_end]
                row["touched_expert_bytes_mb"] = round(touched / 1024 / 1024, 1)
            results.append(row)
            print(f"[{phase}] {task_id}: tpot={row['tpot_ms']} ms, "
                  f"decode_wall={row['decode_wall_s']}s", flush=True)
    finally:
        time.sleep(4)
        tp.terminate()
        time.sleep(2)
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    # map decode windows onto 1 Hz counter samples
    try:
        with open(counter_csv, "r", errors="replace") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        rows = []
    samples = [(t_counter_start - 3 + i, float(row[1]))
               for i, row in enumerate(rows[1:]) if len(row) >= 2]
    for row in results:
        if "decode_window" not in row:
            continue
        ws, we = row["decode_window"]
        in_win = [s for _, s in samples if ws <= _ <= we]
        ssd_mb = sum(in_win) * 4096 / 1024 / 1024
        touched = row.get("touched_expert_bytes_mb", 1) * 1024 * 1024
        hit = 1.0 - (ssd_mb * 1024 * 1024) / touched if touched else 0.0
        row["ssd_read_decode_mb"] = round(ssd_mb, 1)
        row["dram_expert_hit_rate"] = round(max(0.0, min(1.0, hit)), 4)
        row["window_samples"] = len(in_win)

    with open("edge-moe-results/server_hit_rate.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
