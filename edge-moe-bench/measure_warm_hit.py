# Test the user's warmup hypothesis: does warming with 2 problems lift the
# DRAM expert hit rate? Persistent-server semantics are reproduced with
# llama-cli passes (the OS standby cache carries over across processes; each
# pass's load prefetch is identical).
#
# sequence: problem 0 (warmup), problem 5 (warmup), problem 0 again
# (warm_same, measured), problem 10 (new_problem, measured).
import csv
import json
import subprocess
import sys
import threading
import time

sys.path.insert(0, "edge-moe-bench")
from measure_dram_hit2 import (  # noqa: E402
    PROMPT, set_ws_cap,
)

PROMPT_0 = PROMPT

PROBLEMS = {}
with open("HumanEval.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            PROBLEMS[p["task_id"]] = p["prompt"]

PROMPT_5 = PROBLEMS["HumanEval/5"]
PROMPT_10 = PROBLEMS["HumanEval/10"]

PER_TOKEN_EXPERT_BYTES = 4 * 40 * 2039808  # opt1 skip
N_PREDICT = 32


def run_cli(prompt, log_path):
    cmd = [
        "build-win/bin/Release/llama-cli.exe",
        "-m", "D:\\workspace\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "-p", prompt,
        "-n", str(N_PREDICT), "-t", "8", "--ctx-size", "2048",
        "--no-warmup", "--perf", "--verbose", "--temp", "0",
        "--no-prefetch", "-fit", "off",
        "--moe-skip-k1", "4", "--moe-skip-k2", "16",
        "--single-turn", "--simple-io",
    ]
    log = open(log_path, "wb")
    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    set_ws_cap(proc.pid, int(6.5 * 1024 ** 3))
    decode_started = False
    t_decode = None
    while proc.poll() is None:
        log.flush()
        try:
            tail = open(log_path, "rb").read()[-20000:]
        except OSError:
            tail = b""
        if not decode_started and b'"predicted_n":1,' in tail:
            t_decode = time.time()
            decode_started = True
        time.sleep(0.3)
    ret = proc.wait()
    log.close()
    return ret, t_decode, time.time(), time.time() - t_start


def main():
    counter_csv = "edge-moe-results/pages_input_warm.csv"
    tp = subprocess.Popen(
        ["typeperf", "\\Memory\\Pages Input/sec", "-si", "1", "-f", "csv",
         "-o", counter_csv, "-y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    t_counter_start = time.time()

    plan = [
        ("warmup1", PROMPT_0, "edge-moe-results/warm_p0a.log", False),
        ("warmup2", PROMPT_5, "edge-moe-results/warm_p5a.log", False),
        ("warm_same", PROMPT_0, "edge-moe-results/warm_p0b.log", True),
        ("new_problem", PROMPT_10, "edge-moe-results/warm_p10.log", True),
    ]
    results = []
    for name, prompt, log_path, measure in plan:
        ret, t_decode, t_end, wall = run_cli(prompt, log_path)
        if ret != 0 or t_decode is None:
            print(f"{name}: FAILED (exit {ret})", file=sys.stderr)
            continue
        row = {"phase": name, "exit": ret, "decode_wall_s": round(t_end - t_decode, 1),
               "load_wall_s": round(t_decode, 1), "wall_s": round(wall, 1)}
        if measure:
            row["decode_window"] = [t_decode, t_end]
            touched = (N_PREDICT - 1) * PER_TOKEN_EXPERT_BYTES
            row["touched_expert_bytes_mb"] = round(touched / 1024 / 1024, 1)
        results.append(row)
        print(f"[{name}] load={row['load_wall_s']}s decode_wall={row['decode_wall_s']}s wall={row['wall_s']}s", flush=True)

    time.sleep(4)
    tp.terminate()
    time.sleep(2)

    try:
        with open(counter_csv, "r", errors="replace") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        rows = []
    samples = []
    for i, row in enumerate(rows[1:]):
        if len(row) < 2:
            continue
        try:
            v = float(row[1])
        except ValueError:
            continue
        samples.append((t_counter_start - 3 + i, v))
    for row in results:
        if "decode_window" not in row:
            continue
        ws, we = row["decode_window"]
        in_win = [s for _, s in samples if ws <= _ <= we]
        ssd_mb = sum(in_win) * 4096 / 1024 / 1024
        touched = row["touched_expert_bytes_mb"] * 1024 * 1024
        hit = 1.0 - (ssd_mb * 1024 * 1024) / touched if touched else 0.0
        row["ssd_read_decode_mb"] = round(ssd_mb, 1)
        row["dram_expert_hit_rate"] = round(max(0.0, min(1.0, hit)), 4)
        row["window_samples"] = len(in_win)

    with open("edge-moe-results/warm_hit_rate.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
