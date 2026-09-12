# Measure the actual DRAM expert hit rate during decode, v2.
#
# v1 used GetProcessIoCounters, which does not see memory-manager reads for
# mapped-file hard faults (it only reported 21 MB during a multi-GB prefill).
# v2 samples the system-wide "\Memory\Pages Input/sec" perf counter (pages
# read from disk to resolve hard faults, incl. mapped files) at 1 s and sums
# the samples inside each decode window. With llama-cli running solo, the
# system hard-fault traffic is dominated by our process.
#
# DRAM expert hit rate = 1 - ssd_read_bytes / touched_expert_bytes
import ctypes
import csv
import json
import subprocess
import sys
import threading
import time

PROCESS_SET_QUOTA = 0x0100
QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x00000004

PER_TOKEN_BYTES = {  # expert bytes touched per decode step
    "opt1_skip": 4 * 40 * 2039808,   # 4 experts x 40 layers x (gate+up+down)
    "baseline": 8 * 40 * 2039808,
}


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


def run_one(config_name, extra, log_path, counter_path):
    per_token = PER_TOKEN_BYTES[config_name]
    cmd = [
        "build-win/bin/Release/llama-cli.exe",
        "-m", "D:\\workspace\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "-p", PROMPT,
        "-n", "32", "-t", "8", "--ctx-size", "2048",
        "--no-warmup", "--perf", "--verbose", "--temp", "0",
        "--single-turn", "--simple-io",
    ] + extra
    log = open(log_path, "wb")
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
    t_end = time.time()
    if not decode_started:
        print(f"{config_name}: FAILED to observe decode window (exit {ret})", file=sys.stderr)
        return None
    return {
        "config": config_name,
        "exit": ret,
        "decode_window": [t_decode, t_end],
        "decode_wall_s": round(t_end - t_decode, 1),
    }


PROMPT = ("from typing import List\n\n\ndef has_close_elements(numbers: List[float], "
          "threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any "
          "two numbers closer to each other than given threshold.\n"
          "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n"
          "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"")

if __name__ == "__main__":
    counter_csv = "edge-moe-results/pages_input.csv"
    tp = subprocess.Popen(
        ["typeperf", "\\Memory\\Pages Input/sec", "\\Memory\\Pages/sec",
         "-si", "1", "-f", "csv", "-o", counter_csv, "-y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    t0 = time.time()

    results = []
    r = run_one("opt1_skip", ["--moe-skip-k1", "4", "--moe-skip-k2", "16"],
                "edge-moe-results/dramhit2_opt1.log", counter_csv)
    if r:
        results.append(r)
    # flush pause so typeperf writes out the tail samples before we stop it
    time.sleep(4)
    r = run_one("baseline", [], "edge-moe-results/dramhit2_baseline.log", counter_csv)
    if r:
        results.append(r)

    time.sleep(4)
    tp.terminate()
    time.sleep(2)

    # map each decode window onto counter samples by sample index (1 Hz from t0-3s)
    try:
        with open(counter_csv, "r", errors="replace") as f:
            rows = list(csv.reader(f))
        samples = []
        for i, row in enumerate(rows[1:]):
            if len(row) < 3:
                continue
            try:
                val_in = float(row[1])
                val_all = float(row[2])
            except ValueError:
                continue
            samples.append((t0 - 3 + i, val_in, val_all))
    except FileNotFoundError:
        samples = []

    out = []
    for r in results:
        ws, we = r["decode_window"]
        in_win = [s for s in samples if ws <= s[0] <= we]
        pages_input = sum(s[1] for s in in_win)
        pages_all = sum(s[2] for s in in_win)
        touched = 31 * PER_TOKEN_BYTES[r["config"]]
        ssd_bytes = pages_input * 4096
        hit = 1.0 - ssd_bytes / touched if touched else 0.0
        out.append({
            "config": r["config"],
            "decode_wall_s": r["decode_wall_s"],
            "window_samples": len(in_win),
            "pages_input_total": round(pages_input),
            "ssd_read_decode_mb": round(ssd_bytes / 1024 / 1024, 1),
            "touched_expert_bytes_mb": round(touched / 1024 / 1024, 1),
            "dram_expert_hit_rate": round(max(0.0, min(1.0, hit)), 4),
            "pages_all_total": round(pages_all),
        })
    with open("edge-moe-results/dram_hit_rate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
