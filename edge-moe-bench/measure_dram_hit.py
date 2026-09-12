# Measure the actual DRAM expert hit rate during decode.
#
# Method: run llama-cli for one HumanEval problem with --verbose (streamed
# chunks), snapshot the process IO counters when the first decoded chunk
# appears (prefill done, decode window starts) and at process exit. The
# ReadTransferCount delta is the SSD bytes read during decode. Compare with
# the logical expert bytes touched per decode step:
#   per expert per layer = gate(q4_K) 589824 + up(q4_K) 589824 + down(q6_K) 860160
#   baseline: 8 experts x 40 layers, opt1 skip: 4 experts x 40 layers
# DRAM expert hit rate = 1 - ssd_read_bytes / touched_expert_bytes
import ctypes
import json
import subprocess
import sys
import time

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_SET_QUOTA = 0x0100
QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x00000004


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


def io_snapshot(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
    if not h:
        return None
    io = IO_COUNTERS()
    ok = k32.GetProcessIoCounters(h, ctypes.byref(io))
    k32.CloseHandle(h)
    return io if ok else None


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


def measure(config_name, extra, expert_slices_per_layer, log_path):
    per_token_bytes = expert_slices_per_layer * 40 * 2039808
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
    io_start = io_snapshot(proc.pid)
    t0 = time.time()

    decode_started = False
    io_decode_start = None
    t_decode = None
    while proc.poll() is None:
        log.flush()
        try:
            tail = open(log_path, "rb").read()[-20000:]
        except OSError:
            tail = b""
        if not decode_started and b'"predicted_n":1,' in tail:
            io_decode_start = io_snapshot(proc.pid)
            t_decode = time.time()
            decode_started = True
        time.sleep(0.3)
    ret = proc.wait()
    log.close()
    io_end = io_snapshot(proc.pid)
    wall = time.time() - t0

    if not decode_started or io_decode_start is None or io_end is None:
        print(f"{config_name}: FAILED to observe decode window (exit {ret})", file=sys.stderr)
        return None

    prefill_read = io_decode_start.ReadTransferCount - io_start.ReadTransferCount
    decode_read = io_end.ReadTransferCount - io_decode_start.ReadTransferCount
    touched = 31 * per_token_bytes
    hit = 1.0 - decode_read / touched if touched else 0.0
    result = {
        "config": config_name,
        "exit": ret,
        "decode_wall_s": round(time.time() - (t_decode or t0), 1),
        "wall_s": round(wall, 1),
        "expert_bytes_per_token_mb": round(per_token_bytes / 1024 / 1024, 1),
        "touched_expert_bytes_mb": round(touched / 1024 / 1024, 1),
        "ssd_read_prefill_mb": round(prefill_read / 1024 / 1024, 1),
        "ssd_read_decode_mb": round(decode_read / 1024 / 1024, 1),
        "dram_expert_hit_rate": round(max(0.0, min(1.0, hit)), 4),
    }
    print(json.dumps(result, indent=2))
    return result


PROMPT = ("from typing import List\n\n\ndef has_close_elements(numbers: List[float], "
          "threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any "
          "two numbers closer to each other than given threshold.\n"
          "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n"
          "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"")

if __name__ == "__main__":
    results = []
    # opt1: 4 of 8 experts computed per layer
    r = measure("opt1_skip", ["--moe-skip-k1", "4", "--moe-skip-k2", "16"],
                4 * 3, "edge-moe-results/dramhit_opt1.log")
    if r:
        results.append(r)
    # baseline: all 8 experts
    r = measure("baseline", [], 8 * 3, "edge-moe-results/dramhit_baseline.log")
    if r:
        results.append(r)
    with open("edge-moe-results/dram_hit_rate.json", "w") as f:
        json.dump(results, f, indent=2)
