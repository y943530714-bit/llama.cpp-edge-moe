"""Profile an already-running mmap + expert-skip + DFlash server on Windows.

The script is intentionally attach-only: it does not start, stop, or reconfigure
llama-server.  It correlates streamed completion timings with system hard-page
inputs, physical-disk reads, and process CPU/working-set samples.
"""

import argparse
import csv
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request


COUNTERS = [
    r"\Memory\Pages Input/sec",
    r"\Memory\Page Reads/sec",
    r"\PhysicalDisk(_Total)\Disk Read Bytes/sec",
    r"\PhysicalDisk(_Total)\Avg. Disk sec/Read",
    r"\PhysicalDisk(_Total)\Current Disk Queue Length",
    r"\Processor Information(_Total)\% Processor Utility",
]

PROMPT_A = """Complete the following Python function. Return only the implementation.

def merge_intervals(intervals):
    \"\"\"Merge overlapping integer intervals and return them sorted.\"\"\"
"""

PROMPT_B = """Complete the following Python function. Return only the implementation.

def shortest_path(grid):
    \"\"\"Return the shortest 4-neighbour path length from S to E; # cells are blocked.\"\"\"
"""


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)


class FILETIME(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


def ft_value(v):
    return (v.hi << 32) | v.lo


class ProcessSampler:
    def __init__(self, pid, interval=0.25):
        self.pid = pid
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.handle = kernel32.OpenProcess(0x1000 | 0x0400, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        kernel32.CloseHandle(self.handle)

    def _read(self):
        creation = FILETIME()
        exit_time = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        if not kernel32.GetProcessTimes(
            self.handle, ctypes.byref(creation), ctypes.byref(exit_time),
            ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        mem = PROCESS_MEMORY_COUNTERS_EX()
        mem.cb = ctypes.sizeof(mem)
        if not psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(mem), mem.cb):
            return None
        io = IO_COUNTERS()
        if not kernel32.GetProcessIoCounters(self.handle, ctypes.byref(io)):
            return None
        return {
            "ts": time.time(),
            "cpu_100ns": ft_value(kernel) + ft_value(user),
            "ws_bytes": mem.WorkingSetSize,
            "private_bytes": mem.PrivateUsage,
            "page_faults": mem.PageFaultCount,
            "io_read_bytes": io.ReadTransferCount,
        }

    def _run(self):
        while not self.stop_event.is_set():
            sample = self._read()
            if sample:
                self.samples.append(sample)
            self.stop_event.wait(self.interval)


def request_stream(port, prompt, n_predict, timeout):
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": True,
        "timings_per_token": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.time()
    token_times = []
    timings = {}
    content = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
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
            piece = chunk.get("content", "")
            if piece:
                content.append(piece)
                token_times.append(time.time())
            if chunk.get("timings"):
                timings = chunk["timings"]
            if chunk.get("stop"):
                break
    t_end = time.time()
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    decode_start = t_end - predicted_ms / 1000.0 if predicted_ms else (
        token_times[0] if token_times else t_start
    )
    gaps = [1000.0 * (b - a) for a, b in zip(token_times, token_times[1:])]
    return {
        "request_start": t_start,
        "decode_start": decode_start,
        "request_end": t_end,
        "wall_s": t_end - t_start,
        "stream_chunks": len(token_times),
        "inter_chunk_ms": gaps,
        "timings": timings,
        "content": "".join(content),
    }


def parse_typeperf(path):
    rows = []
    # typeperf writes UTF-16 when redirected to a CSV file on this host.
    for encoding in ("utf-16", "utf-8-sig", "mbcs"):
        try:
            with path.open("r", encoding=encoding, newline="") as stream:
                table = list(csv.reader(stream))
            if table and len(table[0]) > 1:
                break
        except (UnicodeError, UnicodeDecodeError):
            table = []
    if not table:
        return [], []
    headers = table[0]
    for row in table[1:]:
        if len(row) != len(headers):
            continue
        try:
            # The host uses the invariant typeperf timestamp format.
            stamp = time.mktime(time.strptime(row[0].split(".")[0], "%m/%d/%Y %H:%M:%S"))
            if "." in row[0]:
                stamp += float("0." + row[0].rsplit(".", 1)[1])
            values = [float(v) for v in row[1:]]
        except ValueError:
            continue
        rows.append({"ts": stamp, "values": values})
    return headers[1:], rows


def find_counter(headers, suffix):
    suffix = suffix.lower()
    for i, header in enumerate(headers):
        if header.lower().endswith(suffix):
            return i
    raise KeyError(suffix)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_process(samples, start, end):
    selected = [s for s in samples if start <= s["ts"] <= end]
    if len(selected) < 2:
        return {}
    first, last = selected[0], selected[-1]
    elapsed = last["ts"] - first["ts"]
    cpu_s = (last["cpu_100ns"] - first["cpu_100ns"]) / 10_000_000.0
    return {
        "sample_count": len(selected),
        "cpu_cores_avg": cpu_s / elapsed if elapsed else None,
        "cpu_machine_pct": 100.0 * cpu_s / elapsed / (os.cpu_count() or 1) if elapsed else None,
        "ws_gib_avg": statistics.fmean(s["ws_bytes"] for s in selected) / 2**30,
        "ws_gib_max": max(s["ws_bytes"] for s in selected) / 2**30,
        "private_gib_max": max(s["private_bytes"] for s in selected) / 2**30,
        "page_faults_delta": last["page_faults"] - first["page_faults"],
        "reported_io_read_mib": (last["io_read_bytes"] - first["io_read_bytes"]) / 2**20,
    }


def summarize_system(headers, samples, start, end, baseline):
    selected = [s for s in samples if start <= s["ts"] <= end + 1.0]
    if not selected:
        return {}
    idx_pages = find_counter(headers, r"\memory\pages input/sec")
    idx_reads = find_counter(headers, r"\memory\page reads/sec")
    idx_bytes = find_counter(headers, r"\physicaldisk(_total)\disk read bytes/sec")
    idx_latency = find_counter(headers, r"\physicaldisk(_total)\avg. disk sec/read")
    idx_queue = find_counter(headers, r"\physicaldisk(_total)\current disk queue length")
    idx_cpu = find_counter(headers, r"\processor information(_total)\% processor utility")
    duration = max(end - start, 0.001)
    page_rate = statistics.fmean(s["values"][idx_pages] for s in selected)
    disk_rate = statistics.fmean(s["values"][idx_bytes] for s in selected)
    page_excess = max(0.0, page_rate - baseline.get("pages_input_per_s", 0.0))
    disk_excess = max(0.0, disk_rate - baseline.get("disk_read_mib_s", 0.0) * 2**20)
    return {
        "sample_count": len(selected),
        "pages_input_per_s_avg": page_rate,
        "page_reads_per_s_avg": statistics.fmean(s["values"][idx_reads] for s in selected),
        "hard_page_input_mib_raw": page_rate * duration * 4096 / 2**20,
        "hard_page_input_mib_above_idle": page_excess * duration * 4096 / 2**20,
        "disk_read_mib_s_avg": disk_rate / 2**20,
        "disk_read_mib_raw": disk_rate * duration / 2**20,
        "disk_read_mib_above_idle": disk_excess * duration / 2**20,
        "disk_read_latency_ms_avg": 1000.0 * statistics.fmean(s["values"][idx_latency] for s in selected),
        "disk_queue_avg": statistics.fmean(s["values"][idx_queue] for s in selected),
        "disk_queue_max": max(s["values"][idx_queue] for s in selected),
        "machine_cpu_utility_pct_avg": statistics.fmean(s["values"][idx_cpu] for s in selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counter_path = args.out.with_suffix(".typeperf.csv")

    sampler = ProcessSampler(args.pid)
    sampler.start()
    typeperf = subprocess.Popen(
        ["typeperf", *COUNTERS, "-si", "1", "-f", "CSV", "-o", str(counter_path), "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sequence = [
        ("first_A", PROMPT_A),
        ("same_A", PROMPT_A),
        ("new_B", PROMPT_B),
        ("same_B", PROMPT_B),
        ("return_A", PROMPT_A),
    ]
    results = []
    run_start = time.time()
    print("idle baseline: 6 s", flush=True)
    time.sleep(6)
    baseline_end = time.time()
    try:
        for name, prompt in sequence:
            print(f"start {name}", flush=True)
            result = request_stream(args.port, prompt, args.tokens, args.timeout)
            result["phase"] = name
            results.append(result)
            timing = result["timings"]
            print(
                f"done {name}: wall={result['wall_s']:.2f}s, "
                f"pred={timing.get('predicted_n')} tok, "
                f"tpot={timing.get('predicted_per_token_ms')} ms, "
                f"draft={timing.get('draft_n')}/{timing.get('draft_n_accepted')}",
                flush=True,
            )
            time.sleep(3)
    finally:
        typeperf.terminate()
        try:
            typeperf.wait(timeout=10)
        except subprocess.TimeoutExpired:
            typeperf.kill()
        sampler.stop()

    headers, system_samples = parse_typeperf(counter_path)
    base_samples = [s for s in system_samples if run_start <= s["ts"] <= baseline_end]
    baseline = {}
    if base_samples:
        idx_pages = find_counter(headers, r"\memory\pages input/sec")
        idx_bytes = find_counter(headers, r"\physicaldisk(_total)\disk read bytes/sec")
        baseline = {
            "sample_count": len(base_samples),
            "pages_input_per_s": statistics.median(s["values"][idx_pages] for s in base_samples),
            "disk_read_mib_s": statistics.median(s["values"][idx_bytes] for s in base_samples) / 2**20,
        }

    for result in results:
        start, end = result["decode_start"], result["request_end"]
        result["process_decode"] = summarize_process(sampler.samples, start, end)
        result["system_decode"] = summarize_system(headers, system_samples, start, end, baseline)
        gaps = result.pop("inter_chunk_ms")
        result["stream_gap_ms"] = {
            "count": len(gaps),
            "median": percentile(gaps, 0.5),
            "p95": percentile(gaps, 0.95),
            "max": max(gaps) if gaps else None,
        }

    document = {
        "metadata": {
            "pid": args.pid,
            "port": args.port,
            "n_predict": args.tokens,
            "logical_cpu_count": os.cpu_count(),
            "run_start_epoch": run_start,
            "run_end_epoch": time.time(),
            "counter_headers": headers,
        },
        "idle_baseline": baseline,
        "requests": results,
    }
    with args.out.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
    print(json.dumps(document, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
