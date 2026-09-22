"""Measure DFlash draft, target verify, cold page-in penalty and mmap hit rate."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request

from profile_mmap_spec_stack import (
    COUNTERS,
    ProcessSampler,
    find_counter,
    parse_typeperf,
    request_stream,
    summarize_process,
    summarize_system,
)


# Exact Q4_K_M layout: 37 layers use 1,900,544 bytes/expert and
# layers 34/38/39 use 2,039,808 bytes/expert.
EXPERT_BYTES_PER_TARGET_ROW = 4 * (37 * 1_900_544 + 3 * 2_039_808)
PROFILE_RE = re.compile(r"EDGEPROF phase=(\S+) us=(\d+)(.*)$")
FIELD_RE = re.compile(r"(\w+)=(\d+)")
BATCH_CACHE_RE = re.compile(r"EDGEMOE phase=(\S+)(.*)$")
CACHE_PHASE_RE = re.compile(
    r": (prefill|decode/verify) batches = (\d+), hits = (\d+), misses = (\d+), "
    r"hit rate = ([0-9.]+)%, evictions = (\d+), requests = (\d+), "
    r"requested = ([0-9.]+) MiB, transferred = ([0-9.]+) MiB, I/O wait = ([0-9.]+) ms"
)
MEMORY_RE = re.compile(
    r"streaming working set = ([0-9.]+) MiB, peak = ([0-9.]+) MiB, "
    r"private = ([0-9.]+) MiB, target = ([0-9.]+) MiB"
)

PROMPT_C = """Complete this C++ function. Return only the implementation.
std::vector<int> topological_order(int n, const std::vector<std::pair<int, int>>& edges) {
"""

PROMPT_D = """Complete the following Python function. Return only the implementation.
def rolling_median(values, window):
    \"\"\"Return one median for every full sliding window, including duplicate values.\"\"\"
"""


def set_ws_cap(pid, cap_bytes):
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x0100, False, pid)
    if not handle:
        return False
    try:
        return bool(kernel32.SetProcessWorkingSetSizeEx(
            handle,
            ctypes.c_size_t(64 * 1024 * 1024),
            ctypes.c_size_t(cap_bytes),
            ctypes.c_uint(0x2 | 0x4),
        ))
    finally:
        kernel32.CloseHandle(handle)


def wait_ready(port, process, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if json.loads(response.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError("server did not become ready")


def parse_phase_lines(lines):
    phases = {}
    for line in lines:
        match = PROFILE_RE.search(line)
        if not match:
            continue
        phase, us, tail = match.groups()
        fields = {key: int(value) for key, value in FIELD_RE.findall(tail)}
        item = phases.setdefault(phase, {"calls": 0, "total_us": 0, "rows": 0, "tokens": 0})
        item["calls"] += 1
        item["total_us"] += int(us)
        item["rows"] += fields.get("rows", 0)
        item["tokens"] += fields.get("tokens", 0)
        if "candidates" in fields:
            item["candidates"] = item.get("candidates", 0) + fields["candidates"]
        if "accepted" in fields:
            item["accepted"] = item.get("accepted", 0) + fields["accepted"]
    for item in phases.values():
        item["total_ms"] = item["total_us"] / 1000.0
        item["mean_ms"] = item["total_ms"] / item["calls"] if item["calls"] else None
    return phases


def parse_cache_batch_lines(lines):
    phases = {}
    for line in lines:
        match = BATCH_CACHE_RE.search(line)
        if not match:
            continue
        name, tail = match.groups()
        fields = {key: int(value) for key, value in FIELD_RE.findall(tail)}
        item = phases.setdefault(name, {
            "batches": 0, "hits": 0, "misses": 0, "evictions": 0,
            "io_requests": 0, "requested_bytes": 0, "transferred_bytes": 0,
            "io_wait_us": 0, "read_calls": 0, "loaded_experts": 0,
            "prefetch_loaded": 0, "prefetch_useful": 0, "prefetch_wasted": 0,
            "prefetch_wait_us": 0,
        })
        for key in item:
            source_key = "requests" if key == "io_requests" else key
            item[key] += fields.get(source_key, 0)
    for item in phases.values():
        total = item["hits"] + item["misses"]
        item["hit_rate_pct"] = 100.0 * item["hits"] / total if total else 0.0
        item["requested_mib"] = item["requested_bytes"] / 2**20
        item["transferred_mib"] = item["transferred_bytes"] / 2**20
        item["io_wait_ms"] = item["io_wait_us"] / 1000.0
    return phases


def aggregate_request_cache(requests):
    result = {}
    for request in requests:
        for name, source in request.get("cache_phases", {}).items():
            item = result.setdefault(name, {
                "batches": 0, "hits": 0, "misses": 0, "evictions": 0,
                "io_requests": 0, "requested_bytes": 0, "transferred_bytes": 0,
                "io_wait_us": 0, "read_calls": 0, "loaded_experts": 0,
                "prefetch_loaded": 0, "prefetch_useful": 0, "prefetch_wasted": 0,
                "prefetch_wait_us": 0,
            })
            for key in item:
                item[key] += source.get(key, 0)
    for item in result.values():
        total = item["hits"] + item["misses"]
        item["hit_rate_pct"] = 100.0 * item["hits"] / total if total else 0.0
        item["requested_mib"] = item["requested_bytes"] / 2**20
        item["transferred_mib"] = item["transferred_bytes"] / 2**20
        item["io_wait_ms"] = item["io_wait_us"] / 1000.0
    return result


def parse_streaming_summary(lines):
    result = {"cache_phases": {}}
    for line in lines:
        match = CACHE_PHASE_RE.search(line)
        if match:
            name, batches, hits, misses, hit_rate, evictions, requests, requested, transferred, io_wait = match.groups()
            result["cache_phases"][name] = {
                "batches": int(batches),
                "hits": int(hits),
                "misses": int(misses),
                "hit_rate_pct": float(hit_rate),
                "evictions": int(evictions),
                "io_requests": int(requests),
                "requested_mib": float(requested),
                "transferred_mib": float(transferred),
                "io_wait_ms": float(io_wait),
            }
        match = MEMORY_RE.search(line)
        if match:
            working_set, peak_working_set, private_bytes, target = match.groups()
            result["process_memory"] = {
                "working_set_mib": float(working_set),
                "peak_working_set_mib": float(peak_working_set),
                "private_bytes_mib": float(private_bytes),
                "target_mib": float(target),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--mode", choices=("mmap", "streaming"), default="mmap")
    parser.add_argument("--ws-cap-mib", type=int, default=8192)
    parser.add_argument("--streaming-budget-mib", type=int, default=8192)
    parser.add_argument("--streaming-io-depth", type=int, default=8)
    parser.add_argument("--spec-draft-n-max", type=int, default=7)
    parser.add_argument("--moe-trace", type=Path, help="optional route trace JSONL output")
    parser.add_argument("--moe-trace-max-events", type=int, default=0)
    parser.add_argument("--baseline-seconds", type=float, default=4.0)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.spec_draft_n_max <= 0:
        parser.error("--spec-draft-n-max must be positive")
    if args.moe_trace_max_events < 0:
        parser.error("--moe-trace-max-events must be non-negative")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    typeperf_path = args.out.with_suffix(".typeperf.csv")
    log_path = args.out.with_suffix(".server.log")

    command = [
        str(args.server.resolve()), "-m", str(args.model.resolve()), "-md", str(args.draft.resolve()),
        "--spec-type", "draft-dflash", "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--spec-relaxed-verify", "--spec-relaxed-eps", "0.09", "--spec-relaxed-alpha", "0.3",
        "--moe-skip-k1", "4", "--moe-skip-k2", "16", "--no-prefetch", "--no-repack",
        "-fit", "off", "--parallel", "1", "-c", "2048", "-t", "8", "-tb", "8",
        "--spec-draft-threads", "8", "--spec-draft-threads-batch", "8",
        "--cache-ram", "0", "--no-warmup", "-lv", "4", "--port", str(args.port),
    ]
    if args.mode == "streaming":
        command.extend([
            "--moe-streaming-budget-mib", str(args.streaming_budget_mib),
            "--moe-streaming-io-depth", str(args.streaming_io_depth),
        ])
    if args.moe_trace is not None:
        args.moe_trace.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--moe-trace", str(args.moe_trace.resolve())])
        if args.moe_trace_max_events:
            command.extend(["--moe-trace-max-events", str(args.moe_trace_max_events)])

    lines = []
    lines_lock = threading.Lock()
    process_env = os.environ.copy()
    process_env["LLAMA_EDGE_MOE_PROFILE"] = "1"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        env=process_env,
    )

    def read_log():
        assert process.stdout is not None
        for line in process.stdout:
            with lines_lock:
                lines.append(line.rstrip())

    reader = threading.Thread(target=read_log, daemon=True)
    reader.start()
    if args.mode == "mmap" and not set_ws_cap(process.pid, args.ws_cap_mib * 2**20):
        process.terminate()
        raise RuntimeError("failed to apply working-set cap")

    process_sampler = ProcessSampler(process.pid)
    process_sampler.start()

    counters = [*COUNTERS, r"\Memory\Available MBytes"]
    typeperf = subprocess.Popen(
        ["typeperf", *counters, "-si", "1", "-f", "CSV", "-o", str(typeperf_path), "-y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=0x08000000,
    )
    run_start = time.time()
    requests = []
    try:
        wait_ready(args.port, process)
        print(f"idle baseline: {args.baseline_seconds:.1f} s", flush=True)
        time.sleep(args.baseline_seconds)
        baseline_end = time.time()
        sequence = [
            ("cold_C", PROMPT_C), ("warm_C", PROMPT_C),
            ("cold_D", PROMPT_D), ("warm_D", PROMPT_D),
        ]
        for name, prompt in sequence:
            with lines_lock:
                log_start = len(lines)
            print(f"start {name}", flush=True)
            result = request_stream(args.port, prompt, args.tokens, 2400)
            time.sleep(0.25)
            with lines_lock:
                segment = list(lines[log_start:])
            result["phase"] = name
            result["profile_phases"] = parse_phase_lines(segment)
            result["cache_phases"] = parse_cache_batch_lines(segment)
            requests.append(result)
            target = result["profile_phases"].get("target_verify", {})
            draft_gen = result["profile_phases"].get("draft_generate", {})
            print(
                f"done {name}: tpot={result['timings'].get('predicted_per_token_ms'):.2f} ms, "
                f"draft={draft_gen.get('total_ms', 0):.2f} ms, "
                f"target_verify={target.get('total_ms', 0):.2f} ms",
                flush=True,
            )
            time.sleep(args.pause_seconds)
    finally:
        process_sampler.stop()
        typeperf.terminate()
        try:
            typeperf.wait(timeout=10)
        except subprocess.TimeoutExpired:
            typeperf.kill()
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        reader.join(timeout=3)
        with log_path.open("w", encoding="utf-8") as stream:
            with lines_lock:
                stream.write("\n".join(lines) + "\n")

    headers, system_samples = parse_typeperf(typeperf_path)
    base_samples = [s for s in system_samples if run_start <= s["ts"] <= baseline_end]
    idx_pages = find_counter(headers, r"\memory\pages input/sec")
    idx_bytes = find_counter(headers, r"\physicaldisk(_total)\disk read bytes/sec")
    idx_available = find_counter(headers, r"\memory\available mbytes")
    baseline = {
        "sample_count": len(base_samples),
        "pages_input_per_s": sorted(s["values"][idx_pages] for s in base_samples)[len(base_samples) // 2],
        "disk_read_mib_s": sorted(s["values"][idx_bytes] for s in base_samples)[len(base_samples) // 2] / 2**20,
    }

    for result in requests:
        result.pop("inter_chunk_ms", None)
        system = summarize_system(
            headers, system_samples, result["decode_start"], result["request_end"], baseline
        )
        result["system_decode"] = system
        result["process"] = summarize_process(
            process_sampler.samples, result["request_start"], result["request_end"]
        )
        pressure_samples = [
            sample["values"][idx_available] for sample in system_samples
            if result["request_start"] <= sample["ts"] <= result["request_end"]
        ]
        result["available_memory_mib"] = {
            "minimum": min(pressure_samples) if pressure_samples else None,
            "average": sum(pressure_samples) / len(pressure_samples) if pressure_samples else None,
        }
        phases = result["profile_phases"]
        draft_ms = phases.get("draft_generate", {}).get("total_ms", 0.0)
        draft_ms += phases.get("draft_ingest_verify", {}).get("total_ms", 0.0)
        target_ms = phases.get("target_verify", {}).get("total_ms", 0.0)
        accept_ms = phases.get("relaxed_accept", {}).get("total_ms", 0.0)
        rows = phases.get("target_verify", {}).get("rows", 0)
        hard_mib = system.get("hard_page_input_mib_above_idle", 0.0)
        logical_bytes = rows * EXPERT_BYTES_PER_TARGET_ROW
        result["derived"] = {
            "draft_total_ms": draft_ms,
            "target_verify_ms": target_ms,
            "relaxed_accept_ms": accept_ms,
            "verify_total_ms": target_ms + accept_ms,
            "target_rows": rows,
            "logical_expert_touch_mib": logical_bytes / 2**20,
            "mmap_expert_hit_rate_lower_bound": max(0.0, min(1.0, 1.0 - hard_mib * 2**20 / logical_bytes)) if logical_bytes else None,
        }

    by_name = {item["phase"]: item for item in requests}
    comparisons = {}
    for suffix in ("C", "D"):
        cold = by_name[f"cold_{suffix}"]
        warm = by_name[f"warm_{suffix}"]
        comparisons[suffix] = {
            "cold_expert_load_penalty_ms": max(
                0.0, cold["derived"]["target_verify_ms"] - warm["derived"]["target_verify_ms"]
            ),
            "cold_hard_page_input_mib": cold["system_decode"].get("hard_page_input_mib_above_idle", 0.0),
            "warm_hard_page_input_mib": warm["system_decode"].get("hard_page_input_mib_above_idle", 0.0),
        }

    streaming_summary = parse_streaming_summary(lines)
    if not streaming_summary["cache_phases"]:
        streaming_summary["cache_phases"] = aggregate_request_cache(requests)

    document = {
        "metadata": {
            "command": command, "pid": process.pid, "mode": args.mode, "n_predict": args.tokens,
            "working_set_cap_mib": args.ws_cap_mib if args.mode == "mmap" else args.streaming_budget_mib,
            "expert_bytes_per_target_row": EXPERT_BYTES_PER_TARGET_ROW,
            "hit_rate_definition": "1 - decode hard-page-input bytes / logical expert bytes for all target verify rows; conservative lower bound",
        },
        "idle_baseline": baseline,
        "requests": requests,
        "cold_vs_warm": comparisons,
        "streaming_summary": streaming_summary,
    }
    with args.out.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
    print(json.dumps(document, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
