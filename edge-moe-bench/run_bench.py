# Edge MoE benchmark driver: runs llama-cli per HumanEval problem and parses
# the slot timing lines printed with --verbose --perf.
#
# -n 2 is used per the user's fast-test instruction (n=1 performs no decode
# step at all, so one decode step is the minimum measurable TPOT).
import argparse
import ctypes
import csv
import json
import os
import re
import subprocess
import sys
import time

PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
QUOTA_LIMITS_HARDWS_MAX_ENABLE  = 0x00000004


def set_hard_ws_cap(pid, cap_bytes):
    """Force a hard maximum working set so the process can never hold more
    resident memory than the edge device budget allows."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    if not h:
        return False
    try:
        ok = k32.SetProcessWorkingSetSizeEx(
            h,
            ctypes.c_size_t(64 * 1024 * 1024),
            ctypes.c_size_t(cap_bytes),
            ctypes.c_uint(QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE),
        )
        return bool(ok)
    finally:
        k32.CloseHandle(h)


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


RE_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_EVAL = re.compile(r"eval time =\s*([\d.]+) ms /\s*(\d+) tokens")


def run_one(cli, model, prompt, args_extra, n_predict, threads, ctx, log_path, ws_cap_gb):
    cmd = [
        cli,
        "-m", model,
        "-p", prompt,
        "-n", str(n_predict),
        "-t", str(threads),
        "--ctx-size", str(ctx),
        "--no-warmup",
        "--perf",
        "--verbose",
        "--single-turn",
        "--simple-io",
    ] + args_extra
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if ws_cap_gb > 0:
            ok = set_hard_ws_cap(proc.pid, int(ws_cap_gb * (1024 ** 3)))
            if not ok:
                proc.kill()
                raise RuntimeError("failed to set the hard working set cap")
        ret = proc.wait(timeout=3600)
    wall = time.time() - t0
    if ret != 0:
        with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
            tail = lf.read()[-800:]
        raise RuntimeError(f"llama-cli exit {ret}: {tail}")

    txt = open(log_path, "r", encoding="utf-8", errors="replace").read()
    mp = RE_PROMPT.search(txt)
    me = RE_EVAL.search(txt)
    # eval-time regex also matches "prompt eval time"; take the LAST match
    me = None
    for m in RE_EVAL.finditer(txt):
        me = m
    prompt_ms = float(mp.group(1)) if mp else None
    prompt_n = int(mp.group(2)) if mp else 0
    eval_ms = float(me.group(1)) if me else None
    eval_n = int(me.group(2)) if me else 0
    return {
        "prompt_ms": prompt_ms, "prompt_n": prompt_n,
        "eval_ms": eval_ms, "eval_n": eval_n,
        "tpot_ms": (eval_ms / eval_n) if eval_ms and eval_n else None,
        "wall_s": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="build-win/bin/Release/llama-cli.exe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="HumanEval.jsonl")
    ap.add_argument("--n-problems", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--predict", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--extra", default="",
                    help="extra llama-cli args as a single string, e.g. \"--temp 0 --moe-skip-k1 4\";"
                         " {task} inside values is replaced by the task id")
    ap.add_argument("--out", default="edge-moe-results/bench.csv")
    ap.add_argument("--log-dir", default="edge-moe-results/logs")
    ap.add_argument("--min-free-gb", type=float, default=1.2)
    ap.add_argument("--ws-cap-gb", type=float, default=4.5,
                    help="hard working set cap in GB for each llama-cli process (0 = off)")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

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

    rows = []
    extra_args = args.extra.split() if args.extra else []
    for n, prob in enumerate(picked):
        free = free_mem_gb()
        if free < args.min_free_gb:
            print(f"ABORT: free memory {free:.2f} GB below {args.min_free_gb} GB", file=sys.stderr)
            sys.exit(2)

        log_path = os.path.join(args.log_dir, f"{args.config_name}_{prob['task_id'].replace('/', '_')}.log")
        # {task} in extra args is replaced by the task id, so each problem
        # can write its own trace file
        extra = [v.replace("{task}", prob["task_id"].replace("/", "_")) for v in extra_args]
        try:
            r = run_one(args.cli, args.model, prob["prompt"], extra,
                        args.predict, args.threads, args.ctx, log_path, args.ws_cap_gb)
        except RuntimeError as e:
            print(f"FAILED on {prob['task_id']}: {e}", file=sys.stderr)
            sys.exit(3)

        row = {"config": args.config_name, "task_id": prob["task_id"],
               "prompt_n": r["prompt_n"], "prompt_ms": r["prompt_ms"],
               "eval_n": r["eval_n"], "eval_ms": r["eval_ms"],
               "tpot_ms": r["tpot_ms"], "wall_s": r["wall_s"],
               "free_gb_before": round(free, 2)}
        rows.append(row)
        print(f"[{n + 1}/{len(picked)}] {prob['task_id']}: "
              f"prefill={r['prompt_ms']:.0f} ms ({r['prompt_n']} tok), "
              f"decode={r['eval_ms']:.0f} ms / {r['eval_n']} tok, "
              f"tpot={r['tpot_ms']:.0f} ms, wall={r['wall_s']}s", flush=True)

    # summary: pooled and per-problem
    tpots = [r["tpot_ms"] for r in rows if r["tpot_ms"]]
    prefills = [r["prompt_ms"] for r in rows if r["prompt_ms"]]
    summary = {
        "config": args.config_name,
        "extra": args.extra,
        "n_problems": len(rows),
        "predict": args.predict,
        "mean_tpot_ms": round(sum(tpots) / len(tpots), 1) if tpots else None,
        "median_tpot_ms": round(sorted(tpots)[len(tpots) // 2], 1) if tpots else None,
        "mean_prefill_ms": round(sum(prefills) / len(prefills), 1) if prefills else None,
    }

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(args.log_dir, f"{args.config_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
