#!/usr/bin/env python3

"""Run llama-cli with the edge MoE route trace callback enabled."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-cli", default="./build/bin/llama-cli")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file")
    parser.add_argument("--trace", required=True, help="output JSONL path")
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run llama-cli's warmup before tracing (disabled by default)",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing trace")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cli = Path(args.llama_cli)
    model = Path(args.model)
    trace = Path(args.trace)
    if not cli.is_file() or not cli.stat().st_mode & 0o111:
        print(f"llama-cli is not executable: {cli}", file=sys.stderr)
        return 2
    if not model.is_file():
        print(f"model does not exist: {model}", file=sys.stderr)
        return 2
    if trace.exists() and not args.force:
        print(f"trace already exists, pass --force to replace it: {trace}", file=sys.stderr)
        return 2
    if args.predict < 0 or args.max_events < 0:
        print("--predict and --max-events must be non-negative", file=sys.stderr)
        return 2

    trace.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli),
        "-m", str(model),
        "-n", str(args.predict),
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "-ngl", args.gpu_layers,
        "--moe-trace", str(trace),
    ]
    if args.prompt is not None:
        command.extend(["-p", args.prompt])
    else:
        command.extend(["-f", args.prompt_file])
    if args.max_events:
        command.extend(["--moe-trace-max-events", str(args.max_events)])
    if not args.warmup:
        command.append("--no-warmup")

    print("command:", shlex.join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return result.returncode

    route_events = 0
    with trace.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "route":
                route_events += 1
    print(f"route events: {route_events}")
    print(f"trace: {trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
