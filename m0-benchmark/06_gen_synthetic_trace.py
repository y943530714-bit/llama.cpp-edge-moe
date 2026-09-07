#!/usr/bin/env python3

"""Generate a deterministic synthetic MoE route trace for simulator checks."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--zipf", type=float, default=1.0)
    parser.add_argument("--drift", type=float, default=0.02, help="per-token cyclic distribution drift")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def make_route(rng: random.Random, experts: int, topk: int, layer: int, token: int, zipf: float, drift: float) -> tuple[list[int], list[float]]:
    shift = int((layer * 0.37 + token * drift * experts) % experts)
    weights = [1.0 / math.pow(rank + 1, zipf) for rank in range(experts)]
    choices = rng.choices(range(experts), weights=weights, k=max(topk * 4, topk))
    selected: list[int] = []
    for expert in choices:
        value = (expert + shift) % experts
        if value not in selected:
            selected.append(value)
        if len(selected) == topk:
            break
    if len(selected) < topk:
        for expert in range(experts):
            value = (expert + shift) % experts
            if value not in selected:
                selected.append(value)
            if len(selected) == topk:
                break
    raw = [1.0 / (index + 1) for index in range(topk)]
    total = sum(raw)
    return selected, [value / total for value in raw]


def main() -> int:
    args = parse_args()
    if args.layers <= 0 or args.experts <= 0 or args.topk <= 0 or args.topk > args.experts:
        print("layers, experts, and topk are invalid", file=sys.stderr)
        return 2
    if args.prefill_tokens < 0 or args.decode_tokens < 0 or args.zipf <= 0:
        print("prefill/decode token counts and zipf must be valid", file=sys.stderr)
        return 2
    if args.output.exists() and not args.force:
        print(f"output exists, pass --force to replace it: {args.output}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version": 1, "event": "header", "format": "edge-moe-route-v1"}) + "\n")

        for layer in range(args.layers):
            ids: list[list[int]] = []
            weights: list[list[float]] = []
            for token in range(args.prefill_tokens):
                token_ids, token_weights = make_route(rng, args.experts, args.topk, layer, token, args.zipf, args.drift)
                ids.append(token_ids)
                weights.append(token_weights)
            if ids:
                stream.write(json.dumps({
                    "schema_version": 1,
                    "event": "route",
                    "event_id": layer,
                    "layer": layer,
                    "stage": "prefill",
                    "n_tokens": len(ids),
                    "n_experts_used": args.topk,
                    "weights_normalized": True,
                    "expert_ids": ids,
                    "weights": weights,
                }) + "\n")

        event_id = args.layers
        for token in range(args.decode_tokens):
            for layer in range(args.layers):
                token_ids, token_weights = make_route(rng, args.experts, args.topk, layer, token + args.prefill_tokens, args.zipf, args.drift)
                stream.write(json.dumps({
                    "schema_version": 1,
                    "event": "route",
                    "event_id": event_id,
                    "layer": layer,
                    "stage": "decode",
                    "n_tokens": 1,
                    "n_experts_used": args.topk,
                    "weights_normalized": True,
                    "expert_ids": [token_ids],
                    "weights": [token_weights],
                }) + "\n")
                event_id += 1

    print(f"wrote synthetic trace: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
