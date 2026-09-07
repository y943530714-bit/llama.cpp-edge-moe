#!/usr/bin/env python3

"""Analyze route traces and simulate a fixed-size expert cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

Key = tuple[int, int]


@dataclass
class RouteEvent:
    layer: int
    stage: str
    experts: list[int]
    weights: list[float]


@dataclass
class SimStats:
    route_events: int = 0
    selected_experts: int = 0
    resident_hits: int = 0
    prefetched_hits: int = 0
    demand_misses: int = 0
    prefetch_requests: int = 0
    demand_requests: int = 0
    suppressed_prefetch: int = 0
    wait_events: int = 0
    evictions: int = 0
    tokens: int = 0

    def add(self, other: "SimStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def load_trace(path: Path) -> list[RouteEvent]:
    events: list[RouteEvent] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if record.get("event") != "route":
                continue
            ids = record.get("expert_ids")
            weights = record.get("weights")
            if not isinstance(ids, list) or not isinstance(weights, list) or len(ids) != len(weights):
                raise ValueError(f"{path}:{line_number}: expert_ids and weights must have equal lengths")
            stage = str(record.get("stage", "decode" if len(ids) == 1 else "prefill"))
            for token_ids, token_weights in zip(ids, weights):
                if not isinstance(token_ids, list) or not isinstance(token_weights, list):
                    raise ValueError(f"{path}:{line_number}: route arrays must be nested by token")
                if len(token_ids) != len(token_weights) or not token_ids:
                    raise ValueError(f"{path}:{line_number}: route IDs and weights have inconsistent lengths")
                events.append(RouteEvent(
                    layer=int(record["layer"]),
                    stage=stage,
                    experts=[int(value) for value in token_ids],
                    weights=[float(value) for value in token_weights],
                ))
    return events


def count_profile(events: Iterable[RouteEvent]) -> dict[int, Counter[int]]:
    profile: dict[int, Counter[int]] = {}
    for event in events:
        counter = profile.setdefault(event.layer, Counter())
        for expert in event.experts:
            counter[expert] += 1
    return profile


def top_keys(profile: dict[int, Counter[int]], topk: int) -> set[Key]:
    result: set[Key] = set()
    if topk <= 0:
        return result
    for layer, counter in profile.items():
        result.update((layer, expert) for expert, _ in counter.most_common(topk))
    return result


@dataclass
class ExpertCache:
    capacity: int
    policy: str
    alpha: float
    pinned: set[Key]
    resident: OrderedDict[Key, None] = field(default_factory=OrderedDict)
    prefetched_resident: set[Key] = field(default_factory=set)
    loading: dict[Key, int] = field(default_factory=dict)
    scores: dict[Key, float] = field(default_factory=dict)
    last_used: dict[Key, int] = field(default_factory=dict)
    stats: SimStats = field(default_factory=SimStats)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("cache capacity must be positive")
        if len(self.pinned) > self.capacity:
            raise ValueError("pinned floor is larger than cache capacity")
        for key in sorted(self.pinned):
            self.resident[key] = None

    def touch(self, key: Key, weight: float, now: int) -> None:
        self.last_used[key] = now
        self.scores[key] = self.alpha * self.scores.get(key, 0.0) + (1.0 - self.alpha) * max(weight, 0.0)
        if key in self.resident:
            self.resident.move_to_end(key)

    def complete(self, now: int) -> None:
        ready = [key for key, ready_at in self.loading.items() if ready_at <= now]
        for key in sorted(ready):
            del self.loading[key]
            self._make_room()
            self.resident[key] = None
            self.prefetched_resident.add(key)
            self.resident.move_to_end(key)

    def _make_room(self) -> bool:
        while len(self.resident) + len(self.loading) >= self.capacity:
            candidates = [key for key in self.resident if key not in self.pinned]
            if not candidates:
                return False
            if self.policy == "lru":
                victim = next(key for key in self.resident if key not in self.pinned)
            else:
                victim = min(candidates, key=lambda key: (self.scores.get(key, 0.0), self.last_used.get(key, -1)))
            del self.resident[victim]
            self.prefetched_resident.discard(victim)
            self.stats.evictions += 1
        return True

    def schedule_prefetch(self, key: Key, ready_at: int) -> bool:
        if key in self.resident or key in self.loading:
            return True
        if not self._make_room():
            self.stats.suppressed_prefetch += 1
            return False
        self.loading[key] = ready_at
        self.stats.prefetch_requests += 1
        return True

    def access(self, key: Key, weight: float, now: int) -> None:
        self.complete(now)
        self.stats.selected_experts += 1
        if key in self.resident:
            if key in self.prefetched_resident:
                self.stats.prefetched_hits += 1
                self.prefetched_resident.discard(key)
            else:
                self.stats.resident_hits += 1
            self.touch(key, weight, now)
            return
        if key in self.loading:
            ready_at = self.loading[key]
            if ready_at <= now:
                self.complete(now)
                if key in self.resident:
                    self.stats.prefetched_hits += 1
                    self.touch(key, weight, now)
                    return
            self.stats.wait_events += 1
            self.stats.demand_misses += 1
            self.complete(ready_at)
            if key in self.resident:
                self.prefetched_resident.discard(key)
                self.touch(key, weight, ready_at)
            return

        self.stats.demand_misses += 1
        self.stats.demand_requests += 1
        if not self._make_room():
            if self.loading:
                self.complete(min(self.loading.values()))
            if not self._make_room():
                raise RuntimeError("cache has no evictable slot for a demand miss")
        self.resident[key] = None
        self.prefetched_resident.discard(key)
        self.resident.move_to_end(key)
        self.touch(key, weight, now)


def frequency_prefetch_keys(
        event: RouteEvent,
        profile: dict[int, Counter[int]],
        max_layer: int,
        topk: int,
        horizon: int) -> set[Key]:
    if max_layer < 0 or topk <= 0 or horizon <= 0:
        return set()
    result: set[Key] = set()
    for offset in range(1, horizon + 1):
        layer = (event.layer + offset) % (max_layer + 1)
        result.update((layer, expert) for expert, _ in profile.get(layer, Counter()).most_common(topk))
    return result


def simulate(
        events: list[RouteEvent],
        stage: str,
        capacity: int,
        policy: str,
        alpha: float,
        prefetch: str,
        prefetch_topk: int,
        prefetch_horizon: int,
        prefetch_latency: int,
        pin_topk: int) -> SimStats:
    selected_events = [event for event in events if stage == "all" or event.stage == stage]
    if not selected_events:
        return SimStats()

    profile = count_profile(selected_events)
    pinned = top_keys(profile, pin_topk)
    cache = ExpertCache(capacity, policy, alpha, pinned)
    max_layer = max(event.layer for event in selected_events)

    for index, event in enumerate(selected_events):
        cache.stats.route_events += 1
        if event.stage == "decode" and event.layer == max_layer:
            cache.stats.tokens += 1

        if prefetch == "oracle":
            future = selected_events[index + 1:index + 1 + prefetch_horizon]
            candidates = {(future_event.layer, expert)
                          for future_event in future for expert in future_event.experts}
        elif prefetch == "frequency":
            candidates = frequency_prefetch_keys(
                event, profile, max_layer, prefetch_topk, prefetch_horizon)
        else:
            candidates = set()

        for key in sorted(candidates):
            cache.schedule_prefetch(key, index + prefetch_latency)

        for expert, weight in zip(event.experts, event.weights):
            cache.access((event.layer, expert), weight, index)

    return cache.stats


def summary(stats: SimStats, expert_bytes: int, ssd_mb_s: float, compute_ms: float, overlap: bool) -> dict[str, object]:
    selected = stats.selected_experts
    hit_rate = (stats.resident_hits + stats.prefetched_hits) / selected if selected else 0.0
    resident_hit_rate = stats.resident_hits / selected if selected else 0.0
    io_rate = stats.demand_misses / selected if selected else 0.0
    prefetch_hit_rate = stats.prefetched_hits / selected if selected else 0.0
    prefetch_mb = stats.prefetch_requests * expert_bytes / (1024 * 1024)
    demand_mb = stats.demand_requests * expert_bytes / (1024 * 1024)
    if stats.tokens > 0:
        io_ms = demand_mb / stats.tokens / ssd_mb_s * 1000.0
        step_ms = max(compute_ms, io_ms) if overlap else compute_ms + io_ms
        estimated_tok_s = 1000.0 / step_ms if step_ms > 0 else 0.0
    else:
        step_ms = 0.0
        estimated_tok_s = 0.0
    return {
        "route_events": stats.route_events,
        "selected_experts": selected,
        "resident_hits": stats.resident_hits,
        "prefetched_hits": stats.prefetched_hits,
        "demand_misses": stats.demand_misses,
        "hit_rate": hit_rate,
        "resident_hit_rate": resident_hit_rate,
        "io_rate": io_rate,
        "prefetch_hit_rate": prefetch_hit_rate,
        "prefetch_requests": stats.prefetch_requests,
        "demand_requests": stats.demand_requests,
        "prefetch_mb": prefetch_mb,
        "demand_mb": demand_mb,
        "wait_events": stats.wait_events,
        "evictions": stats.evictions,
        "suppressed_prefetch": stats.suppressed_prefetch,
        "estimated_step_ms": step_ms,
        "estimated_tok_s": estimated_tok_s,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--stage", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--capacity-gb", type=float, default=3.0)
    parser.add_argument("--capacity-slots", type=int, default=0, help="override capacity derived from --capacity-gb")
    parser.add_argument("--expert-mb", type=float, default=1.65)
    parser.add_argument("--policy", choices=("lru", "ema"), default="ema")
    parser.add_argument("--ema-alpha", type=float, default=0.9)
    parser.add_argument("--prefetch", choices=("none", "frequency", "oracle"), default="none")
    parser.add_argument("--prefetch-topk", type=int, default=8)
    parser.add_argument("--prefetch-horizon", type=int, default=20)
    parser.add_argument("--prefetch-latency", type=int, default=0)
    parser.add_argument("--pin-topk", type=int, default=0)
    parser.add_argument("--ssd-mb-s", type=float, default=2700.0)
    parser.add_argument("--compute-ms", type=float, default=10.0)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a human-readable summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.capacity_gb <= 0 or args.expert_mb <= 0 or args.ssd_mb_s <= 0 or args.compute_ms <= 0:
        print("capacity, expert size, SSD bandwidth, and compute time must be positive", file=sys.stderr)
        return 2
    if not 0.0 <= args.ema_alpha < 1.0:
        print("--ema-alpha must be in [0, 1)", file=sys.stderr)
        return 2
    if args.capacity_slots < 0 or args.prefetch_topk < 0 or args.prefetch_horizon < 0 or args.prefetch_latency < 0 or args.pin_topk < 0:
        print("slot and prefetch parameters must be non-negative", file=sys.stderr)
        return 2

    all_stats = SimStats()
    route_count = 0
    stage_counts = Counter()
    capacity = args.capacity_slots or max(1, int(args.capacity_gb * 1024 / args.expert_mb))
    expert_bytes = int(args.expert_mb * 1024 * 1024)

    for trace in args.traces:
        events = load_trace(trace)
        route_count += len(events)
        stage_counts.update(event.stage for event in events)
        stats = simulate(
            events,
            args.stage,
            capacity,
            args.policy,
            args.ema_alpha,
            args.prefetch,
            args.prefetch_topk,
            args.prefetch_horizon,
            args.prefetch_latency,
            args.pin_topk,
        )
        all_stats.add(stats)

    result = summary(all_stats, expert_bytes, args.ssd_mb_s, args.compute_ms, not args.no_overlap)
    result.update({
        "traces": [str(path) for path in args.traces],
        "stage": args.stage,
        "capacity_slots": capacity,
        "capacity_gb": capacity * args.expert_mb / 1024,
        "policy": args.policy,
        "prefetch": args.prefetch,
        "prefetch_horizon": args.prefetch_horizon,
        "prefetch_latency": args.prefetch_latency,
        "pin_topk": args.pin_topk,
        "input_route_events": route_count,
        "input_stage_counts": dict(stage_counts),
    })
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"trace route events : {route_count}")
        print(f"simulated stage    : {args.stage}")
        print(f"cache capacity     : {capacity} slots ({result['capacity_gb']:.3f} GiB)")
        print(f"policy / prefetch  : {args.policy} / {args.prefetch}")
        print(f"selected experts   : {all_stats.selected_experts}")
        print(f"total hit rate     : {result['hit_rate']:.4f}")
        print(f"resident hit rate  : {result['resident_hit_rate']:.4f}")
        print(f"demand IO rate     : {result['io_rate']:.4f}")
        print(f"prefetch hit rate  : {result['prefetch_hit_rate']:.4f}")
        print(f"demand / prefetch  : {all_stats.demand_requests} / {all_stats.prefetch_requests}")
        print(f"estimated tok/s    : {result['estimated_tok_s']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
