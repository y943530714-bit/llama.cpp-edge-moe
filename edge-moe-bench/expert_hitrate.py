# Expert hit-rate analysis for the edge MoE expert-skipping experiments.
#
# Reads a route trace (JSONL from --moe-trace) and measures how well a
# per-layer "hot window" of the top-k2 experts covers the router selections,
# in particular the ranks k1..k used experts that the k1 skip discards.
#
# A high hit rate on the skipped ranks means the skipped experts are mostly
# recurring hot experts, so a resident-window variant could recover quality
# at a modest compute cost. A low hit rate means the skipped selections are
# cold tails and there is little left to exploit.
import argparse
import json
import sys
from collections import defaultdict


def load_routes(path, stage):
    routes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") != "route":
                continue
            if stage != "all" and rec.get("stage") != stage:
                continue
            ids = rec["expert_ids"]
            w = rec.get("weights")
            for t in range(rec.get("n_tokens", len(ids))):
                tok_ids = ids[t]
                tok_w = w[t] if w else None
                routes.append((rec["layer"], tok_ids, tok_w))
    return routes


def layer_windows(routes, k2):
    # window = top-k2 experts per layer by selection frequency, ties broken by weight mass
    counts = defaultdict(lambda: defaultdict(int))
    mass = defaultdict(lambda: defaultdict(float))
    for layer, ids, ws in routes:
        for pos, e in enumerate(ids):
            counts[layer][e] += 1
            if ws:
                mass[layer][e] += ws[pos]
    windows = {}
    for layer, c in counts.items():
        ranked = sorted(c.keys(), key=lambda e: (-c[e], -mass[layer].get(e, 0.0), e))
        windows[layer] = ranked[:k2]
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--k1", type=int, default=4)
    ap.add_argument("--k-used", type=int, default=8, help="experts selected by the router per token")
    ap.add_argument("--k2", type=int, default=16)
    ap.add_argument("--window-sweep", default="8,12,16,24,32,64")
    ap.add_argument("--stage", default="decode", choices=["decode", "prefill", "all"])
    args = ap.parse_args()

    routes = load_routes(args.trace, args.stage)
    if not routes:
        print("no route records found", file=sys.stderr)
        sys.exit(1)

    n_sel = min(args.k_used, len(routes[0][1]))
    keep_ranks = list(range(min(args.k1, n_sel)))
    skip_ranks = list(range(min(args.k1, n_sel), n_sel))

    windows16 = layer_windows(routes, max(args.k2, max(
        int(x) for x in args.window_sweep.split(",") if x)))

    # coverage of the kept ranks (sanity: should be 1.0 for k2 >= usage spread)
    def coverage(window_size, ranks):
        hit = 0
        total = 0
        for layer, ids, _ in routes:
            win = set(windows16[layer][:window_size])
            for r in ranks:
                if r >= len(ids):
                    continue
                total += 1
                if ids[r] in win:
                    hit += 1
        return hit / total if total else 0.0, total

    print(f"routes: {len(routes)} token-layer selections, stage={args.stage}")
    print(f"kept ranks 0..{keep_ranks[-1] if keep_ranks else -1}, skipped ranks "
          f"{skip_ranks[0] if skip_ranks else '-'}..{skip_ranks[-1] if skip_ranks else '-'}")
    print()

    print("coverage by window size (fraction of selections inside the per-layer hot window):")
    print(f"{'window':>8} {'kept-ranks':>12} {'skipped-ranks':>15} {'all-ranks':>11}")
    sweep = sorted(set([args.k2] + [int(x) for x in args.window_sweep.split(",")]))
    results = {}
    for wsize in sweep:
        cov_keep, _ = coverage(wsize, keep_ranks)
        cov_skip, _ = coverage(wsize, skip_ranks)
        cov_all, _ = coverage(wsize, list(range(n_sel)))
        results[wsize] = {"kept": round(cov_keep, 4), "skipped": round(cov_skip, 4), "all": round(cov_all, 4)}
        print(f"{wsize:>8} {cov_keep:>12.4f} {cov_skip:>15.4f} {cov_all:>11.4f}")

    # headline metric at the requested k2
    cov_skip_k2, n_skip = coverage(args.k2, skip_ranks)
    cov_all_k2, n_all = coverage(args.k2, list(range(n_sel)))

    # per-expert usage concentration per layer
    n_layers = len(windows16)
    used_per_layer = [len(windows16[l]) for l in sorted(windows16)]

    extra_compute = cov_skip_k2 * len(skip_ranks)
    summary = {
        "trace": args.trace,
        "stage": args.stage,
        "k1": args.k1,
        "k_used": n_sel,
        "k2": args.k2,
        "skipped_rank_hits_at_k2": round(cov_skip_k2, 4),
        "all_rank_hits_at_k2": round(cov_all_k2, 4),
        "skipped_selections_total": n_skip,
        "extra_experts_per_token_if_window_kept": round(extra_compute, 3),
        "experts_seen_per_layer_min": min(used_per_layer),
        "experts_seen_per_layer_max": max(used_per_layer),
        "n_layers": n_layers,
        "sweep": results,
    }
    print()
    print(json.dumps(summary, indent=2))

    verdict = (
        f"skipped-rank hit rate at window={args.k2}: {cov_skip_k2:.3f} "
        f"(extra compute if window members were computed: ~{extra_compute:.2f} experts/token)"
    )
    print()
    print(verdict)
    if cov_skip_k2 >= 0.7:
        print("interpretation: skipped experts are mostly hot recurring experts - a "
              "resident-window or verify-style refinement has headroom to recover quality")
    elif cov_skip_k2 <= 0.3:
        print("interpretation: skipped selections are mostly cold-tail experts - the k1 skip "
              "already captures most of the benefit, little room left")
    else:
        print("interpretation: mixed - partial headroom depends on the quality cost of the cold half")


if __name__ == "__main__":
    main()
