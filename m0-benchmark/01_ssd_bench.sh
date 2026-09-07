#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: 01_ssd_bench.sh [directory] [size_gb]

The benchmark creates and removes one uniquely named temporary file in the
directory. The default directory is ./edge-moe-bench-data and the default size
is 1 GiB.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

bench_dir="${1:-./edge-moe-bench-data}"
size_gb="${2:-1}"

if ! [[ "$size_gb" =~ ^[1-9][0-9]*$ ]]; then
    echo "size_gb must be a positive integer" >&2
    exit 2
fi

mkdir -p -- "$bench_dir"
test_file="$(mktemp "$bench_dir/edge-moe-ssd.XXXXXX")"

cleanup() {
    if [[ -n "${test_file:-}" && -f "$test_file" ]]; then
        rm -f -- "$test_file"
    fi
}
trap cleanup EXIT

size_mb=$((size_gb * 1024))
echo "creating $size_gb GiB test file on: $bench_dir"
dd if=/dev/zero of="$test_file" bs=1M count="$size_mb" conv=fsync >/dev/null 2>&1

python3 - "$test_file" <<'PY'
import os
import platform
import random
import sys
import time

path = sys.argv[1]
size = os.path.getsize(path)
chunk = 8 * 1024 * 1024


def prepare(fd):
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass

    if sys.platform == "darwin":
        try:
            import fcntl
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
        except (ImportError, OSError):
            pass


def report(label, total, elapsed):
    mib_s = total / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
    print(f"{label:12s}: {mib_s:10.1f} MiB/s ({total / (1024 * 1024):.0f} MiB in {elapsed:.3f} s)")


fd = os.open(path, os.O_RDONLY)
try:
    prepare(fd)
    start = time.perf_counter()
    total = 0
    while True:
        data = os.read(fd, chunk)
        if not data:
            break
        total += len(data)
    report("sequential", total, time.perf_counter() - start)
finally:
    os.close(fd)


fd = os.open(path, os.O_RDONLY)
try:
    prepare(fd)
    rng = random.Random(0xED9E)
    reads = max(32, min(512, size // chunk))
    max_offset = max(0, size - chunk)
    start = time.perf_counter()
    total = 0
    for _ in range(reads):
        offset = (rng.randrange(max_offset // chunk + 1) * chunk) if max_offset else 0
        total += len(os.pread(fd, chunk, offset))
    report("random", total, time.perf_counter() - start)
finally:
    os.close(fd)

print(f"platform    : {platform.platform()}")
print(f"test file   : {size / (1024 * 1024):.0f} MiB")
PY
