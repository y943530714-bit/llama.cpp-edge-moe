#!/usr/bin/env python3

"""Probe the largest amount of anonymous memory that can be mlocked."""

from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import platform
import resource
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gb", type=float, default=8.0, help="maximum probe size (default: 8 GiB)")
    parser.add_argument("--step-mb", type=int, default=256, help="allocation step (default: 256 MiB)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.name == "nt":
        print("mlock probing is not implemented on Windows", file=sys.stderr)
        return 2
    if args.max_gb <= 0 or args.step_mb <= 0:
        print("--max-gb and --step-mb must be positive", file=sys.stderr)
        return 2

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munlock.restype = ctypes.c_int
    memset = ctypes.memset
    memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]

    if hasattr(resource, "RLIMIT_MEMLOCK"):
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        print(f"rlimit_memlock: soft={soft} hard={hard} bytes")
    print(f"platform      : {platform.platform()}")
    print(f"probe limit   : {args.max_gb:.2f} GiB")
    print(f"probe step    : {args.step_mb} MiB")
    print("warning       : this temporarily reserves mlocked memory")

    step = args.step_mb * 1024 * 1024
    limit = int(args.max_gb * 1024 * 1024 * 1024)
    mappings: list[tuple[mmap.mmap, int, int]] = []
    locked = 0

    try:
        while locked < limit:
            current = min(step, limit - locked)
            mapping = mmap.mmap(-1, current, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            memset(pointer, 0, current)
            result = libc.mlock(pointer, current)
            if result != 0:
                error = ctypes.get_errno()
                print(f"mlock failed at {locked / (1024 ** 3):.3f} GiB: errno={error} ({os.strerror(error)})")
                mapping.close()
                break
            mappings.append((mapping, pointer, current))
            locked += current
            print(f"locked       : {locked / (1024 ** 3):.3f} GiB")
    finally:
        for mapping, pointer, current in reversed(mappings):
            libc.munlock(pointer, current)
            mapping.close()

    print(f"locked_peak  : {locked / (1024 ** 3):.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
