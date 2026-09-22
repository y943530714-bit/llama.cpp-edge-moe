"""Hold physical memory on Windows until a target amount remains available.

The allocated pages are touched and locked so Windows cannot silently page out
the pressure process while the benchmark is running.  The process prints one
JSON READY record and releases all memory when stdin closes or Ctrl+C is sent.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import sys
import time


MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
QUOTA_LIMITS_HARDWS_MIN_ENABLE = 0x00000001

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
kernel32.VirtualAlloc.restype = wintypes.LPVOID
kernel32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
kernel32.VirtualFree.restype = wintypes.BOOL
kernel32.VirtualLock.argtypes = [wintypes.LPVOID, ctypes.c_size_t]
kernel32.VirtualLock.restype = wintypes.BOOL
kernel32.VirtualUnlock.argtypes = [wintypes.LPVOID, ctypes.c_size_t]
kernel32.VirtualUnlock.restype = wintypes.BOOL
kernel32.SetProcessWorkingSetSizeEx.argtypes = [
    wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD,
]
kernel32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_bytes():
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.ullAvailPhys


def configure_working_set(bytes_to_hold, margin_bytes):
    minimum = bytes_to_hold + margin_bytes
    maximum = minimum + margin_bytes
    if not kernel32.SetProcessWorkingSetSizeEx(
        kernel32.GetCurrentProcess(), minimum, maximum, QUOTA_LIMITS_HARDWS_MIN_ENABLE
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def allocate_locked(size):
    address = kernel32.VirtualAlloc(None, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
    if not address:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        ctypes.memset(address, 0xA5, size)
        if not kernel32.VirtualLock(address, size):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel32.VirtualFree(address, 0, MEM_RELEASE)
        raise
    return address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--available-mib", type=int, required=True)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--tolerance-mib", type=int, default=128)
    parser.add_argument("--max-hold-mib", type=int, default=24576)
    args = parser.parse_args()

    if args.available_mib <= 0 or args.chunk_mib <= 0 or args.tolerance_mib < 0:
        parser.error("memory sizes must be positive")

    mib = 1024 * 1024
    target = args.available_mib * mib
    tolerance = args.tolerance_mib * mib
    chunk = args.chunk_mib * mib
    initial_available = available_bytes()
    wanted = max(0, initial_available - target)
    wanted = min(wanted, args.max_hold_mib * mib)
    wanted = wanted // chunk * chunk
    max_to_hold = min(args.max_hold_mib * mib, wanted + 1024 * mib)

    blocks = []
    locked = 0
    try:
        if max_to_hold:
            configure_working_set(max_to_hold, 256 * mib)
        stable_samples = 0
        deadline = time.time() + 15.0
        while locked < max_to_hold and time.time() < deadline:
            current_available = available_bytes()
            if current_available <= target + tolerance:
                stable_samples += 1
                if stable_samples >= 2:
                    break
                time.sleep(0.5)
                continue
            stable_samples = 0
            size = min(chunk, max_to_hold - locked, current_available - target)
            size = size // (64 * 1024) * (64 * 1024)
            if size <= 0:
                break
            address = allocate_locked(size)
            blocks.append((address, size))
            locked += size

        time.sleep(1.0)
        final_available = available_bytes()
        if final_available > target + tolerance:
            raise RuntimeError(
                f"could not reach requested pressure: available={final_available / mib:.2f} MiB"
            )
        print(json.dumps({
            "status": "READY",
            "initial_available_mib": round(initial_available / mib, 2),
            "target_available_mib": args.available_mib,
            "final_available_mib": round(final_available / mib, 2),
            "locked_mib": round(locked / mib, 2),
            "blocks": len(blocks),
        }), flush=True)

        while sys.stdin.buffer.read(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        for address, size in reversed(blocks):
            kernel32.VirtualUnlock(address, size)
            kernel32.VirtualFree(address, 0, MEM_RELEASE)


if __name__ == "__main__":
    main()
