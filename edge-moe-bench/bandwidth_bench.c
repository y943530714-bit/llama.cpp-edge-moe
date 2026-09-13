// Minimal bandwidth probe: DRAM triad (MT) and SSD sequential/random reads
// with FILE_FLAG_NO_BUFFERING (bypasses the file cache, true device speed).
// windows-only, MSVC: cl /O2 /openmp bandwidth_bench.c /Fe:bandwidth_bench.exe
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static double now_ms(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return 1000.0 * (double)t.QuadPart / (double)f.QuadPart;
}

#define N_THREADS 8
#define ARR_FLOATS (256 * 1024 * 1024) // 1 GiB per array
static float *a, *b, *c;

static DWORD WINAPI triad_part(LPVOID p) {
    int id = (int)(intptr_t)p;
    size_t chunk = ARR_FLOATS / N_THREADS;
    size_t lo = id * chunk, hi = (id == N_THREADS - 1) ? ARR_FLOATS : lo + chunk;
    const float s = 2.0f;
    for (size_t i = lo; i < hi; i++) c[i] = a[i] + s * b[i];
    return 0;
}

static void dram_triad(int reps) {
    a = VirtualAlloc(NULL, ARR_FLOATS * 4, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    b = VirtualAlloc(NULL, ARR_FLOATS * 4, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    c = VirtualAlloc(NULL, ARR_FLOATS * 4, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (size_t i = 0; i < ARR_FLOATS; i++) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.0f; }
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        HANDLE th[N_THREADS];
        double t0 = now_ms();
        for (int i = 0; i < N_THREADS; i++) th[i] = CreateThread(NULL, 0, triad_part, (LPVOID)(intptr_t)i, 0, NULL);
        WaitForMultipleObjects(N_THREADS, th, TRUE, INFINITE);
        double ms = now_ms() - t0;
        if (ms < best) best = ms;
        // bytes: read a + read b + write c = 3 * array size
        printf("  dram triad rep%d: %.0f ms, %.1f GB/s\n", r, ms, 3.0 * ARR_FLOATS * 4 / (ms * 1e6));
    }
    printf("DRAM triad best: %.1f GB/s (theoretical DDR4-3200 dual-channel peak: 51.2 GB/s)\n", 3.0 * ARR_FLOATS * 4 / (best * 1e6));
}

static int align_buf(unsigned char **p, size_t size) {
    *p = VirtualAlloc(NULL, size + 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return *p ? 0 : 1;
}

static void ssd_seq(const wchar_t *path, uint64_t bytes) {
    HANDLE h = CreateFileW(path, FILE_READ_DATA, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING, NULL);
    if (h == INVALID_HANDLE_VALUE) { printf("  ssd seq: open failed %lu\n", GetLastError()); return; }
    const size_t chunk = 1024 * 1024;
    unsigned char *buf; if (align_buf(&buf, chunk)) { printf("alloc fail\n"); return; }
    uint64_t off = 4ull << 30; // start 4 GiB into the file
    double t0 = now_ms();
    uint64_t got = 0;
    while (got < bytes) {
        DWORD rd = 0;
        if (!ReadFile(h, buf, (DWORD)chunk, &rd, NULL) || rd == 0) break;
        got += rd;
    }
    double ms = now_ms() - t0;
    CloseHandle(h);
    printf("SSD sequential read: %.2f GiB in %.0f ms = %.1f MB/s (NO_BUFFERING)\n",
           got / 1073741824.0, ms, got / 1048576.0 / (ms / 1000.0));
}

static void ssd_rand(const wchar_t *path, int ops) {
    HANDLE h = CreateFileW(path, FILE_READ_DATA, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING, NULL);
    if (h == INVALID_HANDLE_VALUE) { printf("  ssd rand: open failed %lu\n", GetLastError()); return; }
    const DWORD chunk = 4096;
    unsigned char *buf; if (align_buf(&buf, chunk)) { printf("alloc fail\n"); return; }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(h, &fsz) || fsz.QuadPart < 16 * chunk) { printf("  ssd rand: file too small\n"); return; }
    uint64_t lo = 0, hi = (uint64_t)fsz.QuadPart; // random region across the whole file
    srand(42);
    double t0 = now_ms();
    uint64_t tot = 0;
    for (int i = 0; i < ops; i++) {
        LARGE_INTEGER off;
        off.QuadPart = lo + (uint64_t)(rand() % 100000) * chunk * 10 % (hi - lo);
        off.QuadPart -= off.QuadPart % chunk;
        OVERLAPPED ov = {0};
        ov.Offset = off.LowPart; ov.OffsetHigh = off.HighPart;
        DWORD rd = 0;
        if (!ReadFile(h, buf, chunk, &rd, &ov)) { if (GetLastError() != ERROR_IO_PENDING) break; GetOverlappedResult(h, &ov, &rd, TRUE); }
        tot += rd;
    }
    double ms = now_ms() - t0;
    CloseHandle(h);
    printf("SSD random 4KB QD1: %d ops in %.0f ms = %.0f IOPS, %.1f MB/s, avg latency %.2f ms\n",
           ops, ms, ops / (ms / 1000.0), tot / 1048576.0 / (ms / 1000.0), ms / ops);
}

int main(int argc, char **argv) {
    wchar_t ssd_file[512] = L"\\\\?\\D:\\workspace\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    if (argc > 2) MultiByteToWideChar(CP_UTF8, 0, argv[2], -1, ssd_file, 512);
    if (argc > 1 && strcmp(argv[1], "dram") == 0) { dram_triad(3); return 0; }
    if (argc > 1 && strcmp(argv[1], "seq") == 0) { ssd_seq(ssd_file, argc > 3 ? _atoi64(argv[3]) : 3ull << 30); return 0; }
    if (argc > 1 && strcmp(argv[1], "rand") == 0) { ssd_rand(ssd_file, 1000); return 0; }
    printf("usage: bandwidth_bench dram|seq|rand [file]\n");
    return 1;
}
