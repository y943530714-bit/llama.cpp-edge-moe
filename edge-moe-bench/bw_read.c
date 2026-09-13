// read-only bandwidth: sum two arrays with AVX2, 8 threads
#include <windows.h>
#include <stdio.h>
#include <immintrin.h>
#define N_THREADS 8
#define ARR_FLOATS (256*1024*1024)
static float *a, *b;
static float sink;
static DWORD WINAPI read_part(LPVOID p) {
    int id = (int)(intptr_t)p;
    size_t chunk = ARR_FLOATS / N_THREADS, lo = id*chunk, hi = (id==N_THREADS-1)?ARR_FLOATS:lo+chunk;
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = lo; i + 8 <= hi; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_load_ps(a+i), _mm256_load_ps(b+i)));
    }
    float tmp[8]; _mm256_storeu_ps(tmp, acc); float t = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (size_t i = hi & ~(size_t)7; i < hi; i++) t += a[i] + b[i];
    sink += t;
    return 0;
}
int main(void) {
    a = VirtualAlloc(NULL, ARR_FLOATS*4, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
    b = VirtualAlloc(NULL, ARR_FLOATS*4, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
    for (size_t i = 0; i < ARR_FLOATS; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    double best = 1e9;
    for (int r = 0; r < 4; r++) {
        HANDLE th[N_THREADS]; LARGE_INTEGER f, t0, t1;
        QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
        for (int i = 0; i < N_THREADS; i++) th[i] = CreateThread(NULL,0,read_part,(LPVOID)(intptr_t)i,0,NULL);
        WaitForMultipleObjects(N_THREADS, th, TRUE, INFINITE);
        QueryPerformanceCounter(&t1);
        double ms = 1000.0*(double)(t1.QuadPart-t0.QuadPart)/f.QuadPart;
        if (ms < best) best = ms;
        printf("  read rep%d: %.0f ms, %.1f GB/s\n", r, ms, 2.0*ARR_FLOATS*4/(ms*1e6));
    }
    printf("DRAM read best: %.1f GB/s\n", 2.0*ARR_FLOATS*4/(best*1e6));
    return sink > 1e30;
}
