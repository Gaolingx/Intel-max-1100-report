//===----------------------------------------------------------------------===//
// host_stream.c —— 主机内存带宽基线（真正的 STREAM Triad 类测试）
//
// 为什么不用 /usr/bin/stream？
//   这台 Ubuntu 上 /usr/bin/stream 是 **ImageMagick 的 stream 命令**（图像处理），
//   不是 UVA STREAM 基准。不能拿来测内存带宽（会输出 ImageMagick 的 usage）。
//   所以这里自己实现一份最小 STREAM：Copy / Scale / Add / Triad，
//   用 OpenMP 并行，数组大小远超 LLC（默认每个数组 512 MiB）。
//
// 意义：GPU 的 H2D/D2H 只能跑到 PCIe 上限（~32 GB/s），而主机内存带宽通常在
//       几十 ~ 上百 GB/s。这个基线决定了：
//         · 主机侧预处理（dataloader、tokenizer）会不会成为瓶颈
//         · 多卡情况下主机内存带宽够不够喂 GPU
//       也用于和 docs/hardware.md 里“只有 45 GiB 主机内存”的结论互相印证。
//
// 编译 / 运行
//   gcc -O3 -fopenmp -o host_stream host_stream.c
//   ./host_stream [N_elements] [repeats] [threads]
//     默认 N = 64M doubles (512 MiB/数组)，repeats = 5，threads = {1,8,32,72,144}
//     给了 threads 就只跑那一档（方便 Python 侧扫数组大小）
//
// ⚠ 数组大小极度敏感：本机 **L3 = 432 MiB**，三个数组合计 ≤432 MiB 时测到的是 L3 带宽
//   （虚高 3~5 倍）。要测真 DRAM 必须让单数组 ≥ 256 MiB、合计远大于 L3。
// 输出：每个 kernel 一行 `HOSTBW {json}`
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#if 0  // 废弃：C99 不允许 #pragma 出现在宏展开中（gcc 报 "for statement expected before 'do'"）
// 计时内核：跑 repeats 次取最小值（减少噪声）
#define TIME_KERNEL(body, moved_bytes, threads)                                   \
    do {                                                                          \
        double best = 1e300;                                                      \
        for (int rep = 0; rep < repeats; ++rep) {                                 \
            const double t0 = now_s();                                            \
            body;                                                                 \
            const double dt = now_s() - t0;                                       \
            if (dt < best) best = dt;                                             \
        }                                                                         \
        printf("HOSTBW {\"kernel\":\"%s\",\"threads\":%d,\"n\":%zu,"             \
               "\"array_MiB\":%.1f,\"moved_bytes\":%zu,\"seconds\":%.6f,"         \
               "\"gbps\":%.2f}\n",                                                \
               name, (threads), n, (double)(n * sizeof(double)) / (1 << 20),      \
               (size_t)(moved_bytes), best,                                       \
               (double)(moved_bytes) / best / 1e9);                               \
        fflush(stdout);                                                           \
    } while (0)

#endif  // 废弃宏结束

// 内核写成独立函数，`#pragma omp parallel for` 直接写在函数体里（合法）。
static void k_copy(double *a, double *b, double *c, size_t n, double s) {
    (void)s;
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) c[i] = a[i];
}
static void k_scale(double *a, double *b, double *c, size_t n, double s) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) b[i] = s * c[i];
}
static void k_add(double *a, double *b, double *c, size_t n, double s) {
    (void)s;
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
static void k_triad(double *a, double *b, double *c, size_t n, double s) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
}

typedef void (*kernel_fn)(double *, double *, double *, size_t, double);

// 跑一个内核 repeats 次取最优，输出一行 HOSTBW json。
// io_factor: copy/scale=2（读1写1），add/triad=3（读2写1）
static void report(const char *name, kernel_fn fn, size_t n, int repeats, int threads,
                   double scalar, double io_factor) {
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (!a || !b || !c) { free(a); free(b); free(c); return; }
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) { a[i] = 1.0 + (double)(i % 1024); b[i] = 2.0; c[i] = 0.0; }

    omp_set_num_threads(threads);
    double best = 1e300;
    for (int rep = 0; rep < repeats; ++rep) {
        const double t0 = now_s();
        fn(a, b, c, n, scalar);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double moved = io_factor * (double)n * sizeof(double);
    printf("HOSTBW {\"kernel\":\"%s\",\"threads\":%d,\"n\":%zu,\"array_MiB\":%.1f,"
           "\"io_factor\":%.0f,\"moved_bytes\":%.0f,\"seconds\":%.6f,\"gbps\":%.2f}\n",
           name, threads, n, (double)n * sizeof(double) / (1 << 20), io_factor, moved, best,
           moved / best / 1e9);
    fflush(stdout);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv) {
    const size_t n = (argc > 1) ? strtoull(argv[1], NULL, 0) : ((size_t)64 << 20);
    const int repeats = (argc > 2) ? atoi(argv[2]) : 5;
    const int only_threads = (argc > 3) ? atoi(argv[3]) : 0;
    const double scalar = 2.0;

    printf("# host_stream: n=%zu (%.1f MiB/array, total %.1f MiB), omp_max_threads=%d\n",
           n, (double)n * sizeof(double) / (1 << 20), 3.0 * n * sizeof(double) / (1 << 20),
           omp_get_max_threads());

    const int all_threads[] = {1, 8, 32, 72, 144};
    const int nt = only_threads ? 1 : (int)(sizeof(all_threads) / sizeof(all_threads[0]));
    for (int t = 0; t < nt; ++t) {
        const int threads = only_threads ? only_threads : all_threads[t];
        report("Copy", k_copy, n, repeats, threads, scalar, 2.0);   // 读1 写1
        report("Scale", k_scale, n, repeats, threads, scalar, 2.0); // 读1 写1
        report("Add", k_add, n, repeats, threads, scalar, 3.0);     // 读2 写1
        report("Triad", k_triad, n, repeats, threads, scalar, 3.0); // 读2 写1
    }
    return 0;
}
