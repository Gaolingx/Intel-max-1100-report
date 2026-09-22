//===----------------------------------------------------------------------===//
// xmx_peak.cpp —— XMX（Xe Matrix eXtensions / DPAS）峰值探针
//
// 目的
// ----
// 用 SYCL 的 **joint_matrix** 扩展直接发射 DPAS（Dot Product Accumulate
// Systolic）指令，测出 Intel Data Center GPU Max 1100 上 XMX 单元的本征算力，
// 作为 torch / oneDNN GEMM 实测值（bf16 237.6、fp16 230、int8 398.9 TFLOPS）
// 的**独立对照**。
//
// 为什么不用 torch？
// ------------------
// torch 的 `matmul` 走 oneDNN，受 tiling / 数据搬运 / 线程调度影响。要判断
// “实测 GEMM 到没到硬件上限”，需要一条**绕开整个软件栈**的路径。
//
// 方法（把访存降到 0，只测 DPAS issue 上限）
// -----------------------------------------
// · 每个 work-group = 1 个 sub-group（16 lane），对应 1 个 EU 的 XMX 单元。
// · A / B 只 `joint_matrix_load` **一次**，之后全部在寄存器里跑 `joint_matrix_mad`。
// · 用 NACC 个**互相独立**的累加器 tile 交替发射，用来隐藏 DPAS 的长延迟
//   （只用 1 个累加器会形成串行依赖链，测到的是 latency 而不是 throughput）。
// · **内层循环长度 IT 必须是编译期常量并 `#pragma unroll`**——这是本探针最大的坑，
//   见下方「坑 1」。
//
// ⚠ 坑 1：运行时循环长度会让整个测量失效（本探针最初就掉进去了）
// ----------------------------------------------------------
// 如果写成 `for (i < runtime_iters) for (j < NACC) mad(acc[j], tA, tB, acc[j])`，
// 其中 tA/tB 是循环不变量，后端会把它当成**可折叠的循环不变量**：循环仍然存在，
// 但每条独立累加链只保留 1 条 DPAS 的效果，于是
//
//     实测耗时与 NACC 完全无关（NACC=1 与 NACC=16 都是 0.003522 s）
//
// 用「按 NACC 计数的 FLOP / 没有相应增长的耗时」就会虚报出 500~1000 TFLOPS
// 的荒谬峰值。**现场自检：耗时必须随 NACC 线性增长。**
// 本探针把 IT 做成编译期常量、用运行时 `outer` 循环包在外层，实测耗时随 NACC
// 线性增长，且 NACC=2/4/8 算出的 TFLOPS 互相吻合 → 计数可信。
//
// 计数口径
// --------
//   每个 DPAS: 2·M·N·K FLOP（1 次乘加 = 2 FLOP）
//   DPAS 数 = (global/sub_group_size) × outer × IT × NACC
//   int8 累加为 int32，同样按 2 OP 计，输出 GOPS。
//   归一化：每 EU 每 cycle 的 MAC = DPAS数 × M·N·K ÷ 时间 ÷ 448 ÷ 1.55e9
//
// 形状（Ponte Vecchio / Xe-HPC）
// -----------------------------
//   bf16 / fp16 : M=8, N=16, K=16   → 4096 FLOP / DPAS
//   int8        : M=8, N=16, K=32   → 8192 OP   / DPAS
//
// 编译 / 运行
// ----------
//   icpx -fsycl -O3 -o xmx_peak xmx_peak.cpp
//   ./xmx_peak all 7168              # 448 个 sub-group = 每 EU 一个
//   ./xmx_peak bf16 7168 8192 3 5 16 128
//   # 参数：<dtype|all> [global] [outer] [warmup] [repeats] [local] [it]
//
// 输出 JSON 行：`DEVICE {...}` / `CHECK {...}` / `RESULT {...}`
//
// ⚠ 关于「标称值」
// ---------------
// PVC 单卡 XMX 的标称值通常按 `448 EU × 每 EU 每 cycle FLOP × 频率` 推算，
// 代入 1550 MHz 得 bf16 ≈ 176 TFLOPS。但本机驱动**读不到真实主频**
// （`/sys/class/drm/card0/gt_act_freq_mhz = 0`，所有工具都只回声配置值 1550 MHz），
// 且实测值系统性高于按 1550 MHz 推算的标称值。因此本探针**只报实测**，
// 报告里把标称值标注为「公式值（有保留）」，不做任何缩放。
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int EU_PER_DEVICE = 448;
constexpr double NOMINAL_GHZ = 1.55;   // 驱动回报值，非实测
constexpr int SG_SIZE = 16;
constexpr int NACC = 4;                // 独立累加器个数
constexpr int DEFAULT_IT = 128;        // 内层编译期展开次数

// ---------------------------------------------------------------------------
// DPAS 峰值 kernel
//   T/Tacc : A/B 与累加器类型
//   M/N/K  : DPAS tile 形状
//   NACC   : 独立累加器个数（隐藏 DPAS 延迟，≥2 才可能打满）
//   IT     : 内层展开次数（**编译期常量**，见文件头「坑 1」）
// ---------------------------------------------------------------------------
template <typename T, typename Tacc, int M, int N, int K, int NACC_, int IT>
struct XmxKernel {
  T *A;
  T *B;
  Tacc *C;          // 布局: [subgroup][NACC][M*N]
  std::size_t lda, ldb;
  std::int64_t outer;

  void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    auto ga = sycl::multi_ptr<T, sycl::access::address_space::global_space>(A);
    auto gb = sycl::multi_ptr<T, sycl::access::address_space::global_space>(B);

    mx::joint_matrix<sycl::sub_group, T, mx::use::a, M, K, mx::layout::row_major> tA;
    mx::joint_matrix<sycl::sub_group, T, mx::use::b, K, N, mx::layout::row_major> tB;
    mx::joint_matrix<sycl::sub_group, Tacc, mx::use::accumulator, M, N> acc[NACC_];

    // A / B 只加载一次：之后循环体里没有任何访存，纯 DPAS
    mx::joint_matrix_load(sg, tA, ga, lda);
    mx::joint_matrix_load(sg, tB, gb, ldb);
#pragma unroll
    for (int j = 0; j < NACC_; ++j) {
      mx::joint_matrix_fill(sg, acc[j], Tacc(0));
    }

    // outer 是运行时值（防止外层也被整段折叠）；IT 是编译期常量，真正展开成
    // IT×NACC 条指令流。
    for (std::int64_t o = 0; o < outer; ++o) {
#pragma unroll
      for (int i = 0; i < IT; ++i) {
#pragma unroll
        for (int j = 0; j < NACC_; ++j) {
          mx::joint_matrix_mad(sg, acc[j], tA, tB, acc[j]);
        }
      }
    }

    // 把所有累加器都写回：NACC>1 时若只写 acc[0]，其余会被 DCE 掉，
    // 又会退化成「耗时与 NACC 无关」的假测量。
    const std::size_t sgid = item.get_global_id(0) / SG_SIZE;
    if (item.get_global_id(0) % SG_SIZE == 0) {
#pragma unroll
      for (int j = 0; j < NACC_; ++j) {
        auto gc = sycl::multi_ptr<Tacc, sycl::access::address_space::global_space>(
            C + (sgid * NACC_ + j) * (M * N));
        mx::joint_matrix_store(sg, acc[j], gc, N, mx::layout::row_major);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// 计时
// ---------------------------------------------------------------------------
struct RunResult {
  double best_s;
  double avg_s;
  std::size_t nsub;
};

template <typename T, typename Tacc, int M, int N, int K, int NACC_, int IT>
RunResult run_case(sycl::queue &q, std::size_t global, std::size_t local,
                   std::int64_t outer, int warmup, int repeats) {
  const std::size_t nsub = global / local;
  T *A = sycl::malloc_device<T>(M * K * 8, q);
  T *B = sycl::malloc_device<T>(K * N * 8, q);
  Tacc *C = sycl::malloc_device<Tacc>(nsub * NACC_ * M * N, q);
  q.memset(A, 0, sizeof(T) * M * K * 8).wait();
  q.memset(B, 0, sizeof(T) * K * N * 8).wait();
  q.memset(C, 0, sizeof(Tacc) * nsub * NACC_ * M * N).wait();

  auto launch = [&]() {
    q.parallel_for(
         sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
         XmxKernel<T, Tacc, M, N, K, NACC_, IT>{A, B, C, K, N, outer})
        .wait();
  };
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  std::vector<double> samples;
  for (int i = 0; i < repeats; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    launch();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  double best = samples[0], sum = 0.0;
  for (double s : samples) {
    best = s < best ? s : best;
    sum += s;
  }
  return RunResult{best, sum / static_cast<double>(samples.size()), nsub};
}

// ---------------------------------------------------------------------------
// 正确性自检：A 全 1、B 全 1 → 单次 MAD 后 C 应等于 K
// ---------------------------------------------------------------------------
template <typename T, typename Tacc, int M, int N, int K>
bool check_case(sycl::queue &q, const char *name) {
  T *A = sycl::malloc_shared<T>(M * K, q);
  T *B = sycl::malloc_shared<T>(K * N, q);
  Tacc *C = sycl::malloc_shared<Tacc>(M * N, q);
  for (int i = 0; i < M * K; ++i) A[i] = T(1);
  for (int i = 0; i < K * N; ++i) B[i] = T(1);
  for (int i = 0; i < M * N; ++i) C[i] = Tacc(0);

  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)),
                 [=](sycl::nd_item<1> item) {
                   auto sg = item.get_sub_group();
                   auto ga = sycl::multi_ptr<T, sycl::access::address_space::global_space>(A);
                   auto gb = sycl::multi_ptr<T, sycl::access::address_space::global_space>(B);
                   auto gc = sycl::multi_ptr<Tacc, sycl::access::address_space::global_space>(C);
                   mx::joint_matrix<sycl::sub_group, T, mx::use::a, M, K,
                                    mx::layout::row_major> tA;
                   mx::joint_matrix<sycl::sub_group, T, mx::use::b, K, N,
                                    mx::layout::row_major> tB;
                   mx::joint_matrix<sycl::sub_group, Tacc, mx::use::accumulator, M, N> tC;
                   mx::joint_matrix_load(sg, tA, ga, K);
                   mx::joint_matrix_load(sg, tB, gb, N);
                   mx::joint_matrix_fill(sg, tC, Tacc(0));
                   mx::joint_matrix_mad(sg, tC, tA, tB, tC);
                   if (item.get_global_id(0) == 0) {
                     mx::joint_matrix_store(sg, tC, gc, N, mx::layout::row_major);
                   }
                 })
      .wait();
  const double expect = static_cast<double>(K);
  const double got = static_cast<double>(C[0]);
  const bool ok = (got == expect);
  std::printf("CHECK {\"dtype\":\"%s\",\"expect\":%.1f,\"got\":%.1f,\"ok\":%s}\n", name,
              expect, got, ok ? "true" : "false");
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
  return ok;
}

// ---------------------------------------------------------------------------
enum class Kind { Bf16, Fp16, Int8 };
constexpr int M8 = 8, N16 = 16;

const char *kind_name(Kind k) {
  switch (k) {
  case Kind::Bf16: return "bf16";
  case Kind::Fp16: return "fp16";
  case Kind::Int8: return "int8";
  }
  return "?";
}
const char *kind_unit(Kind k) { return k == Kind::Int8 ? "GOPS" : "TFLOPS"; }
int kind_k(Kind k) { return k == Kind::Int8 ? 32 : 16; }

// IT 是编译期常量，这里只实例化几档，避免编译爆炸
template <typename T, typename Tacc, int M, int N, int K, int NACC_>
RunResult run_dispatch(sycl::queue &q, std::size_t global, std::size_t local,
                       std::int64_t outer, int warmup, int repeats, int it) {
  switch (it) {
  case 32:
    return run_case<T, Tacc, M, N, K, NACC_, 32>(q, global, local, outer, warmup, repeats);
  case 64:
    return run_case<T, Tacc, M, N, K, NACC_, 64>(q, global, local, outer, warmup, repeats);
  case 256:
    return run_case<T, Tacc, M, N, K, NACC_, 256>(q, global, local, outer, warmup, repeats);
  default:
    return run_case<T, Tacc, M, N, K, NACC_, DEFAULT_IT>(q, global, local, outer, warmup,
                                                        repeats);
  }
}

} // namespace

int main(int argc, char **argv) {
  const std::string want = argc > 1 ? argv[1] : "all";
  const std::size_t global =
      argc > 2 ? std::strtoull(argv[2], nullptr, 10) : std::size_t(EU_PER_DEVICE * SG_SIZE);
  const std::int64_t outer = argc > 3 ? std::strtoll(argv[3], nullptr, 10) : 8192;
  const int warmup = argc > 4 ? std::atoi(argv[4]) : 3;
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 5;
  std::size_t local = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : SG_SIZE;
  const int it = argc > 7 ? std::atoi(argv[7]) : DEFAULT_IT;
  if (local > global || global % local != 0) {
    local = SG_SIZE;
    while (local > 1 && global % local != 0) {
      local /= 2;
    }
  }

  std::vector<Kind> kinds;
  if (want == "all") {
    kinds = {Kind::Bf16, Kind::Fp16, Kind::Int8};
  } else if (want == "bf16") {
    kinds = {Kind::Bf16};
  } else if (want == "fp16") {
    kinds = {Kind::Fp16};
  } else if (want == "int8") {
    kinds = {Kind::Int8};
  } else {
    std::fprintf(stderr, "unknown dtype '%s'\n", want.c_str());
    return 2;
  }

  try {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("DEVICE {\"name\":\"%s\",\"driver\":\"%s\",\"max_cu\":%u,"
                "\"global_size\":%zu,\"local_size\":%zu,\"sub_groups\":%zu,"
                "\"nacc\":%d,\"it\":%d,\"outer\":%lld,\"warmup\":%d,\"repeats\":%d}\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::driver_version>().c_str(),
                dev.get_info<sycl::info::device::max_compute_units>(), global, local,
                global / local, NACC, it, static_cast<long long>(outer), warmup, repeats);

    for (Kind k : kinds) {
      const char *name = kind_name(k);
      const int K = kind_k(k);
      RunResult rr{0.0, 0.0, 0};
      bool ok = false;
      switch (k) {
      case Kind::Bf16:
        ok = check_case<bf16, float, M8, N16, 16>(q, name);
        rr = run_dispatch<bf16, float, M8, N16, 16, NACC>(q, global, local, outer, warmup,
                                                          repeats, it);
        break;
      case Kind::Fp16:
        ok = check_case<sycl::half, float, M8, N16, 16>(q, name);
        rr = run_dispatch<sycl::half, float, M8, N16, 16, NACC>(q, global, local, outer,
                                                                warmup, repeats, it);
        break;
      case Kind::Int8:
        ok = check_case<std::int8_t, std::int32_t, M8, N16, 32>(q, name);
        rr = run_dispatch<std::int8_t, std::int32_t, M8, N16, 32, NACC>(
            q, global, local, outer, warmup, repeats, it);
        break;
      }
      const double dpas = static_cast<double>(rr.nsub) * static_cast<double>(outer) *
                          static_cast<double>(it) * static_cast<double>(NACC);
      const double flops = dpas * 2.0 * M8 * N16 * K;
      const double tflops = flops / rr.best_s / 1e12;
      // 每 EU 每 cycle 的 MAC（按名义频率换算，仅作横向可比量，不是实测频率）
      const double mac_per_cycle_eu =
          dpas * static_cast<double>(M8 * N16 * K) / rr.best_s /
          static_cast<double>(EU_PER_DEVICE) / (NOMINAL_GHZ * 1e9);
      const double per_eu_gflops = tflops * 1e3 / static_cast<double>(EU_PER_DEVICE);
      std::printf("RESULT {\"dtype\":\"%s\",\"unit\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
                  "\"global_size\":%zu,\"sub_groups\":%zu,\"outer\":%lld,\"it\":%d,"
                  "\"nacc\":%d,\"dpas_count\":%.0f,\"seconds\":%.6f,\"value\":%.2f,"
                  "\"per_eu_gflops\":%.1f,\"mac_per_cycle_eu_nominal\":%.1f,"
                  "\"check_ok\":%s}\n",
                  name, kind_unit(k), M8, N16, K, global, rr.nsub,
                  static_cast<long long>(outer), it, NACC, dpas, rr.best_s, tflops,
                  per_eu_gflops, mac_per_cycle_eu, ok ? "true" : "false");
    }
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
  }
  return 0;
}
