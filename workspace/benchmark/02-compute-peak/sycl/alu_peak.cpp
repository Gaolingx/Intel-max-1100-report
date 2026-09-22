//===----------------------------------------------------------------------===//
// alu_peak.cpp —— 纯 ALU（vector engine）FMA 峰值探针
//
// 目的
// ----
// 用**手写 SYCL kernel** 把 Intel Data Center GPU Max 1100（Ponte Vecchio）
// 的 vector/ALU 引擎压到极限，得到 FP32 / FP64 / FP16 / INT32 的**理论峰值实测值**。
//
// 为什么不只用 torch？
// --------------------
// torch 的 GEMM 走 oneDNN，对小尺寸/特殊 dtype 可能走不饱和的路径；而且
// FP64 的实测（17.37 TFLOPS = FP32 的 0.78×）与「FP64 = FP32/2」的教科书假设冲突，
// 必须用**纯寄存器驻留的 FMA 循环**（无任何访存、无 GEMM tiling 开销）来裁定。
//
// 方法
// ----
// · 每个 work-item 持有 ACC 条**独立**的 FMA 依赖链，每条链是 VEC 宽向量，
//   即 32 个在飞 FMA，足以隐藏 FMA 延迟（PVC FMA latency ~7 cycle / II=1）。
// · kernel 里**没有任何访存**，只有常量初始化 + FMA 循环 + 一个永不成立的条件写回，
//   因此测到的是 ALU 的 issue 上限，而不是内存带宽。
// · global size 默认 448 EU × 16 lane = 7168，正好铺满一个 tile 的所有 SIMD lane。
// · 设备侧用 SYCL event 计时，warmup 若干次后取多次重复的**最小值**。
//
// 计数口径
// --------
//   FLOPs = global_size × VEC × ACC × iters × 2      (1 次 FMA = 2 FLOP)
//   INT32 同样按 2 OP（1 mul + 1 add）计，输出 GOPS。
//
// 编译 / 运行
// ----------
//   icpx -fsycl -O3 -ffp-contract=fast -o alu_peak alu_peak.cpp
//   ./alu_peak all 7168
//   ./alu_peak fp32 7168 33554432 5
//
// 注意
// ----
// · **不要**开 `-ffast-math` 之外的重排优化，否则 FMA 可能被常量折叠掉；
//   本文件用 runtime `guard` 参数阻止死代码消除。
// · fp16 在 PVC 的 vector 引擎上是**原生半精度**还是**转成 fp32 再算**，由本测试裁定。
// · bf16 的 vector 路径在 PVC 上无原生支持（会走转换），因此本探针不测 bf16，
//   其吞吐见 `run_bench.py --suite torch` 的 vector 结果。
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

#ifndef ALU_VEC
#define ALU_VEC 8
#endif
#ifndef ALU_ACC
#define ALU_ACC 8
#endif
#ifndef ALU_UNROLL
#define ALU_UNROLL 4
#endif

constexpr int VEC = ALU_VEC;      // 向量宽度（sycl::vec<T, VEC>，1 个 GRF）
constexpr int ACC = ALU_ACC;      // 独立 FMA 依赖链条数
constexpr int UNROLL = ALU_UNROLL; // 每条链在一个内层迭代里连续做几次 FMA
constexpr int LANES_PER_EU = 16;
constexpr int EU_PER_DEVICE = 448;

// ---------------------------------------------------------------------------
// 真·FMA 循环 kernel
//
// 反优化要点（非常关键）：
//   `acc = mul*acc + c` 是**仿射递推**，常量系数下 LLVM 会把它折成闭式
//   （或做强度削减），实测会得到 >2× 理论值的假峰值。因此这里改用
//   **非线性递推** `acc = acc*acc + c`（logistic-like，无闭式解），
//   编译器无法消解，必须老老实实每条都发射 FMA。
//
// · ACC 条互不相干的依赖链 → 藏 FMA latency（PVC FMA latency ≈ 4~7 cycle）
// · VEC 宽向量 → 1 次指令做 VEC 个 lane，减少 issue 压力
// · UNROLL → 摊薄循环计数/分支开销
// ---------------------------------------------------------------------------
template <typename T>
struct AluKernel {
  sycl::vec<T, VEC> *out;
  std::size_t guard;
  std::int64_t iters;
  T seed;

  void operator()(sycl::nd_item<1> item) const {
    const T s = seed;
    sycl::vec<T, VEC> acc[ACC];
    const sycl::vec<T, VEC> one{static_cast<T>(1)};
#pragma unroll
    for (int j = 0; j < ACC; ++j) {
      // 初值取 (0,1) 内的不同小数，混沌映射不发散
      acc[j] = sycl::vec<T, VEC>{
          static_cast<T>(s / static_cast<T>(4 * ACC + j + 2))};
    }

    for (std::int64_t i = 0; i < iters; ++i) {
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
        for (int j = 0; j < ACC; ++j) {
          // x <- 1 - x*x：logistic 型的混沌映射，**1 次 FMA**、输出恒有界，
          // 无闭式解 → 编译器必须真实发射每一条 FMA。
          // float 下 contract 成 `fma(-x, x, 1)`；int 下是 1 mul + 1 sub。
          acc[j] = one - acc[j] * acc[j];
        }
      }
    }

    sycl::vec<T, VEC> r{static_cast<T>(0)};
#pragma unroll
    for (int j = 0; j < ACC; ++j) {
      r = r + acc[j];
    }
    // guard 是 runtime 参数（默认 SIZE_MAX），编译器无法证明条件恒假，
    // 因此上面的 FMA 循环不会被 DCE 掉；实际运行时永远不会写内存。
    if (item.get_global_id(0) == guard) {
      out[0] = r;
    }
  }
};

// ---------------------------------------------------------------------------
// 运行一次（warmup + repeats 次计时，取最小值）
// ---------------------------------------------------------------------------
struct RunResult {
  double best_s = 0.0;
  double avg_s = 0.0;
};

template <typename T>
RunResult run_case(sycl::queue &q, std::size_t global, std::size_t local,
                   std::int64_t iters, int warmup, int repeats) {
  T *dev_out = sycl::malloc_device<T>(VEC, q);
  q.memset(dev_out, 0, sizeof(T) * VEC).wait();

  const std::size_t guard = std::numeric_limits<std::size_t>::max();
  auto launch = [&]() {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   AluKernel<T>{reinterpret_cast<sycl::vec<T, VEC> *>(dev_out),
                                guard, iters, T(1)})
        .wait();
  };

  for (int i = 0; i < warmup; ++i) {
    launch();
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    launch();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  sycl::free(dev_out, q);

  double best = samples[0], sum = 0.0;
  for (double s : samples) {
    best = s < best ? s : best;
    sum += s;
  }
  return RunResult{best, sum / static_cast<double>(samples.size())};
}

// ---------------------------------------------------------------------------
// dtype 表
// ---------------------------------------------------------------------------
enum class Kind { Fp32, Fp64, Fp16, Int32 };

const char *kind_name(Kind k) {
  switch (k) {
  case Kind::Fp32: return "fp32";
  case Kind::Fp64: return "fp64";
  case Kind::Fp16: return "fp16";
  case Kind::Int32: return "int32";
  }
  return "?";
}

const char *kind_unit(Kind k) { return k == Kind::Int32 ? "GOPS" : "GFLOPS"; }

std::int64_t default_iters(Kind k) {
  // 目标 ~0.2-0.5 s/次：FLOPs/iter = global(7168) × VEC(8) × ACC(4) × 2
  switch (k) {
  case Kind::Fp32: return 1 << 22;   // 22.2 TFLOPS → ~0.35 s
  case Kind::Fp64: return 1 << 22;   // ~11-17 TFLOPS → ~0.7 s
  case Kind::Fp16: return 1 << 22;
  case Kind::Int32: return 1 << 20;  // int mul 慢很多
  }
  return 1 << 22;
}

} // namespace

int main(int argc, char **argv) {
  std::string want = argc > 1 ? argv[1] : "all";
  std::size_t global = argc > 2 ? std::strtoull(argv[2], nullptr, 10)
                                : std::size_t(EU_PER_DEVICE * LANES_PER_EU);
  std::int64_t iters = argc > 3 ? std::strtoll(argv[3], nullptr, 10) : -1;
  int warmup = argc > 4 ? std::atoi(argv[4]) : 3;
  int repeats = argc > 5 ? std::atoi(argv[5]) : 5;
  std::size_t local = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 128;
  if (local > global || global % local != 0) {
    local = 128;
    while (global % local != 0 && local > 1) {
      local /= 2;
    }
  }

  std::vector<Kind> kinds;
  if (want == "all") {
    kinds = {Kind::Fp32, Kind::Fp64, Kind::Fp16, Kind::Int32};
  } else if (want == "fp32") {
    kinds = {Kind::Fp32};
  } else if (want == "fp64") {
    kinds = {Kind::Fp64};
  } else if (want == "fp16") {
    kinds = {Kind::Fp16};
  } else if (want == "int32") {
    kinds = {Kind::Int32};
  } else {
    std::fprintf(stderr, "unknown dtype '%s'\n", want.c_str());
    return 2;
  }

  try {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();

    std::printf("DEVICE {\"name\":\"%s\",\"driver\":\"%s\",\"max_cu\":%u,"
                "\"global_size\":%zu,\"local_size\":%zu,\"vec\":%d,\"acc\":%d,"
                "\"unroll\":%d,\"warmup\":%d,\"repeats\":%d}\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::driver_version>().c_str(),
                dev.get_info<sycl::info::device::max_compute_units>(),
                global, local, VEC, ACC, UNROLL, warmup, repeats);

    for (Kind k : kinds) {
      const std::int64_t it = iters > 0 ? iters : default_iters(k);
      double secs = 0.0;
      switch (k) {
      case Kind::Fp32:
        secs = run_case<float>(q, global, local, it, warmup, repeats).best_s;
        break;
      case Kind::Fp64:
        secs = run_case<double>(q, global, local, it, warmup, repeats).best_s;
        break;
      case Kind::Fp16:
        secs = run_case<sycl::half>(q, global, local, it, warmup, repeats).best_s;
        break;
      case Kind::Int32:
        secs = run_case<std::int32_t>(q, global, local, it, warmup, repeats).best_s;
        break;
      }

      const double flops = static_cast<double>(global) * VEC * ACC * UNROLL *
                           static_cast<double>(it) * 2.0;
      const double gflops = flops / secs / 1e9;
      std::printf("RESULT {\"dtype\":\"%s\",\"unit\":\"%s\",\"global_size\":%zu,"
                  "\"iters\":%lld,\"seconds\":%.6f,\"value\":%.2f}\n",
                  kind_name(k), kind_unit(k), global,
                  static_cast<long long>(it), secs, gflops);
      std::fflush(stdout);
    }
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
  }
  return 0;
}
