// alu_clock_probe.cpp —— 用 device-scope clock() 直接测量 GPU 真实运行频率。
//
// 背景：alu_peak.cpp 实测 fp32 约 37~41 TFLOPS，而“教科书公式”
//   448 CU × 16 lane × 2 FLOP × 1.55 GHz = 22.2 TFLOPS
// 只能解释其中 60%。xpu-smi / clinfo 都报 1550 MHz，但那可能是“配置值”而非
// “实际值”。本探针在 kernel 内部读取 cl_khr_kernel_clock 的 device 计数器，
// 用 (cycles / 墙钟秒数) 得到真实核心频率，从而判定：
//   - 若真实频率 ≈ 1550 MHz  -> ALU 每 lane 每周期能发 >1 条 FMA（双发射）
//   - 若真实频率 ≈ 2600+ MHz -> 公式中的频率项被低估（ES 硅片跑得更高）
//
// 用一个 work-group（1024 个 work-item）跑很长的循环，使 clock 差值几乎覆盖
// 整个 kernel 生命周期，墙钟时间 ≈ 循环时间。
//
// 编译：
//   source /opt/intel/oneapi/setvars.sh
//   icpx -fsycl -O3 -ffp-contract=fast -o /tmp/alu_clock_probe alu_clock_probe.cpp
// 运行：
//   ZE_AFFINITY_MASK=0 /tmp/alu_clock_probe [iters] [vec] [acc]

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace exp_ext = sycl::ext::oneapi::experimental;

template <int VEC, int ACC>
static void run(sycl::queue &q, std::int64_t iters) {
  std::uint64_t *buf = sycl::malloc_shared<std::uint64_t>(4, q);
  buf[0] = buf[1] = buf[2] = buf[3] = 0;

  // 单个 work-group：nd_range{1024,1024} -> 1 个 group，1024 个 work-item
  auto ev = q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>{1024, 1024}, [=](sycl::nd_item<1> item) {
      using namespace sycl::ext::oneapi::experimental;
      const std::size_t gid = item.get_global_id(0);

      sycl::vec<float, VEC> acc[ACC];
#pragma unroll
      for (int j = 0; j < ACC; ++j)
        acc[j] = sycl::vec<float, VEC>{static_cast<float>(1.0f / (4 * ACC + j + 2))};

      const sycl::vec<float, VEC> one{1.0f};

      std::uint64_t c0 = 0, c1 = 0;
      if (gid == 0) {
        c0 = clock<clock_scope::device>(); // 预热计数器
        c0 = clock<clock_scope::device>();
      }
      for (std::int64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < ACC; ++j)
          acc[j] = one - acc[j] * acc[j]; // 非可折叠 logistic 映射：1 FMA
      }
      if (gid == 0) {
        c1 = clock<clock_scope::device>();
        buf[0] = c0;
        buf[1] = c1;
      }

      // 防止整个循环被 DCE 掉
      sycl::vec<float, VEC> r{0.0f};
#pragma unroll
      for (int j = 0; j < ACC; ++j)
        r = r + acc[j];
      if (item.get_local_id(0) == 1023 && r[0] == 1234.5678f)
        buf[3] = 1;
    });
  });

  const auto t0 = std::chrono::steady_clock::now();
  ev.wait();
  const auto t1 = std::chrono::steady_clock::now();

  const double wall = std::chrono::duration<double>(t1 - t0).count();
  const std::uint64_t cycles = buf[1] - buf[0];
  const double freq_ghz = static_cast<double>(cycles) / wall / 1e9;
  // 单 work-group（1024 work-item）每周期完成的 lane-FMA 数
  const double lane_fma = 1024.0 * VEC * ACC * static_cast<double>(iters);
  const double fma_per_cycle = lane_fma / static_cast<double>(cycles);

  std::printf(
      "VEC=%d ACC=%d iters=%lld  wall=%.4f s  cycles=%llu  "
      "=> 实测核心频率 = %.1f MHz (%.3f GHz)\n",
      VEC, ACC, static_cast<long long>(iters), wall,
      static_cast<unsigned long long>(cycles), freq_ghz * 1000.0, freq_ghz);
  std::printf("    lane-FMA=%.3e  ->  %.3f lane-FMA/cycle "
              "(1024 work-item 占 7168 lane 的 %.1f%%)\n",
              lane_fma, fma_per_cycle, 1024.0 / 7168.0 * 100.0);

  sycl::free(buf, q);
}

int main(int argc, char **argv) {
  std::int64_t iters = (argc > 1) ? std::atoll(argv[1]) : 5000000;
  const int vec = (argc > 2) ? std::atoi(argv[2]) : 8;
  const int acc = (argc > 3) ? std::atoi(argv[3]) : 8;

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  auto dev = q.get_device();
  std::printf("device            : %s\n",
              dev.get_info<sycl::info::device::name>().c_str());
  std::printf("max_compute_units : %u\n",
              dev.get_info<sycl::info::device::max_compute_units>());
  std::printf("clock_frequency   : %u MHz   (SYCL 报告的标称值)\n",
              dev.get_info<sycl::info::device::max_clock_frequency>());
  std::printf("has ext_oneapi_clock_device: %d\n",
              static_cast<int>(dev.has(sycl::aspect::ext_oneapi_clock_device)));
  std::printf("\n");

  if (vec == 8 && acc == 8)
    run<8, 8>(q, iters);
  else if (vec == 8 && acc == 4)
    run<8, 4>(q, iters);
  else if (vec == 16 && acc == 8)
    run<16, 8>(q, iters);
  else if (vec == 4 && acc == 8)
    run<4, 8>(q, iters);
  else
    run<8, 8>(q, iters);
  return 0;
}
