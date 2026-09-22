//===----------------------------------------------------------------------===//
// bw_probe.cpp —— 自定义 SYCL 显存带宽探针（BabelStream 的补充）
//
// 为什么还要自己写一个？
// ---------------------
// BabelStream 是事实标准，但它的 5 个 kernel（copy/mul/add/triad/dot）**全是
// 读写混合**，且固定 4 字节向量访问。`docs/TODO/03-memory-bandwidth.md` 要求覆盖
// **访问模式（向量宽度 / 跨步）对带宽的影响**，以及**读、写各自的单向上限**。
// 本探针只做这三件 BabelStream 不做的事：
//   1. 单向 read   —— 只读，逼近 HBM 读上限
//   2. 单向 write  —— 只写，逼近 HBM 写上限（写通常比读慢，因为要 ECC/写合并）
//   3. vec/stride 扫描 —— 每线程访问宽度 4B..64B、跨步 1..16 个向量
//
// 计数口径（与 BabelStream 一致，便于横向比较）
// --------------------------------------------
//   moved_bytes = count × VEC × 4B × io_factor
//     copy: 2 (1 读 + 1 写)   triad: 3 (2 读 + 1 写)
//     read: 1                 write: 1
//   gbps = moved_bytes / t / 1e9
//
// ⚠ stride > 1 时「有效带宽」按**逻辑访问到的字节**计，数值下降反映的是
//   缓存行利用率不足（每个 64B 行只用了一小部分），**不是 HBM 变慢**。报告里会标注。
// ⚠ 数组小于 192 MB（L2 容量）时会命中 L2，得到远高于 HBM 的数字。由调用方
//   （run_bench.py）负责挑尺寸，并在报告里区分“缓存内”与“HBM”。
//
// 编译 / 运行
// ----------
//   source /opt/intel/oneapi/setvars.sh     # 注意：返回 rc=3，别用 && 串联
//   icpx -fsycl -O3 -o bw_probe bw_probe.cpp
//   ZE_AFFINITY_MASK=0 ./bw_probe --mode read --vec 8 --bytes $((1<<30))
//
// 输出：每个配置一行 `BWRESULT {json}`，由 run_bench.py 解析。
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Opts {
    std::string mode = "copy";   // copy | read | write | triad
    std::string sweep = "vec";   // vec | stride | none（none 时用 --vec/--stride 的单值）
    std::size_t bytes = 1ull << 30;   // 每个数组的字节数
    int vec = 4;
    int stride = 1;
    int local = 256;
    int mult = 448;              // global_size = local × mult
    int iters = 20;
    int warmup = 3;
    int device = 0;
};

// ---------------------------------------------------------------------------
// 访问计数：一次访问搬了多少逻辑字节的倍数
// ---------------------------------------------------------------------------
int io_factor(const std::string &m) {
    if (m == "triad") return 3;
    if (m == "read" || m == "write") return 1;
    return 2;  // copy
}

// ---------------------------------------------------------------------------
// 四个 kernel。模板参数 VEC 决定每线程每步访问的 float 车道数。
// 用 sycl::vec 让编译器发出真正的向量 load/store（否则会被拆成标量）。
// ---------------------------------------------------------------------------
template <int VEC, int MODE>
struct BwKernel {
    using V = sycl::vec<float, VEC>;
    const float *__restrict__ a;
    const float *__restrict__ b;
    float *__restrict__ c;
    float *__restrict__ sink;
    std::size_t count;   // 每个 work-item 需要处理的 vec 元素总数
    std::size_t stride;  // 跨步（单位：vec）
    std::size_t g;       // global size

    void operator()(sycl::id<1> id) const {
        const V *__restrict__ av = reinterpret_cast<const V *>(a);
        const V *__restrict__ bv = reinterpret_cast<const V *>(b);
        V *__restrict__ cv = reinterpret_cast<V *>(c);

        if constexpr (MODE == 0) {          // ---- copy：1 读 + 1 写 ----
            for (std::size_t i = id[0]; i < count; i += g)
                cv[i * stride] = av[i * stride];

        } else if constexpr (MODE == 1) {   // ---- read：只读，规约 ----
            V acc{};
            for (std::size_t i = id[0]; i < count; i += g)
                acc = acc + av[i * stride];
            float s = 0.f;
            for (int k = 0; k < VEC; ++k) s += acc[k];
            // 条件恒假：编译器无法删除规约链，实际又不会产生写流量
            if (s == -1.2345e30f) sink[0] = s;

        } else if constexpr (MODE == 2) {   // ---- write：只写 ----
            V val{};
            for (int k = 0; k < VEC; ++k) val[k] = 1.25f + static_cast<float>(k);
            for (std::size_t i = id[0]; i < count; i += g)
                cv[i * stride] = val;

        } else {                            // ---- triad：2 读 + 1 写 ----
            for (std::size_t i = id[0]; i < count; i += g)
                cv[i * stride] = av[i * stride] + 1.5f * bv[i * stride];
        }
    }
};

// 把「VEC 在编译期、MODE 也在编译期」的组合摊平成 switch
#define BW_DISPATCH(V, M)                                                       \
    case V:                                                                     \
        ev = q.parallel_for(sycl::range<1>{g},                                  \
                            BwKernel<V, M>{a, b, c, sink, count, stride, g});   \
        break;

template <int MODE>
static sycl::event launch(sycl::queue &q, int vec, const float *a, const float *b,
                          float *c, float *sink, std::size_t count,
                          std::size_t stride, std::size_t g) {
    sycl::event ev;
    switch (vec) {
        BW_DISPATCH(1, MODE)
        BW_DISPATCH(2, MODE)
        BW_DISPATCH(4, MODE)
        BW_DISPATCH(8, MODE)
        BW_DISPATCH(16, MODE)
        default:
            std::fprintf(stderr, "bw_probe: unsupported --vec %d\n", vec);
            std::exit(2);
    }
    return ev;
}
#undef BW_DISPATCH

}  // namespace

int main(int argc, char **argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--mode") o.mode = val();
        else if (k == "--sweep") o.sweep = val();
        else if (k == "--bytes") o.bytes = std::strtoull(val(), nullptr, 0);
        else if (k == "--vec") o.vec = std::atoi(val());
        else if (k == "--stride") o.stride = std::atoi(val());
        else if (k == "--local") o.local = std::atoi(val());
        else if (k == "--mult") o.mult = std::atoi(val());
        else if (k == "--iters") o.iters = std::atoi(val());
        else if (k == "--warmup") o.warmup = std::atoi(val());
        else if (k == "--device") o.device = std::atoi(val());
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }

    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) { std::fprintf(stderr, "no GPU found\n"); return 3; }
    if (o.device >= static_cast<int>(gpus.size())) o.device = 0;
    sycl::queue q{gpus[o.device], sycl::property::queue::enable_profiling{}};
    std::fprintf(stderr, "[bw_probe] device=%s mode=%s bytes=%zu\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str(),
                 o.mode.c_str(), o.bytes);

    const std::size_t nfloat = o.bytes / sizeof(float);
    float *a = sycl::malloc_device<float>(nfloat, q);
    float *b = sycl::malloc_device<float>(nfloat, q);
    float *c = sycl::malloc_device<float>(nfloat, q);
    float *sink = sycl::malloc_device<float>(16, q);
    if (!a || !b || !c || !sink) { std::fprintf(stderr, "device alloc failed\n"); return 3; }
    // 填非零：避免全零数据被 memset-识别或触发压缩
    q.memset(a, 0x3f, o.bytes);
    q.memset(b, 0x3e, o.bytes);
    q.memset(c, 0x00, o.bytes);
    q.wait();

    std::vector<int> vecs{o.vec}, strides{o.stride};
    if (o.sweep == "vec") vecs = {1, 2, 4, 8, 16};
    else if (o.sweep == "stride") strides = {1, 2, 4, 8, 16};
    else if (o.sweep == "vecstride") { vecs = {1, 2, 4, 8, 16}; strides = {1, 2, 4, 8, 16}; }

    const int io = io_factor(o.mode);

    for (int stride : strides) {
        for (int vec : vecs) {
            const std::size_t nvec = nfloat / static_cast<std::size_t>(vec);
            const std::size_t count = nvec / static_cast<std::size_t>(stride);
            if (count == 0) continue;
            const std::size_t g = static_cast<std::size_t>(o.local) * o.mult;
            const std::size_t accessed = count * static_cast<std::size_t>(vec) * sizeof(float);

            auto once = [&]() -> sycl::event {
                if (o.mode == "read")  return launch<1>(q, vec, a, b, c, sink, count, stride, g);
                if (o.mode == "write") return launch<2>(q, vec, a, b, c, sink, count, stride, g);
                if (o.mode == "triad") return launch<3>(q, vec, a, b, c, sink, count, stride, g);
                return launch<0>(q, vec, a, b, c, sink, count, stride, g);
            };

            for (int w = 0; w < o.warmup; ++w) once().wait_and_throw();

            double best_ms = 1e300;
            for (int t = 0; t < o.iters; ++t) {
                sycl::event ev = once();
                ev.wait_and_throw();
                const double ms = static_cast<double>(
                    ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                    ev.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6;
                if (ms < best_ms) best_ms = ms;
            }

            const double gbps = static_cast<double>(accessed) * io / (best_ms / 1000.0) / 1e9;
            std::printf(
                "BWRESULT {\"mode\":\"%s\",\"vec\":%d,\"local\":%d,\"stride\":%d,"
                "\"array_bytes\":%zu,\"accessed_bytes\":%zu,\"moved_bytes\":%zu,"
                "\"ms\":%.5f,\"gbps\":%.2f}\n",
                o.mode.c_str(), vec, o.local, stride, o.bytes, accessed,
                accessed * static_cast<std::size_t>(io), best_ms, gbps);
            std::fflush(stdout);
        }
    }

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    sycl::free(sink, q);
    return 0;
}
