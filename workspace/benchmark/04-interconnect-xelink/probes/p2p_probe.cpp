// 04-interconnect-xelink / probes/p2p_probe.cpp
//
// 目的：用 **Level Zero 原生 API** 回答两个问题，不依赖任何 MPI/框架：
//
//   A) P2P 到底可不可用？
//        zeDeviceCanAccessPeer / zeDeviceGetP2PProperties / PCI BDF
//   B) 如果可用，卡间真带宽是多少？（这就是 Xe Link 的实测带宽）
//        在一张卡上分配 src、另一张卡上分配 dst，用 dst 卡的 queue 发
//        zeCommandListAppendMemoryCopy —— 如果 P2P 生效，这次拷贝走 Xe Link，
//        不会经过 host。
//
// 输出（每行一条，前缀便于解析）：
//   P2PDEV   {"index":0,"name":"...","pci_bdf":"0000:xx:xx.0","vendor_id":...,"device_id":...}
//   P2PPEER  {"src":0,"dst":1,"can_access":true,"p2p_flags":[...],"reason":"can_access_peer"}
//   P2PBW    {"src":0,"dst":1,"size_bytes":16777216,"seconds":0.000123,"gibps":127.4,"gbps":136.8}
//   P2PINFO  {"p2p_pairs":2,"max_gbps":318.4,"direction_best":"0->1"}
//
// 编译： g++ -O3 -std=c++17 -o p2p_probe p2p_probe.cpp -lze_loader
//
// 注意：本机有 i915 驱动 + Level Zero 1.24.0；两块 Max 1100 各 1 tile。

#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// --------------------------------------------------------------------------- //
// 小工具
// --------------------------------------------------------------------------- //
static const char *ze_rc(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return "SUCCESS";
        case ZE_RESULT_NOT_READY: return "NOT_READY";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "UNINITIALIZED";
        default: return "OTHER";
    }
}

static std::string escape(const char *s) {
    std::string out;
    for (const char *p = s; p && *p; ++p) {
        if (*p == '"' || *p == '\\') out += '\\';
        out += *p;
    }
    return out;
}

static double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// --------------------------------------------------------------------------- //
struct DeviceInfo {
    ze_device_handle_t h = nullptr;
    std::string name;
    std::string bdf;
    uint32_t vendor_id = 0, device_id = 0;
    bool bdf_ok = false;
};

static std::vector<std::string> p2p_flag_names(ze_device_p2p_properties_t p) {
    std::vector<std::string> v;
    // 本机 Level Zero 1.24 只有这两个 flag（没有 P2P_COPY / PEER_ATOMICS）
    if (p.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS)   v.push_back("ACCESS");
    if (p.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS)  v.push_back("ATOMICS");
    return v;
}

int main(int argc, char **argv) {
    // 是否跑带宽测试（默认跑）；第 1 个参数给 "no-bw" 可跳过
    bool do_bw = !(argc > 1 && std::strcmp(argv[1], "no-bw") == 0);
    // P2P 拷贝的最大消息大小（字节），默认 64 MiB
    std::size_t max_bytes = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : (64ull << 20);

    ze_result_t rc = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (rc != ZE_RESULT_SUCCESS) {
        std::printf("P2PERR {\"stage\":\"zeInit\",\"rc\":\"%s\"}\n", ze_rc(rc));
        return 1;
    }

    uint32_t driver_count = 0;
    zeDriverGet(&driver_count, nullptr);
    if (driver_count == 0) {
        std::printf("P2PERR {\"stage\":\"zeDriverGet\",\"note\":\"no driver\"}\n");
        return 1;
    }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    zeDriverGet(&driver_count, drivers.data());

    // 收集所有 driver 的所有 device（真实机器上通常只有一个 driver 管两个 GPU）
    std::vector<DeviceInfo> devs;
    for (uint32_t d = 0; d < driver_count; ++d) {
        uint32_t n = 0;
        if (zeDeviceGet(drivers[d], &n, nullptr) != ZE_RESULT_SUCCESS) continue;
        std::vector<ze_device_handle_t> hs(n);
        zeDeviceGet(drivers[d], &n, hs.data());
        for (auto h : hs) {
            ze_device_properties_t p{};
            p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (zeDeviceGetProperties(h, &p) != ZE_RESULT_SUCCESS) continue;
            // 只要 GPU（排除没有 EU、纯 copy engine 的 device）
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;

            DeviceInfo di;
            di.h = h;
            di.name = p.name;
            di.vendor_id = p.vendorId;
            di.device_id = p.deviceId;

            ze_pci_ext_properties_t pci{};
            pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            if (zeDevicePciGetPropertiesExt(h, &pci) == ZE_RESULT_SUCCESS) {
                char buf[64];
                std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x",
                              pci.address.domain, pci.address.bus,
                              pci.address.device, pci.address.function);
                di.bdf = buf;
                di.bdf_ok = true;
            } else {
                di.bdf = "unavailable";
            }
            devs.push_back(di);
        }
    }

    std::printf("P2PINFO {\"driver_count\":%u,\"gpu_device_count\":%zu}\n",
                driver_count, devs.size());
    for (std::size_t i = 0; i < devs.size(); ++i) {
        std::printf("P2PDEV {\"index\":%zu,\"name\":\"%s\",\"pci_bdf\":\"%s\","
                    "\"vendor_id\":%u,\"device_id\":%u}\n",
                    i, escape(devs[i].name.c_str()).c_str(), devs[i].bdf.c_str(),
                    devs[i].vendor_id, devs[i].device_id);
    }
    if (devs.size() < 2) {
        std::printf("P2PINFO {\"note\":\"只有 %zu 个 GPU device，无法测卡间互连\"}\n", devs.size());
        return 0;
    }

    // ---- A) 能力查询：所有有序对 ---------------------------------------- //
    int accessible_pairs = 0, ordered_pairs = 0;
    for (std::size_t i = 0; i < devs.size(); ++i) {
        for (std::size_t j = 0; j < devs.size(); ++j) {
            if (i == j) continue;
            ++ordered_pairs;
            ze_bool_t can = 0;
            ze_result_t r1 = zeDeviceCanAccessPeer(devs[i].h, devs[j].h, &can);
            ze_device_p2p_properties_t prop{};
            prop.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
            ze_result_t r2 = zeDeviceGetP2PProperties(devs[i].h, devs[j].h, &prop);
            auto fl = p2p_flag_names(prop);
            std::string flags = "[";
            for (std::size_t k = 0; k < fl.size(); ++k)
                flags += (k ? ",\"" : "\"") + fl[k] + "\"";
            flags += "]";
            if (can) ++accessible_pairs;
            std::printf("P2PPEER {\"src\":%zu,\"dst\":%zu,\"can_access\":%s,"
                        "\"p2p_flags\":%s,\"can_access_rc\":\"%s\",\"props_rc\":\"%s\"}\n",
                        i, j, (r1 == ZE_RESULT_SUCCESS && can) ? "true" : "false",
                        flags.c_str(), ze_rc(r1), ze_rc(r2));
        }
    }
    std::printf("P2PINFO {\"ordered_pairs\":%d,\"accessible_pairs\":%d,"
                "\"p2p_available\":%s}\n",
                ordered_pairs, accessible_pairs,
                accessible_pairs > 0 ? "true" : "false");

    if (!do_bw || accessible_pairs == 0) {
        if (accessible_pairs == 0)
            std::printf("P2PINFO {\"note\":\"P2P 不可用，跳过带宽测试——多卡会退化到 "
                        "PCIe/host 中转\"}\n");
        return 0;
    }

    // ---- B) P2P 带宽：单 context 含多 device，跨卡 memcpy ----------------- //
    // 同一个 context 里两个 device 的内存互相可见（前提是 can_access_peer），
    // 因此不需要 IPC handle。
    ze_context_desc_t cdesc{};
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t ctx = nullptr;
    rc = zeContextCreate(drivers[0], &cdesc, &ctx);
    if (rc != ZE_RESULT_SUCCESS) {
        std::printf("P2PERR {\"stage\":\"zeContextCreate\",\"rc\":\"%s\"}\n", ze_rc(rc));
        return 1;
    }

    const std::size_t sizes[] = {
        4u << 10, 64u << 10, 1u << 20, 4u << 20, 16u << 20, 64u << 20, 256u << 20
    };

    double best_gbps = 0.0;
    std::string best_dir;

    for (std::size_t si = 0; si < devs.size(); ++si) {
        for (std::size_t di = 0; di < devs.size(); ++di) {
            if (si == di) continue;
            ze_bool_t can = 0;
            if (zeDeviceCanAccessPeer(devs[si].h, devs[di].h, &can) != ZE_RESULT_SUCCESS || !can)
                continue;

            // 队列/命令表：分配在 dst 卡上（拷贝由 dst 卡发起）
            ze_command_queue_desc_t qd{};
            qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
            qd.ordinal = 0;
            qd.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
            qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
            ze_command_queue_handle_t q = nullptr;
            rc = zeCommandQueueCreate(ctx, devs[di].h, &qd, &q);
            if (rc != ZE_RESULT_SUCCESS) {
                std::printf("P2PERR {\"stage\":\"zeCommandQueueCreate\",\"src\":%zu,"
                            "\"dst\":%zu,\"rc\":\"%s\"}\n", si, di, ze_rc(rc));
                continue;
            }
            ze_command_list_desc_t ld{};
            ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
            ld.commandQueueGroupOrdinal = 0;
            ze_command_list_handle_t lst = nullptr;
            zeCommandListCreate(ctx, devs[di].h, &ld, &lst);

            for (std::size_t sz : sizes) {
                if (sz > max_bytes) continue;

                ze_device_mem_alloc_desc_t dma{};
                dma.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
                void *src = nullptr, *dst = nullptr;
                if (zeMemAllocDevice(ctx, &dma, sz, 4096, devs[si].h, &src) != ZE_RESULT_SUCCESS)
                    continue;
                if (zeMemAllocDevice(ctx, &dma, sz, 4096, devs[di].h, &dst) != ZE_RESULT_SUCCESS) {
                    zeMemFree(ctx, src);
                    continue;
                }
                // 初始化（在各自卡上 memset），避免未映射页
                const unsigned char zero = 0;
                zeCommandListReset(lst);
                zeCommandListAppendMemoryFill(lst, src, &zero, sizeof(zero), sz,
                                              nullptr, 0, nullptr);
                zeCommandListAppendMemoryFill(lst, dst, &zero, sizeof(zero), sz,
                                              nullptr, 0, nullptr);
                zeCommandListClose(lst);
                zeCommandQueueExecuteCommandLists(q, 1, &lst, nullptr);
                zeCommandQueueSynchronize(q, UINT64_MAX);

                double best = 1e30;
                for (int rep = 0; rep < 12; ++rep) {
                    zeCommandListReset(lst);
                    zeCommandListAppendMemoryCopy(lst, dst, src, sz, nullptr, 0, nullptr);
                    zeCommandListClose(lst);
                    double t0 = now_s();
                    zeCommandQueueExecuteCommandLists(q, 1, &lst, nullptr);
                    zeCommandQueueSynchronize(q, UINT64_MAX);
                    double t = now_s() - t0;
                    if (rep >= 2 && t < best) best = t;   // 丢掉前两次（冷启动）
                }
                if (best < 1e29) {
                    double gbps = double(sz) / best / 1e9;      // 10^9 = GB/s
                    double gibps = double(sz) / best / (1ull << 30);
                    std::printf("P2PBW {\"src\":%zu,\"dst\":%zu,\"size_bytes\":%zu,"
                                "\"seconds\":%.9f,\"gbps\":%.2f,\"gibps\":%.2f}\n",
                                si, di, sz, best, gbps, gibps);
                    std::fflush(stdout);
                    if (gbps > best_gbps) {
                        best_gbps = gbps;
                        char b[64];
                        std::snprintf(b, sizeof b, "%zu->%zu", si, di);
                        best_dir = b;
                    }
                }
                zeMemFree(ctx, src);
                zeMemFree(ctx, dst);
            }
            zeCommandListDestroy(lst);
            zeCommandQueueDestroy(q);
        }
    }

    std::printf("P2PINFO {\"max_gbps\":%.2f,\"best_direction\":\"%s\","
                "\"xe_link_spec_gbps\":318.0,\"pct_of_spec\":%.1f}\n",
                best_gbps, best_dir.c_str(),
                best_gbps > 0 ? 100.0 * best_gbps / 318.0 : 0.0);
    zeContextDestroy(ctx);
    return 0;
}
