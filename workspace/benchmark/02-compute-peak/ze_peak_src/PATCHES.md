# `ze_peak_src/` 相对上游的改动

上游：`https://github.com/oneapi-src/level-zero-tests` @ `master`，
路径 `perf_tests/ze_peak/**` + `perf_tests/common/**`。

**源码只改了 1 个文件 1 行**；另外新增了 2 个本地辅助物（不是改上游）。
所有改动都是为了让它在**本机（Ubuntu 25.04 / i915 / L0 1.24.0 / GCC 15）**上能编过。

---

## 改动 1（必需）：`ze_peak/src/ze_peak.cpp:15` —— 消除重复定义

```diff
 #include <algorithm>
 
-bool verbose = false;
+// [local patch 2026-09-22] was: bool verbose = false;
+// Upstream regression: common/src/ze_app.cpp:17 now also defines the same
+// global, so linking both TUs fails with "multiple definition of `verbose'".
+// Keep the single definition in ze_app.cpp and only declare it here.
+extern bool verbose;
 
 uint64_t isolate_lower_nbits(const uint64_t value,
```

**症状**（未打补丁时）：

```
/usr/bin/ld: /tmp/ccXDEnea.o:(.bss+0x0): multiple definition of `verbose';
              /tmp/ccewlRYq.o:(.bss+0x0): first defined here
collect2: error: ld returned 1 exit status
```

**根因**：上游 `common/src/ze_app.cpp:17` **也**写了 `bool verbose = false;`
（该文件 Copyright 已更新到 2026，属于**上游回归**），而 `common/include/common.hpp:66`
本来就有 `extern bool verbose;` —— `ze_peak.cpp` 的那份定义是多余的。
`-fcommon` **不能**解决（两者都是带初始化的强定义）。

**为什么改 `ze_peak.cpp` 而不是 `ze_app.cpp`**：`common/` 是**所有** perf_tests 共享的，
改它会影响 `ze_bandwidth` / `ze_peer` / `ze_pingpong` 等；而本项目只需要 `ze_peak`。

原文件已保留为 `ze_peak/src/ze_peak.cpp.orig`。

---

## 改动 2（新增物，非改上游）：`shim/level_zero/`

本机 `/usr/include/level_zero/` 只有：

```
ze_api.h (v1.13.1)  ze_ddi.h  ze_ddi_common.h  zes_api.h  zes_ddi.h  zet_api.h  zet_ddi.h
```

**缺 `zer_api.h`**，而上游 `common/src/ze_app.cpp:9` 无条件：

```cpp
#include <level_zero/zer_api.h>
```

`shim/level_zero/` 里放的是自洽的 **v1.15.31** 头文件集（16 个 `.h`），
构建时 `-I shim` 放在系统 `-I` 之前。

> **为什么不能只放 `zer_api.h`**：它第 18 行是 `#include "ze_api.h"`（引号 =
> 先搜**本文件所在目录**），所以 `ze_api.h` 也必须同目录，且必须与它版本自洽。
> 只放一个会报 `zer_api.h:18:10: fatal error: ze_api.h: No such file or directory`。
>
> 顺带确认：**`zer_api.h` 的符号在本项目里其实一次都没用**（`grep -n "ZER"` 只命中
> 无关的 "zero drivers" 字符串）。所以本改动**不影响测量结果**，纯粹是编译期补头文件。
> 替代方案是删掉那句 include —— 但那就要改上游源文件，故选补头文件。

---

## 附：`-fcommon` 的必要性

上游头文件里有未加 `extern` 的全局定义，多个 TU 各生成一份，GCC 10+ 默认
`-fno-common` 会链接失败。所以构建命令带 `-fcommon`（与 `shim`、源码补丁三者配合后
链接通过）。

---

## 未改动项（确认）

- ❌ 没有改任何 `.cl` / `.spv` 内核 → **测量内核与原版逐字节相同**。
- ❌ 没有改任何测量逻辑 / FLOP 计数 / 迭代次数默认值。
- ❌ 没有加 `-ffast-math` 或任何会改变浮点语义的编译开关；只有 `-O3`。
      （这条很重要：正因为上游**没有**做任何激进的浮点优化，它才适合当仲裁者。）
