#!/usr/bin/env bash
# 02-compute-peak / ze_peak_src / build.sh
#
# 构建上游 ze_peak（Intel 官方 level-zero-tests 的 perf_tests/ze_peak）。
# 上游用 CMake；这里用一条 g++ 命令，因为 CMake 路线在本机还缺 ninja + 一堆
# lzt 测试框架的依赖，而 ze_peak 本身的依赖是**零**。
#
# 用法：  ./build.sh          # 只构建
#         ./build.sh run      # 构建后跑一次全量（设备 0）
#         ./build.sh run 1    # 构建后跑一次全量（设备 1）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BUILD="$HERE/build"
mkdir -p "$BUILD"

# ① .spv 内核是**预编译**的，且被代码以**相对路径**加载 → 必须和可执行文件同目录。
#    所以运行时的 cwd 也必须是 build/。
cp -f ze_peak/kernels/*.spv "$BUILD/"

# ② -I shim 放在 -I /usr/include 之前：本机 /usr/include/level_zero/ 缺 zer_api.h
#    （ze_app.cpp:9 无条件 include 它），shim/level_zero/ 里放的是自洽的 v1.15 头文件集。
# ③ -fcommon：上游头文件里有未加 extern 的全局（多 TU 重复定义）。
# ④ 源码里唯一需要打的补丁见 PATCHES.md（ze_peak.cpp 的 `bool verbose` 重复定义）。
g++ -O3 -std=c++17 -fcommon \
    -I "$HERE/shim" \
    -I "$HERE/ze_peak/include" \
    -I "$HERE/common/include" \
    ze_peak/src/*.cpp \
    common/src/ze_app.cpp \
    -o "$BUILD/ze_peak" \
    -lze_loader -lpthread

echo "[build] wrote $BUILD/ze_peak"

if [[ "${1:-}" == "run" ]]; then
    DEV="${2:-0}"
    LOG="$HERE/logs/ze_peak_dev${DEV}.log"
    mkdir -p "$HERE/logs"
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    echo "[run] device $DEV -a -i 50 -w 10  ->  $LOG   (约 15 分钟)"
    cd "$BUILD"
    timeout 3600 ./ze_peak -d "$DEV" -a -i 50 -w 10 >"$LOG" 2>&1
    echo "[run] done"
    sed -n '/^Global memory/,/^<</p' "$LOG"
fi
