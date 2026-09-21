# ① 硬件健康 / 稳定性 / 压力测试

**优先级：P0（必须最先做）**
**文档位置：** `docs/TODO/01-health-stress.md`

---

## 1. 测试目标

1. 确认两张 Max 1100 **无硬件故障**（ECC、显存、Xe Link、电源、固件）
2. 确认在**长时间满载**下能稳定运行、不降频、不报错
3. 建立**温度 / 功耗 / 频率**的受控区间，供后续测试判断异常
4. 排查 ES 样片的潜在稳定性问题（这是 ES 平台的**最高风险项**）

## 2. 为什么这个必须最先做

- 若硬件本身有问题，后续所有性能数字都不可信
- ES 样品 + 300 W 满载 + 双卡同时压测，是最容易暴露问题的场景
- 早发现可省下大量无效调优时间

---

## 3. 工具

`xpu-smi`（已装，无需额外准备）。

`xpu-smi` 的 `diag` 子命令能力比常见认知更强，支持：
- 分级诊断（`-l [level]`）
- 单项测试（`--singletest [testIds]`）
- **多卡同时压力测试**（`--stress`）
- `--precheck` 快速预检与类型列举

---

## 4. 执行步骤

### Step 0：环境准备
```bash
source /opt/intel/oneapi/setvars.sh
xpu-smi discovery          # 确认 2 卡在线
xpu-smi ps                 # 确认无其他进程占用 GPU
```

### Step 1：器件健康状态
```bash
xpu-smi health -l                       # 全部设备概览
xpu-smi health -d 0                     # 设备 0 详情
xpu-smi health -d 0 -j                  # JSON 便于归档
```

组件类型（`-c` 参数）：
| ID | 组件 |
|---|---|
| 1 | GPU Core Temperature |
| 2 | GPU Memory Temperature |
| 3 | GPU Power |
| 4 | GPU Memory |
| 5 | Xe Link Port |
| 6 | GPU Frequency |

```bash
for c in 1 2 3 4 5 6; do xpu-smi health -d 0 -c $c; xpu-smi health -d 1 -c $c; done
```

### Step 2：快速预检
```bash
xpu-smi diag --precheck --listtypes     # 列出可用的诊断项
xpu-smi diag --precheck --gpu           # 针对 GPU 快速预检
xpu-smi diag --precheck -j > precheck.json
```

### Step 3：分级诊断
```bash
xpu-smi diag -d 0 -l 1                  # 轻度
xpu-smi diag -d 0 -l 2                  # 中度
xpu-smi diag -d 0 -l 3                  # 重度（注释：确认可用级别）
# 对设备 1 同样执行
```

### Step 4：单项测试（针对可疑项）
```bash
xpu-smi diag -d 0 --singletest 1        # testId 从 --listtypes 获取
```

### Step 5：双卡压力测试（核心步骤）
```bash
# 前台观测：另开一个终端跑遥测
xpu-smi diag -d 0,1 --stress --stresstime 600
```
在**另一个终端并行采集遥测**：
```bash
xpu-smi dump -d -1 -m 0,1,2,3,4,5,6,7,8,12,13,14 -i 1000 -n 620 -j > stress_telemetry.json
```

建议时长梯度：**5 min → 30 min → 2 h**（按需升级，用于捕捉间歇性问题）。

### Step 6：压力后复检
```bash
xpu-smi health -l
xpu-smi health -d 0 -j > health_after.json
xpu-smi stats -d 0
xpu-smi stats -d 1
```

---

## 5. 关键指标与判读

| 指标 | 期望 | 异常信号 |
|---|---|---|
| **GPU Frequency** | 稳定 1550 MHz | 掉频 → 温控/功耗墙 |
| **GPU Core Temperature** | < 85–90 ℃（具体阈值查规格） | 持续贴近上限 |
| **GPU Memory Temperature** | 稳定 | 持续上升 |
| **GPU Power** | ≤ 300 W | 明显超限或剧烈抖动 |
| **Reset Counter** | 0 | > 0 → 严重问题 |
| **Programming Errors** | 0 | > 0 → 驱动/kernel 问题 |
| **Driver Errors** | 0 | > 0 → 需调查 |
| **Cache/Mem Errors Correctable** | 增长速率应为 0 或极低 | 快速累积 → 显存劣化 |
| **Cache/Mem Errors Uncorrectable** | **0** | > 0 → 硬件故障，立即停测 |
| **ECC State** | enabled | 被关闭需说明原因 |
| **Xe Link Port 健康** | 全部 6 端口 up | 端口 down → 链路故障 |

> 压力期间「EU Array Active」应接近 100%，「Idle」接近 0 —— 可用来确认压力测试**真的压满了**。

---

## 6. 输出物

- `precheck.json`、`health_before.json`、`health_after.json`
- `stress_telemetry.json` + 温度/功耗/频率时间序列曲线
- 压力测试期间是否出现错误计数的结论
- 温控/功耗墙是否存在及触发条件

---

## 7. 判定门禁（Gate）

```
通过条件：
  ✓ 无 Uncorrectable Error
  ✓ Reset / Programming / Driver Error 计数保持 0
  ✓ 600 s 满载频率稳定在 1550 MHz
  ✓ 温度在规格范围内且已收敛（不再上升）
  ✓ 6 个 Xe Link 端口全部 up

未通过 → 停止后续所有测试，先解决硬件问题
```

---

## 8. 注意事项

1. `xpu-smi diag -l` 的**级别取值范围**在帮助里未明确，先用 `--listtypes` 探测可用项。
2. 压力测试会**显著发热并拉满功耗**，确保机房散热与供电余量。
3. 这是**双卡同时**压测，注意整机功耗是否超出 PSU 能力。
4. `--stresstime` 单位需确认（秒/分钟），先用短时长验证。
5. 遥测采集的 `-n` 次数要覆盖整个压力周期。
6. 若出现 Uncorrectable Error，**先保存日志**（`xpu-smi log`）再重启。
