# ⑧ 功耗、能效、频率与调度测试

**优先级：P2（可独立开展，不依赖其他测试）**
**文档位置：** `docs/TODO/08-power-efficiency.md`

---

## 1. 测试目标

1. **perf/W**（能效）：每瓦特功耗能换取多少算力/带宽/吞吐
2. **功耗-性能曲线**：`--powerlimit` 150→300 W 扫描
3. **频率-性能曲线**：`--frequencyrange` 200→1550 MHz 扫描
4. **调度策略影响**：`timeslice` vs `exclusive`
5. **Xe Link 端口开关**对多卡性能的影响
6. **虚拟化开销**：SR-IOV vGPU
7. 量化「**锁频在 1550 MHz**」对测试结果的影响

## 2. 当前配置基线

| 项目 | 值 |
|---|---|
| Power Limit | **300 W**（可调范围 **150–300 W**） |
| GPU Min / Max Frequency | **1550 / 1550 MHz**（min == max，**锁定**） |
| 可用频率档位 | 200 起，步进 50 MHz，至 1550 |
| Standby Mode | `default`（可选 `never`） |
| Scheduler Mode | `timeslice`，Interval 5000 µs，Yield Timeout 0 |
| Performance Factor | compute 50 / media 50 |
| ECC | enabled |
| Xe Link 端口 | 1–6 全部 up |
| **空闲实测功耗** | **43 W** |

> ⚠️ 注意：**频率已锁定**（min == max == 1550）。这意味着：
> - 好处：结果重复性极佳
> - 坏处：**测不到真实量产机的动态调频行为**
> - 做本测试时需先用 `--frequencyrange` 显式放开，才能观察频率变化

## 3. 工具

`xpu-smi` 一个工具全覆盖（`config` / `dump` / `stats` / `vgpu`），无需额外准备。

---

## 4. 执行步骤

### Step 0：记录基线配置
```bash
xpu-smi config -d 0
xpu-smi config -d 1
xpu-smi config -d 0 -j > config_baseline.json
xpu-smi config -d 1 -j >> config_baseline.json
```
**务必保存**，测试结束要恢复。

### Step 1：功耗与能耗采集方法

```bash
# metric 1 = GPU Power (W)，metric 8 = GPU Energy Consumed (J)
xpu-smi dump -d -1 -m 1,8 -i 1000 -n 60 -j > power_baseline.json
```
**perf/W 计算**：
```
能效 = 性能指标 / 平均功耗
例如：TFLOPS/W、GB/s/W、img/s/W、tokens/s/W
平均功耗 = Energy 差值 / 时间  （比采样 Power 再平均更准确）
```
> **推荐用 Energy(J) 差分**计算平均功耗 —— 比离散采样 Power 更准确，能捕捉瞬态尖峰。

### Step 2：功耗-性能曲线（`--powerlimit`）

固定一个可重复的工作负载（推荐用 ② 的 GEMM 或 ③ 的 BabelStream），扫功耗：
```bash
for pl in 150 175 200 225 250 275 300; do
  xpu-smi config -d 0 --powerlimit $pl
  # 运行基准，记录性能
  # 同时采集遥测
  xpu-smi dump -d 0 -m 1,2,3,8 -i 500 -n 60 -j > power_${pl}W.json
done
# 恢复
xpu-smi config -d 0 --powerlimit 300
```

记录表：

| Power Limit (W) | 性能 (TFLOPS / GB/s) | 实测平均功耗 (W) | 能效 (perf/W) | 频率 (MHz) | 温度 (℃) |
|---|---|---|---|---|---|
| 150 | | | | | |
| 175 | | | | | |
| 200 | | | | | |
| 225 | | | | | |
| 250 | | | | | |
| 275 | | | | | |
| 300 | | | | | |

**关注「能效拐点」**：通常存在一个功耗点，之后增加功耗带来的性能收益急剧递减。找到它就能给出**最优能效运行点**的建议。

### Step 3：频率-性能曲线（`--frequencyrange`）

```bash
for f in 200 400 600 800 1000 1200 1400 1550; do
  xpu-smi config -d 0 -t 0 --frequencyrange ${f},${f}
  # 运行基准 + 采集遥测
done
xpu-smi config -d 0 -t 0 --frequencyrange 200,1550     # 恢复为动态范围
```
固定频率（min=max=f）可以得到干净的**频率-性能**关系，用于：
- 验证性能是否与频率**线性相关**（若否，说明瓶颈不在核心频率）
- 反推「频率拐点」是否存在

> 注意 `-t 0` 指定 tile ID。本机是单 tile（tile 0）。

### Step 4：Standby Mode 测试
```bash
xpu-smi config -d 0 -t 0 --standby never     # 禁止进入低功耗待机
xpu-smi config -d 0 -t 0 --standby default   # 恢复
```
关注：**短任务/间歇性任务**下的响应延迟差异（待机唤醒有开销）。

### Step 5：调度策略测试（多进程共享场景）
```bash
# timeslice：多进程分时复用
xpu-smi config -d 0 -t 0 --scheduler timeslice,5000,0
# 或指定 yieldtimeout
xpu-smi config -d 0 -t 0 --scheduler timeslice,5000,1000
# exclusive：独占模式
xpu-smi config -d 0 -t 0 --scheduler exclusive
# timeout 模式
xpu-smi config -d 0 -t 0 --scheduler timeout,5000000
```
测试方法：
- **单进程**运行，比较不同模式的绝对性能
- **多进程竞争**同一卡时，比较不同模式的吞吐与公平性
- 参数范围：所有时间值 5000–100,000,000 µs

### Step 6：Xe Link 端口开关测试（多卡场景）
```bash
# 关闭部分端口，重跑 ④ 的 IMB 测试
xpu-smi config -d 0 -t 0 --xelinkport 1,0     # portId 1 关闭
# ... 测量带宽
xpu-smi config -d 0 -t 0 --xelinkport 1,1     # 恢复
```
**目的**：验证 Xe Link 带宽是否**随端口数线性缩放**（6 端口满配 ≈ 318 GB/s 理论值）。

可选：beaconing 测试
```bash
xpu-smi config -d 0 -t 0 --xelinkportbeaconing 1,1   # 点亮链路指示灯
```

### Step 7：ECC 开销测试（⚠️ 谨慎）
```bash
xpu-smi config -d 0 --memoryecc 0     # 关闭 ECC
# 重跑 ③ 带宽测试，对比差异
xpu-smi config -d 0 --memoryecc 1     # 恢复
```
**目的**：量化 ECC 对带宽/容量的开销。

> ⚠️ **高风险操作**：
> - 关闭 ECC 会降低数据完整性保护
> - 需要 **reboot** 或 reset 才能真正生效（看 `Pending` 字段）
> - 可能触发 GPU reset，影响正在运行的任务
> - **仅在确有必要且可接受风险时执行**

### Step 8：vGPU / SR-IOV 开销（可选）
```bash
xpu-smi vgpu -h           # 查看可用子命令
xpu-smi vgpu -c           # 创建 vGPU
# ... 在 vGPU 上运行基准
xpu-smi vgpu -d           # 删除 vGPU
```
**目的**：量化虚拟化带来的性能损失，评估多租户共享场景的可行性。

> 注意：`discovery` 显示当前 Function Type 为 `physical`，说明当前是物理功能模式。

### Step 9：能耗稳定性与恢复
```bash
xpu-smi config -d 0 --powerlimit 300
xpu-smi config -d 0 -t 0 --frequencyrange 200,1550
xpu-smi config -d 0 -t 0 --standby default
xpu-smi config -d 0 -t 0 --scheduler timeslice,5000,0
xpu-smi config -d 0
xpu-smi health -l
```
**必须验证配置已完全恢复**，并保存 `config_after.json` 与 `config_baseline.json` 对比。

---

## 5. 综合指标表

| 工作负载 | 功耗 (W) | 性能 | 能效 (perf/W) | 频率 | 温度 |
|---|---|---|---|---|---|
| GEMM FP32 峰值 | | ___ TFLOPS | ___ TFLOPS/W | | |
| GEMM BF16 峰值 | | ___ TFLOPS | ___ TFLOPS/W | | |
| BabelStream Triad | | ___ GB/s | ___ GB/s/W | | |
| ResNet-50 训练 | | ___ img/s | ___ img/s/W | | |
| Xe Link PingPong | | ___ GB/s | ___ GB/s/W | | |

**关键结论**：
- 能效拐点功耗：___ W
- 最优能效配置：___
- 满载整机功耗（双卡）：___ W
- perf/W 相比空闲基线的提升倍数：___ ×

---

## 6. 判读标准

| 现象 | 诊断 |
|---|---|
| 增加功耗无性能提升 | 已撞到其他瓶颈（带宽/并行度），功耗不是限制因素 |
| 性能随频率线性提升 | 纯算力受限 ✅ 符合预期 |
| 性能不随频率变化 | **瓶颈在显存/互连**（频率只影响核心） |
| 实测功耗远低于 Power Limit | 功耗墙未触发；或负载未压满 |
| 性能随功耗剧烈抖动 | 热降频 → 对照 ① 温度曲线 |
| 多进程 timeslice 模式下吞吐下降大 | 上下文切换开销；考虑 exclusive 模式 |
| Xe Link 带宽不随端口数缩放 | **怀疑未标定**（对照 ④） |

---

## 7. 注意事项

1. ⚠️ **`config` 会改变硬件状态**。测试前**先备份配置**，测试后**必须恢复**。
2. 频率当前锁定 1550 MHz。测频率曲线前要知道 **min==max** 这个前提。
3. **`--powerlimit` / `--frequencyrange` 需要时间生效**，改完先等稳态再测量。
4. `--memoryecc` 改动可能需要 **reboot/GPU reset**，且影响数据完整性 → **高风险**。
5. **双卡满载 600 W** → 确认 PSU 能力，避免触发整机保护。
6. 温度上升会导致频率回落 → 本测试需**同时记录温度**（对照 ①）。
7. `Energy(J)` 指标比 `Power(W)` 采样更适合算平均功耗。
8. 每档配置**至少测 3 次取中位数**，排除瞬态干扰。
9. **恢复配置后必须复测一次**，确认性能回到基线（防止残留状态）。

---

## 8. 与其他测试的衔接

| 衔接 | 用途 |
|---|---|
| ← ② 算力峰值 | 提供 GEMM 负载用于功耗扫描 |
| ← ③ 显存带宽 | 提供带宽负载；判断频率是否只影响核心 |
| ← ④ 互连 | Xe Link 端口开关测试的基础 |
| ← ① 健康压力 | 温度/功耗墙数据 |
| → 交付物 | 能效报告 + 最优运行配置建议 |
