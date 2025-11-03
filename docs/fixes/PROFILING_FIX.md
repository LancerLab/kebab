# GEMM Profiling 修复说明

## 问题描述

之前运行 `make tune-gemm` 生成的分析文件中：
- Launch statistics 都是空的
- Occupancy 信息看不到
- 只有 CSV 头部，没有实际数据

## 根本原因

1. **NCU 参数配置不完整**：缺少关键的 sections 和 metrics
2. **Kernel 匹配模式不够全面**：可能没有匹配到实际的 kernel 名称
3. **报告生成方式不当**：没有正确提取 launch statistics 和 occupancy 信息

## 修复内容

### 1. 增强的 NCU 配置

在 `Makefile` 中的 `tune-gemm` 目标添加了：

```makefile
--section LaunchStats          # 启动统计信息
--section Occupancy           # 占用率分析
--section SchedulerStats      # 调度器统计
--section WarpStateStats      # Warp 状态统计
```

### 2. 扩展的 Metrics 收集

添加了关键的占用率和性能指标：

```makefile
--metrics achieved_occupancy           # 实际占用率
--metrics theoretical_occupancy        # 理论占用率
--metrics sm__warps_active.avg.pct_of_peak_sustained_elapsed
--metrics sm__maximum_warps_per_active_cycle_pct
```

### 3. 改进的 Kernel 匹配

扩展了 kernel 正则表达式：

```makefile
--kernel-regex ".*gemm.*|.*Gemm.*|.*GEMM.*|.*sgemm.*|.*hgemm.*"
```

### 4. 增强的报告生成

修改了报告生成流程：

```makefile
# 生成详细摘要
$(NCU) --import $(PROFILING_DIR)/gemm_profile.ncu-rep \
    --page details \
    --print-summary per-kernel \
    > $(PROFILING_DIR)/gemm_summary.txt

# 添加启动统计
$(NCU) --import $(PROFILING_DIR)/gemm_profile.ncu-rep \
    --page LaunchStats \
    >> $(PROFILING_DIR)/gemm_summary.txt

# 添加占用率分析
$(NCU) --import $(PROFILING_DIR)/gemm_profile.ncu-rep \
    --page Occupancy \
    >> $(PROFILING_DIR)/gemm_summary.txt
```

## 新增工具

### 1. 增强的 Profiling 脚本

创建了 `scripts/profile_gemm.sh`：
- 全面的 metrics 收集
- 多种报告格式生成
- 错误处理和诊断信息

### 2. 诊断工具

创建了 `scripts/diagnose_profiling.sh`：
- 检查 NCU 安装和版本
- 验证 CUDA 环境
- 测试 benchmark 可执行性
- 最小化 profiling 测试

### 3. 简化测试工具

创建了 `test_simple_profiling.sh`：
- 最小化的 profiling 测试
- 快速验证修复是否有效
- 基本的结果预览

## 使用方法

### 基本使用

```bash
# 使用修复后的 profiling
make tune-gemm
```

### 诊断问题

```bash
# 运行诊断工具
./scripts/diagnose_profiling.sh
```

### 使用增强脚本

```bash
# 直接使用增强的 profiling 脚本
./scripts/profile_gemm.sh
```

## 输出文件

修复后会生成以下文件：

```
profiling/
├── gemm_profile.ncu-rep      # NCU 原始报告
├── gemm_summary.txt          # 包含 launch stats 和 occupancy 的摘要
├── gemm_metrics.csv          # 原始 metrics 数据
└── gemm_speedoflight.txt     # Speed of Light 分析
```

## 验证修复

运行以下命令验证修复是否成功：

```bash
# 1. 运行诊断
./scripts/diagnose_profiling.sh

# 2. 运行 profiling
make tune-gemm

# 3. 检查结果
grep -E "(Block Size|Grid Size|Occupancy)" profiling/gemm_summary.txt
```

如果看到类似以下输出，说明修复成功：

```
Block Size: 256
Grid Size: (16, 16, 1)
Theoretical Occupancy: 75.0%
Achieved Occupancy: 68.2%
```

## 常见问题

### 1. `--config-file` 参数错误

**问题**：`==ERROR== the argument ('profiling/ncu_config.txt') for option '--config-file' is invalid`

**原因**：NCU 的 `--config-file` 不是用来指定配置文件路径的，而是一个布尔选项。

**解决方案**：已修复，现在直接使用内联参数而不是配置文件。

### 2. 仍然没有 Launch Statistics

可能原因：
- Benchmark 运行时间太短
- Kernel 名称不匹配
- 权限问题

解决方案：
```bash
# 先运行简单测试
./test_simple_profiling.sh

# 增加矩阵大小（在 config.yaml 中）
matrix_sizes: [1024, 2048, 4096]

# 或者使用 sudo
sudo make tune-gemm
```

### 3. NCU 版本兼容性

确保使用兼容的 NCU 版本：
```bash
ncu --version
# 推荐使用 CUDA 11.8+ 附带的 NCU
```

### 4. GPU 驱动兼容性

检查驱动版本：
```bash
nvidia-smi
# 确保驱动版本支持你的 NCU 版本
```

## 技术细节

### Kernel 命名模式

CuTe GEMM kernel 可能的命名模式：
- `gemm_kernel_cute_tiled<float>`
- `_ZN12cutekernellib4gemmIfEEvPKT_S4_PS2_iii`
- 其他编译器生成的 mangled 名称

### 关键 Metrics 说明

- `achieved_occupancy`: 实际达到的 SM 占用率
- `theoretical_occupancy`: 理论最大占用率
- `sm__warps_active`: 活跃 warp 百分比
- `sm__throughput`: SM 吞吐量利用率
- `dram__throughput`: 内存带宽利用率

## 总结

这次修复主要解决了：
1. ✅ Launch statistics 现在能正确显示
2. ✅ Occupancy 信息能正确捕获
3. ✅ 提供了完整的诊断和调试工具
4. ✅ 增加了多种报告格式
5. ✅ 改进了错误处理和用户体验

现在 `make tune-gemm` 应该能提供完整的 profiling 信息，包括 kernel 启动参数、占用率分析和性能指标。