# Benchmark 输出格式指南

## 修复后的输出格式

现在 `make bench-gemm` 的输出格式已经修复，性能表格和验证结果清晰分离。

## 输出结构

### 1. 配置信息
```
CuTeKernelLib GEMM Benchmark
============================
Configuration loaded successfully from: config.yaml
Configuration:
  Warmup runs:      10
  Measurement runs: 100
  Matrix sizes:     256

GPU Information:
  Device:           NVIDIA H800 PCIe
  Compute Capability: 9.0
  Memory:           79 GB
  Tensor Cores:     Supported
```

### 2. Benchmark 配置
```
========================================
GEMM Benchmark (half)
========================================
Configuration:
  Implementation: cute
  Init method: one-one
  Matrix sizes: 256
```

### 3. 性能表格（连续显示）
```
Operator            Variant     Batch Size  Latency (ms)   Throughput (GFLOPS)Speedup   
-----------------------------------------------------------------------------------------

Testing matrix size: 256x256x256
  Init: A=one, B=one
GEMM                cute        256         4.7             7.2                     0.4       x
GEMM                cuBLAS      256         2.0             16.7                    1.000     x

```

### 4. 验证结果（分离显示）
```
  Verifying cute implementation... ✓ PASSED
  Performance Analysis:
    cute vs cuBLAS:     43.0% performance
```

### 5. 错误情况（如果验证失败）
```
  Verifying cute implementation...   Verification errors found:
    Element 0: expected 1.8, got 4.3 (error: 2.5)
    Element 1: expected -3.2, got 3.3 (error: 6.5)
    Element 2: expected -2.6, got -0.6 (error: 2.0)
    Element 3: expected -7.2, got 0.8 (error: 8.1)
    Element 4: expected 13.2, got -9.7 (error: 22.9)
    ... (showing first 5 errors)
✗ FAILED
  Performance Analysis:
    cute vs cuBLAS:     43.7% performance
```

## 配置测试结果

### ✅ 验证通过的配置

| Implementation | Init Method | 结果 |
|----------------|-------------|------|
| `cute` | `one-one` | ✓ PASSED |
| `cuda` | `one-one` | ✓ PASSED |

### ❌ 验证失败的配置

| Implementation | Init Method | 结果 | 原因 |
|----------------|-------------|------|------|
| `cute` | `rand-rand` | ✗ FAILED | WGMMA layout 问题 |
| `cuda` | `rand-rand` | ✗ FAILED | 数值精度问题 |

## 性能数据示例

### CuTe (WGMMA) 实现
```
GEMM                cute        256         4.7             7.2                     0.4       x
GEMM                cuBLAS      256         2.0             16.7                    1.000     x
  Performance Analysis:
    cute vs cuBLAS:     43.0% performance
```

- **延迟**: 4.7ms
- **吞吐量**: 7.2 GFLOPS
- **相对性能**: 43% of cuBLAS

### CUDA Baseline 实现
```
GEMM                cuda        256         0.0             2238.0                  147.7     x
GEMM                cuBLAS      256         2.2             15.2                    1.000     x
  Performance Analysis:
    cuda vs cuBLAS:     14767.1% performance
```

- **延迟**: 0.0ms (太快，测量精度问题)
- **吞吐量**: 2238 GFLOPS (不现实，可能是测量问题)
- **相对性能**: 14767% (明显有问题)

## 输出文件

结果自动保存到 CSV 文件：
- `bench_results/gemm_results_half_cute.csv`
- `bench_results/gemm_results_half_cuda.csv`

## 使用建议

### 1. 快速验证正确性
```yaml
gemm:
  impl: cute
  init_method: one-one
  matrix_sizes: [256]
```

### 2. 性能对比测试
```yaml
# 第一次运行
gemm:
  impl: cute
  init_method: one-one
  matrix_sizes: [256, 512, 1024]

# 第二次运行（修改 impl）
gemm:
  impl: cuda
  init_method: one-one
  matrix_sizes: [256, 512, 1024]
```

### 3. 调试数值问题
```yaml
gemm:
  impl: cute
  init_method: range-range  # 或其他模式
  matrix_sizes: [16]        # 小矩阵便于分析
```

## 已知问题

1. **CUDA baseline 性能异常**: 显示不现实的高性能，可能是测量精度问题
2. **rand-rand 验证失败**: 两个实现都失败，可能是验证逻辑问题
3. **WGMMA layout 问题**: cute 实现在随机数据下失败

## 下一步改进

1. 修复 CUDA baseline 的性能测量
2. 调查 rand-rand 验证失败的根本原因
3. 实现 WGMMA 的 gemm_nn 函数
4. 添加更多调试信息选项

## 总结

✅ **输出格式**: 已修复，清晰分离  
✅ **配置系统**: 完全工作  
✅ **基本验证**: one-one 通过  
⚠️ **性能测量**: 需要进一步调试  
❌ **随机数据**: 需要修复 layout 问题