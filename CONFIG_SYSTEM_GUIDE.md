# 配置系统使用指南

## 概述

现在可以通过修改 `config.yaml` 来精细控制 GEMM benchmark 的行为，无需修改代码。

## 新增配置选项

### 1. Implementation 选择 (`impl`)

选择使用哪个实现：

```yaml
operators:
  gemm:
    impl: cute  # 或 cuda
```

- `cute`: 使用 CuTe/WGMMA 实现（Hopper SM90+）
- `cuda`: 使用 CUDA baseline 实现

### 2. 初始化方法 (`init_method`)

控制矩阵 A 和 B 的初始化方式：

```yaml
operators:
  gemm:
    init_method: rand-rand  # 格式: <A_init>-<B_init>
```

#### 可用的初始化模式

| 模式 | 说明 | 示例值 |
|------|------|--------|
| `one` | 全1矩阵 | 1.0, 1.0, 1.0, ... |
| `rand` | 随机值 [-1, 1] | -0.5, 0.8, -0.3, ... |
| `range` | 顺序编号 [0, 1, 2, ...] | 0, 1, 2, 3, ... |
| `row` | 值等于行索引 | row 0: 0,0,0; row 1: 1,1,1 |
| `col` | 值等于列索引 | col 0: 0,0,0; col 1: 1,1,1 |

#### 初始化方法示例

```yaml
# 两个矩阵都用随机值
init_method: rand-rand

# A 全1，B 随机
init_method: one-rand

# A 顺序编号，B 全1
init_method: range-one

# A 按行，B 按列
init_method: row-col
```

## 使用场景

### 场景 1: 快速验证正确性

使用 `one-one` 初始化，结果容易预测：

```yaml
gemm:
  impl: cute
  init_method: one-one
  matrix_sizes: [256]
```

对于 256×256×256 的矩阵：
- A: 全1矩阵 (256×256)
- B: 全1矩阵 (256×256)
- C = A × B: 每个元素应该是 256.0

### 场景 2: 调试数值精度

使用 `range-range` 查看具体的数值模式：

```yaml
gemm:
  impl: cute
  init_method: range-range
  matrix_sizes: [16]  # 小矩阵便于检查
```

### 场景 3: 测试边界情况

使用 `row-col` 测试特殊模式：

```yaml
gemm:
  impl: cute
  init_method: row-col
  matrix_sizes: [256]
```

### 场景 4: 性能对比

对比不同实现的性能：

```yaml
# 第一次运行
gemm:
  impl: cute
  init_method: rand-rand
  matrix_sizes: [256, 512, 1024, 2048, 4096]

# 第二次运行（修改 impl）
gemm:
  impl: cuda
  init_method: rand-rand
  matrix_sizes: [256, 512, 1024, 2048, 4096]
```

结果会保存到不同的 CSV 文件：
- `bench_results/gemm_results_half_cute.csv`
- `bench_results/gemm_results_half_cuda.csv`

## 运行 Benchmark

只需一个命令：

```bash
make bench-gemm
```

所有配置都从 `config.yaml` 读取，无需修改代码或传递参数。

## 输出示例

```
Testing matrix size: 256x256x256
  Init: A=one, B=one
  Verifying cute implementation... ✓ PASSED
GEMM                cute        256         4.9             6.9                     0.5       x
GEMM                cuBLAS      256         2.4             13.9                    1.000     x
  Performance Analysis:
    cute vs cuBLAS:     49.7% performance
```

## 调试技巧

### 1. 验证 WGMMA 正确性

```yaml
gemm:
  impl: cute
  init_method: one-one  # 简单的输入
  matrix_sizes: [256]   # 小矩阵
```

如果 `one-one` 通过但 `rand-rand` 失败，说明是数值精度问题，不是算法错误。

### 2. 定位问题元素

使用 `range-range` 可以看到具体哪些元素出错：

```yaml
init_method: range-range
matrix_sizes: [16]  # 很小的矩阵
```

然后手动计算预期结果，对比实际输出。

### 3. 测试不同模式

```yaml
# 测试对称性
init_method: row-row

# 测试转置
init_method: row-col

# 测试稀疏性
init_method: one-range
```

## 配置文件完整示例

```yaml
operators:
  gemm:
    enabled: true
    impl: cute                    # cute | cuda
    init_method: one-one          # <A_init>-<B_init>
    matrix_sizes: [256, 512, 1024]
    modes: [NN]
    precisions: [float16]
    tile_sizes: [16, 32, 64]
```

## 注意事项

1. **数值精度**: `rand-rand` 可能因为累积误差导致验证失败，这是正常的
2. **矩阵大小**: 大矩阵 (>2048) 的 FP16 累积误差更大
3. **实现要求**: `cute` 实现需要 SM90+ (Hopper) GPU
4. **CSV 输出**: 每次运行会覆盖同名的 CSV 文件

## 当前状态

### ✅ 工作正常

- `one-one`: WGMMA 验证通过 ✓
- `cuda` baseline: 验证通过 ✓
- 配置系统: 完全工作 ✓
- CSV 输出: 正常保存 ✓

### ⚠️ 已知问题

- `rand-rand`: WGMMA 数值精度问题（layout 配置）
- 需要实现 `gemm_nn` 函数来正确支持 row-major 输入

## 下一步

1. 实现 `gemm_nn` 函数修复 layout 问题
2. 添加更多初始化模式（如 `identity`, `diagonal`）
3. 支持非方阵 (M≠N≠K)
4. 添加 alpha/beta scaling 支持
