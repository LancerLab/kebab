# CUDA 自动检测功能修复说明

## 问题解决

之前运行以下命令会报错找不到 `ncu`：
- `make tune-gemm`
- `make tune-elementwise-add`  
- `make tune-all`

现在已经全部修复。

### 修复的问题：

1. ✅ **ncu 工具自动检测** - 不再需要手动设置 PATH
2. ✅ **make tune-all 命令修复** - 正确调用所有算子的 profiling
3. ✅ **make bench-all 命令修复** - 正确调用所有算子的 benchmark

## 新增功能

Makefile 现在会自动检测并配置以下 CUDA 工具：

### 1. CUDA 安装路径 (CUDA_PATH)
自动搜索：
- `which nvcc` 返回的路径
- `/usr/local/cuda`
- `/usr/local/cuda-12`
- `/usr/local/cuda-11`
- `/opt/cuda`

### 2. NVCC 编译器
- 自动设置为 `$(CUDA_PATH)/bin/nvcc`
- 验证文件是否存在

### 3. Nsight Compute (ncu) - 性能分析工具
自动搜索：
- 系统 PATH 中的 `ncu`
- `$(CUDA_PATH)/bin/ncu`
- `$(CUDA_PATH)/nsight-compute/ncu`
- `/usr/local/cuda/bin/ncu`
- `/opt/nvidia/nsight-compute/ncu`

### 4. nvidia-smi - GPU 驱动工具
自动搜索：
- 系统 PATH
- `/usr/bin/nvidia-smi`
- CUDA 安装目录

### 5. GPU 架构 (CUDA_ARCH)
- 使用 `nvidia-smi` 自动检测 GPU 计算能力
- 自动转换为 sm_XX 格式（如 sm_90 对应 H100）

## 使用方法

### 查看自动检测结果

```bash
make help
```

输出示例：
```
==========================================
CuTeKernelLib Build System
==========================================
Detected Configuration:
  OS:           Linux
  CUDA_PATH:    /usr/local/cuda
  NVCC:         /usr/local/cuda/bin/nvcc
  CUDA_ARCH:    sm_90
  BUILD_MODE:   release
  NCU:          /usr/local/cuda/bin/ncu
  NVIDIA_SMI:   /usr/bin/nvidia-smi
==========================================
```

### 查看 GPU 详细信息

```bash
make gpu-info
```

### 运行性能分析（现在可以直接使用）

```bash
# 分析单个算子
make tune-elementwise-add
make tune-gemm

# 分析所有算子（已修复）
make tune-all
```

### 运行基准测试

```bash
# 测试单个算子
make bench-elementwise-add
make bench-gemm

# 测试所有算子（已修复）
make bench-all
```

## 技术细节

### 修复 tune-all 和 bench-all

**问题原因：**
- `OPERATORS` 变量使用下划线：`elementwise_add gemm`
- 但 target 名称使用连字符：`tune-elementwise-add`, `bench-elementwise-add`
- 导致 `make tune-elementwise_add` 找不到 target

**解决方案：**
在循环中将下划线转换为连字符：
```makefile
@for op in $(OPERATORS); do \
    target=$$(echo $$op | tr '_' '-'); \
    $(MAKE) tune-$$target || exit 1; \
done
```

## 错误提示改进

### 如果找不到 ncu

现在会显示详细的搜索路径和安装建议：

```
✗ ERROR: Nsight Compute (ncu) not found or not executable

Searched locations:
  - System PATH
  - /usr/local/cuda/bin/ncu
  - /usr/local/cuda/nsight-compute/ncu
  - /usr/local/cuda/bin/ncu

Nsight Compute is required for profiling.
Installation options:
  1. Install with CUDA toolkit (recommended)
  2. Download standalone from:
     https://developer.nvidia.com/nsight-compute

After installation, ensure ncu is in PATH:
  export PATH=/usr/local/cuda/bin:$PATH
Or set NCU variable:
  make tune-elementwise-add NCU=/path/to/ncu
```

## 手动覆盖（可选）

如果自动检测不正确，可以手动指定：

```bash
# 指定 CUDA 路径
export CUDA_PATH=/custom/cuda/path
make build

# 指定 GPU 架构
export CUDA_ARCH=sm_80
make build

# 指定 NCU 路径
make tune-gemm NCU=/custom/path/to/ncu
```

## 测试验证

运行测试脚本验证所有功能：

```bash
./test_cuda_detection.sh
```

测试内容：
- ✅ make help 显示配置信息
- ✅ make gpu-info 显示 GPU 信息
- ✅ bench-elementwise-add 目标存在
- ✅ bench-gemm 目标存在
- ✅ bench-all 正确调用所有 bench 目标
- ✅ tune-elementwise-add 目标存在
- ✅ tune-gemm 目标存在
- ✅ tune-all 正确调用所有 tune 目标

## 优势

1. ✅ **零配置启动**：大多数情况下无需手动设置环境变量
2. ✅ **智能搜索**：自动在多个标准位置查找 CUDA 工具
3. ✅ **清晰错误**：找不到工具时提供详细的安装指导
4. ✅ **灵活覆盖**：支持手动指定路径
5. ✅ **即时可用**：所有 `make tune-*` 和 `make bench-*` 命令现在可以直接运行
6. ✅ **批量操作**：`make tune-all` 和 `make bench-all` 现在正常工作

## 修改的文件

- `Makefile` - 添加 CUDA 工具自动检测，修复 tune-all 和 bench-all
- `README.md` - 更新文档说明自动检测功能
- `test_cuda_detection.sh` - 新增测试脚本

## 快速验证

```bash
# 1. 查看检测结果
make help

# 2. 查看 GPU 信息
make gpu-info

# 3. 测试单个命令
make tune-elementwise-add

# 4. 测试批量命令（已修复）
make tune-all
```
