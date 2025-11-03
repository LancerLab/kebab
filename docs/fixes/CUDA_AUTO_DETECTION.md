# CUDA 自动检测和版本选择

## 功能概述
Makefile现在具备智能CUDA版本检测和自动选择功能，无需手动配置即可解决驱动-工具包兼容性问题。

## 工作原理

### 1. 驱动检测
```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# 输出: 570.172.08
```

### 2. 版本映射
根据NVIDIA官方兼容性矩阵，将驱动版本映射到支持的最高CUDA版本：
```
570+ → CUDA 12.8  ← 你的系统
580+ → CUDA 13.0
560+ → CUDA 12.6
...
```

### 3. 工具包扫描
自动扫描系统中所有可用的CUDA安装：
```
/usr/local/cuda-12.8  ← 选中
/usr/local/cuda-13.0  ← 跳过（超出驱动支持）
/usr/local/cuda
```

### 4. 智能选择
选择最高但不超过驱动支持版本的CUDA工具包。

## 使用方法

### 正常使用（推荐）
```bash
make build      # 自动选择CUDA 12.8
make bench-all  # 自动选择CUDA 12.8
make test       # 自动选择CUDA 12.8
```

### 手动覆盖（可选）
```bash
export CUDA_PATH=/usr/local/cuda-12.8
make build
```

## 输出示例
```
Detected driver supports CUDA version: 12.8
Selected compatible CUDA toolkit: /usr/local/cuda-12.8 (version 12.8)
```

## 错误处理
如果没有找到兼容版本，会显示详细的诊断信息：
- 驱动支持的CUDA版本
- 所有可用的CUDA安装
- 解决建议

## 测试脚本
运行 `./test_cuda_auto_detection.sh` 来测试检测功能。

## 优势
1. **零配置**: 无需手动设置环境变量
2. **自动兼容**: 避免运行时版本冲突
3. **智能选择**: 总是选择最优的兼容版本
4. **向后兼容**: 仍支持手动CUDA_PATH设置
5. **详细诊断**: 提供清晰的错误信息和解决建议