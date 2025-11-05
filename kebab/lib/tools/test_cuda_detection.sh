#!/bin/bash
# Test script for CUDA auto-detection in Makefile

echo "=========================================="
echo "Testing CUDA Auto-Detection"
echo "=========================================="
echo ""

# Test 1: Check help output
echo "[Test 1] Checking 'make help' output..."
if make help 2>&1 | grep -q "CUDA_PATH"; then
    echo "  ✓ CUDA_PATH detected"
else
    echo "  ✗ CUDA_PATH not found in help output"
    exit 1
fi

if make help 2>&1 | grep -q "NVCC"; then
    echo "  ✓ NVCC detected"
else
    echo "  ✗ NVCC not found in help output"
    exit 1
fi

if make help 2>&1 | grep -q "CUDA_ARCH"; then
    echo "  ✓ CUDA_ARCH detected"
else
    echo "  ✗ CUDA_ARCH not found in help output"
    exit 1
fi

if make help 2>&1 | grep -q "NCU"; then
    echo "  ✓ NCU status shown"
else
    echo "  ✗ NCU status not shown"
    exit 1
fi

echo ""

# Test 2: Check gpu-info
echo "[Test 2] Checking 'make gpu-info' output..."
if make gpu-info 2>&1 | grep -q "GPU Information"; then
    echo "  ✓ GPU info command works"
else
    echo "  ✗ GPU info command failed"
    exit 1
fi

echo ""

# Test 3: Verify CUDA_PATH is set correctly
echo "[Test 3] Verifying CUDA_PATH..."
CUDA_PATH_FROM_MAKE=$(make help 2>&1 | grep "CUDA_PATH:" | awk '{print $2}')
if [ -d "$CUDA_PATH_FROM_MAKE" ]; then
    echo "  ✓ CUDA_PATH exists: $CUDA_PATH_FROM_MAKE"
else
    echo "  ✗ CUDA_PATH does not exist: $CUDA_PATH_FROM_MAKE"
    exit 1
fi

echo ""

# Test 4: Verify NVCC exists
echo "[Test 4] Verifying NVCC..."
NVCC_FROM_MAKE=$(make help 2>&1 | grep "NVCC:" | awk '{print $2}')
if [ -x "$NVCC_FROM_MAKE" ]; then
    echo "  ✓ NVCC is executable: $NVCC_FROM_MAKE"
    NVCC_VERSION=$($NVCC_FROM_MAKE --version | grep "release" | awk '{print $5}')
    echo "    Version: $NVCC_VERSION"
else
    echo "  ✗ NVCC not found or not executable: $NVCC_FROM_MAKE"
    exit 1
fi

echo ""

# Test 5: Check NCU availability
echo "[Test 5] Checking NCU availability..."
NCU_FROM_MAKE=$(make help 2>&1 | grep "NCU:" | awk '{print $2}')
if [ "$NCU_FROM_MAKE" = "Not" ]; then
    echo "  ⚠ NCU not found (profiling will not work)"
    echo "    This is OK if Nsight Compute is not installed"
elif [ -x "$NCU_FROM_MAKE" ]; then
    echo "  ✓ NCU is executable: $NCU_FROM_MAKE"
else
    echo "  ⚠ NCU path detected but not executable: $NCU_FROM_MAKE"
fi

echo ""

# Test 6: Check nvidia-smi
echo "[Test 6] Checking nvidia-smi..."
NVIDIA_SMI_FROM_MAKE=$(make help 2>&1 | grep "NVIDIA_SMI:" | awk '{print $2}')
if [ "$NVIDIA_SMI_FROM_MAKE" = "Not" ]; then
    echo "  ✗ nvidia-smi not found"
    exit 1
elif [ -x "$NVIDIA_SMI_FROM_MAKE" ]; then
    echo "  ✓ nvidia-smi is executable: $NVIDIA_SMI_FROM_MAKE"
    GPU_COUNT=$($NVIDIA_SMI_FROM_MAKE --list-gpus | wc -l)
    echo "    Detected GPUs: $GPU_COUNT"
else
    echo "  ✗ nvidia-smi not executable: $NVIDIA_SMI_FROM_MAKE"
    exit 1
fi

echo ""

# Test 7: Verify CUDA_ARCH format
echo "[Test 7] Verifying CUDA_ARCH format..."
CUDA_ARCH_FROM_MAKE=$(make help 2>&1 | grep "CUDA_ARCH:" | awk '{print $2}')
if [[ "$CUDA_ARCH_FROM_MAKE" =~ ^sm_[0-9]+$ ]]; then
    echo "  ✓ CUDA_ARCH format is correct: $CUDA_ARCH_FROM_MAKE"
else
    echo "  ✗ CUDA_ARCH format is incorrect: $CUDA_ARCH_FROM_MAKE"
    exit 1
fi

echo ""
echo "=========================================="
echo "All Tests Passed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  CUDA_PATH:    $CUDA_PATH_FROM_MAKE"
echo "  NVCC:         $NVCC_FROM_MAKE"
echo "  CUDA_ARCH:    $CUDA_ARCH_FROM_MAKE"
echo "  NCU:          $NCU_FROM_MAKE"
echo "  NVIDIA_SMI:   $NVIDIA_SMI_FROM_MAKE"
echo ""
echo "You can now run:"
echo "  make build              # Compile the library"
echo "  make bench-gemm         # Run benchmarks"
if [ -x "$NCU_FROM_MAKE" ]; then
    echo "  make tune-gemm          # Profile with Nsight Compute"
fi
echo "=========================================="
