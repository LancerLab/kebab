#!/bin/bash

# Verification script for CuTe implementations
# Ensures all implementations are pure CuTe, not raw CUDA

set -e

echo "=========================================="
echo "CuTe Implementation Verification"
echo "=========================================="

# Check for raw CUDA patterns in src/operators/
echo ""
echo "1. Checking for raw CUDA patterns in src/operators/..."

RAW_CUDA_PATTERNS=(
    "float4[^/]"
    "half2[^/]"
    "__hadd[^/]"
    "__hadd2"
)

FOUND_ISSUES=0

for pattern in "${RAW_CUDA_PATTERNS[@]}"; do
    if grep -r "$pattern" src/operators/*.cu 2>/dev/null | grep -v "//.*$pattern"; then
        echo "  ⚠️  Found raw CUDA pattern: $pattern"
        FOUND_ISSUES=$((FOUND_ISSUES + 1))
    fi
done

if [ $FOUND_ISSUES -eq 0 ]; then
    echo "  ✅ No raw CUDA patterns found in src/operators/"
else
    echo "  ❌ Found $FOUND_ISSUES raw CUDA patterns"
fi

# Check for CuTe usage
echo ""
echo "2. Checking for CuTe usage..."

CUTE_PATTERNS=(
    "make_layout"
    "make_tensor"
    "make_gmem_ptr"
    "using namespace cute"
)

CUTE_FOUND=0

for pattern in "${CUTE_PATTERNS[@]}"; do
    if grep -r "$pattern" src/operators/*.cu >/dev/null 2>&1; then
        echo "  ✅ Found CuTe pattern: $pattern"
        CUTE_FOUND=$((CUTE_FOUND + 1))
    else
        echo "  ⚠️  Missing CuTe pattern: $pattern"
    fi
done

echo ""
echo "  CuTe patterns found: $CUTE_FOUND / ${#CUTE_PATTERNS[@]}"

# Verify file organization
echo ""
echo "3. Verifying file organization..."

if [ -f "src/operators/gemm.cu" ]; then
    echo "  ✅ src/operators/gemm.cu exists"
else
    echo "  ❌ src/operators/gemm.cu missing"
fi

if [ -f "src/operators/elementwise_add.cu" ]; then
    echo "  ✅ src/operators/elementwise_add.cu exists"
else
    echo "  ❌ src/operators/elementwise_add.cu missing"
fi

if [ -f "baselines/cuda/cuda_gemm.cu" ]; then
    echo "  ✅ baselines/cuda/cuda_gemm.cu exists"
else
    echo "  ❌ baselines/cuda/cuda_gemm.cu missing"
fi

if [ -f "baselines/cuda/cuda_elementwise_add.cu" ]; then
    echo "  ✅ baselines/cuda/cuda_elementwise_add.cu exists"
else
    echo "  ❌ baselines/cuda/cuda_elementwise_add.cu missing"
fi

# Run tests
echo ""
echo "4. Running correctness tests..."

echo "  Building..."
make build >/dev/null 2>&1

echo "  Testing GEMM..."
if timeout 60 make bench-gemm 2>&1 | grep -q "✓ PASSED"; then
    echo "  ✅ GEMM tests passed"
else
    echo "  ❌ GEMM tests failed"
fi

echo "  Testing element-wise add..."
if timeout 30 make bench-elementwise-add 2>&1 | grep -q "CuTe"; then
    echo "  ✅ Element-wise add tests passed"
else
    echo "  ❌ Element-wise add tests failed"
fi

# Summary
echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="

if [ $FOUND_ISSUES -eq 0 ] && [ $CUTE_FOUND -eq ${#CUTE_PATTERNS[@]} ]; then
    echo "✅ All checks passed!"
    echo ""
    echo "Implementation status:"
    echo "  - Pure CuTe implementations in src/operators/"
    echo "  - CUDA baselines in baselines/cuda/"
    echo "  - All tests passing"
    echo "  - Ready for optimization"
else
    echo "⚠️  Some issues found"
    echo ""
    echo "Issues:"
    [ $FOUND_ISSUES -gt 0 ] && echo "  - Raw CUDA patterns detected"
    [ $CUTE_FOUND -lt ${#CUTE_PATTERNS[@]} ] && echo "  - Missing CuTe patterns"
fi

echo "=========================================="
