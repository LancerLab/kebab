#!/bin/bash

# Test script to verify profiling fixes
set -e

echo "=========================================="
echo "Testing GEMM Profiling Fixes"
echo "=========================================="

# Check if NCU is available
if ! command -v ncu &> /dev/null; then
    echo "❌ NCU not found. Please install Nsight Compute."
    exit 1
fi

echo "✅ NCU found: $(which ncu)"

# Build the benchmark if needed
if [ ! -f "build/bench_gemm" ]; then
    echo "Building benchmark..."
    make bench-gemm
fi

echo "✅ Benchmark built: build/bench_gemm"

# Clean old profiling data
rm -rf profiling/gemm_*

# Run a quick profiling test with minimal metrics
echo "Running quick profiling test..."
ncu \
    --section LaunchStats \
    --section Occupancy \
    --metrics achieved_occupancy \
    --metrics theoretical_occupancy \
    --export profiling/test_gemm_profile \
    --force-overwrite \
    --target-processes all \
    --kernel-regex ".*gemm.*|.*Gemm.*|.*GEMM.*" \
    --timeout 30 \
    build/bench_gemm || {
        echo "⚠️  Profiling completed with warnings (this may be normal)"
    }

# Check if profile was generated
if [ -f "profiling/test_gemm_profile.ncu-rep" ]; then
    echo "✅ Profile generated: profiling/test_gemm_profile.ncu-rep"
    
    # Generate summary
    echo "Generating summary..."
    ncu --import profiling/test_gemm_profile.ncu-rep \
        --page LaunchStats \
        > profiling/test_summary.txt 2>&1 || true
    
    ncu --import profiling/test_gemm_profile.ncu-rep \
        --page Occupancy \
        >> profiling/test_summary.txt 2>&1 || true
    
    if [ -f "profiling/test_summary.txt" ]; then
        echo "✅ Summary generated: profiling/test_summary.txt"
        echo ""
        echo "Preview of launch statistics and occupancy:"
        echo "----------------------------------------"
        head -30 profiling/test_summary.txt | grep -E "(Kernel|Launch|Block|Grid|Occupancy|Duration|Registers|Shared)" || true
        echo "----------------------------------------"
        
        # Check if we have actual data (not just headers)
        if grep -q "Block Size" profiling/test_summary.txt; then
            echo "✅ Launch statistics captured successfully!"
        else
            echo "⚠️  Launch statistics may be empty"
        fi
        
        if grep -q "Occupancy" profiling/test_summary.txt; then
            echo "✅ Occupancy data captured successfully!"
        else
            echo "⚠️  Occupancy data may be empty"
        fi
    else
        echo "❌ Failed to generate summary"
    fi
else
    echo "❌ Profile not generated"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test completed! Now try: make tune-gemm"
echo "=========================================="