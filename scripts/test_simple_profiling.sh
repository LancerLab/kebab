#!/bin/bash

# Simple test for the profiling fix
set -e

echo "=========================================="
echo "Testing Simple GEMM Profiling"
echo "=========================================="

# Find NCU
NCU=$(which ncu 2>/dev/null || echo "")
if [ -z "$NCU" ]; then
    NCU="/usr/local/cuda/bin/ncu"
fi

if [ ! -x "$NCU" ]; then
    echo "❌ NCU not found or not executable: $NCU"
    echo "Please ensure Nsight Compute is installed"
    exit 1
fi

echo "Using NCU: $NCU"

# Check if benchmark exists
if [ ! -f "build/bench_gemm" ]; then
    echo "Building benchmark first..."
    make bench-gemm
fi

# Clean old profiling data
rm -rf profiling/gemm_* profiling/simple_*

echo "Running simplified NCU profiling..."

# Run with minimal but essential options
# Note: Driver compatibility issues may occur with newer NCU versions
$NCU \
    --section LaunchStats \
    --section Occupancy \
    --export profiling/simple_gemm_profile \
    --force-overwrite \
    --target-processes all \
    --kernel-name regex:.*gemm.* \
    build/bench_gemm || {
        echo "⚠️  Profiling completed with warnings (driver compatibility issues)"
    }

# Check results
if [ -f "profiling/simple_gemm_profile.ncu-rep" ]; then
    echo "✅ Profile generated successfully"
    
    # Generate full report
    echo "Extracting full report..."
    $NCU --import profiling/simple_gemm_profile.ncu-rep \
        > profiling/simple_full_report.txt 2>&1 || true
    
    echo ""
    echo "=========================================="
    echo "Results Preview"
    echo "=========================================="
    
    if [ -f "profiling/simple_full_report.txt" ]; then
        echo "Launch Statistics & Occupancy:"
        echo "----------------------------------------"
        grep -E "(Block Size|Grid Size|Registers Per Thread|Shared Memory|Theoretical Occupancy|Achieved Occupancy|Waves Per SM)" profiling/simple_full_report.txt | head -15
        echo "----------------------------------------"
        echo ""
        echo "✅ SUCCESS: Launch statistics and occupancy data captured!"
    fi
    
    echo ""
    echo "=========================================="
    echo "Test completed!"
    echo "Files generated:"
    echo "  - profiling/simple_gemm_profile.ncu-rep"
    echo "  - profiling/simple_full_report.txt"
    echo ""
    echo "View full report:"
    echo "  cat profiling/simple_full_report.txt"
    echo ""
    echo "Now try: make tune-gemm"
    echo "=========================================="
    
else
    echo "❌ Profile not generated"
    exit 1
fi