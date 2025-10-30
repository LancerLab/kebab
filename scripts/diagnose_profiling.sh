#!/bin/bash

# Diagnostic script for profiling issues
set -e

echo "=========================================="
echo "GEMM Profiling Diagnostic Tool"
echo "=========================================="

# Check NCU version and capabilities
echo "1. Checking NCU installation..."
if command -v ncu &> /dev/null; then
    echo "✅ NCU found: $(which ncu)"
    echo "   Version: $(ncu --version 2>/dev/null || echo 'Version check failed')"
else
    echo "❌ NCU not found"
    exit 1
fi

# Check CUDA driver and runtime
echo ""
echo "2. Checking CUDA environment..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader,nounits | head -1
else
    echo "❌ nvidia-smi not found"
fi

# Check if benchmark exists and runs
echo ""
echo "3. Checking benchmark executable..."
if [ -f "build/bench_gemm" ]; then
    echo "✅ Benchmark found: build/bench_gemm"
    
    # Try to run benchmark briefly to see if it works
    echo "   Testing benchmark execution..."
    timeout 10s build/bench_gemm > /tmp/bench_test.log 2>&1 || {
        echo "⚠️  Benchmark execution test completed (may have timed out)"
    }
    
    if [ -f "/tmp/bench_test.log" ]; then
        if grep -q "GEMM Benchmark" /tmp/bench_test.log; then
            echo "✅ Benchmark runs successfully"
        else
            echo "⚠️  Benchmark may have issues"
            echo "   First few lines of output:"
            head -5 /tmp/bench_test.log
        fi
    fi
else
    echo "❌ Benchmark not found. Run: make bench-gemm"
    exit 1
fi

# Check kernel names in the binary
echo ""
echo "4. Checking kernel names in binary..."
if command -v cuobjdump &> /dev/null; then
    echo "   Kernel functions found:"
    cuobjdump --dump-ptx build/bench_gemm 2>/dev/null | grep -E "\.entry|\.func" | head -10 || {
        echo "   Could not extract kernel names with cuobjdump"
    }
elif command -v objdump &> /dev/null; then
    echo "   Searching for GEMM-related symbols:"
    objdump -t build/bench_gemm 2>/dev/null | grep -i gemm | head -5 || {
        echo "   No GEMM symbols found with objdump"
    }
else
    echo "   No binary analysis tools available"
fi

# Test minimal profiling
echo ""
echo "5. Testing minimal profiling..."
mkdir -p profiling

echo "   Running minimal NCU test..."
ncu \
    --metrics achieved_occupancy \
    --export profiling/diagnostic_test \
    --force-overwrite \
    --target-processes all \
    --timeout 15 \
    build/bench_gemm > /tmp/ncu_test.log 2>&1 || {
        echo "⚠️  NCU test completed with warnings"
    }

if [ -f "profiling/diagnostic_test.ncu-rep" ]; then
    echo "✅ NCU can generate reports"
    
    # Try to extract basic info
    echo "   Extracting kernel information..."
    ncu --import profiling/diagnostic_test.ncu-rep \
        --page raw \
        --csv 2>/dev/null | head -2 > /tmp/kernel_info.csv || true
    
    if [ -f "/tmp/kernel_info.csv" ] && [ -s "/tmp/kernel_info.csv" ]; then
        echo "✅ Kernel data captured"
        echo "   Kernel info preview:"
        cat /tmp/kernel_info.csv | cut -d',' -f1-5
    else
        echo "⚠️  No kernel data in report"
    fi
    
    # Check for launch stats
    ncu --import profiling/diagnostic_test.ncu-rep \
        --page LaunchStats > /tmp/launch_stats.txt 2>&1 || true
    
    if [ -f "/tmp/launch_stats.txt" ] && grep -q "Block Size\|Grid Size\|Kernel" /tmp/launch_stats.txt; then
        echo "✅ Launch statistics available"
    else
        echo "⚠️  Launch statistics empty or missing"
        if [ -f "/tmp/launch_stats.txt" ]; then
            echo "   Launch stats content:"
            head -10 /tmp/launch_stats.txt
        fi
    fi
else
    echo "❌ NCU failed to generate report"
    if [ -f "/tmp/ncu_test.log" ]; then
        echo "   NCU error log:"
        cat /tmp/ncu_test.log
    fi
fi

# Check permissions
echo ""
echo "6. Checking permissions..."
if [ -w "profiling" ]; then
    echo "✅ Can write to profiling directory"
else
    echo "❌ Cannot write to profiling directory"
fi

# Summary and recommendations
echo ""
echo "=========================================="
echo "Diagnostic Summary & Recommendations"
echo "=========================================="

if [ -f "profiling/diagnostic_test.ncu-rep" ]; then
    echo "✅ Basic profiling works"
    
    if grep -q "Block Size\|Grid Size" /tmp/launch_stats.txt 2>/dev/null; then
        echo "✅ Launch statistics are captured"
        echo ""
        echo "🎉 Profiling should work correctly!"
        echo "   Try: make tune-gemm"
    else
        echo "⚠️  Launch statistics are empty"
        echo ""
        echo "💡 Possible solutions:"
        echo "   1. The benchmark may be running too quickly"
        echo "   2. Try with larger matrix sizes in config.yaml"
        echo "   3. Use --kernel-name instead of --kernel-regex"
        echo "   4. Check if kernels are actually being launched"
    fi
else
    echo "❌ Basic profiling failed"
    echo ""
    echo "💡 Possible solutions:"
    echo "   1. Check CUDA driver compatibility with NCU"
    echo "   2. Try running as root (sudo make tune-gemm)"
    echo "   3. Check if GPU is accessible"
    echo "   4. Update CUDA toolkit and NCU"
fi

echo ""
echo "For more help, check:"
echo "  - NVIDIA Nsight Compute documentation"
echo "  - CUDA compatibility matrix"
echo "  - GPU driver version requirements"

# Cleanup
rm -f /tmp/bench_test.log /tmp/ncu_test.log /tmp/kernel_info.csv /tmp/launch_stats.txt