#!/bin/bash

# GEMM Profiling Script with Enhanced Metrics Collection
# This script ensures we capture launch statistics, occupancy, and detailed metrics

set -e

# Configuration
BUILD_DIR=${BUILD_DIR:-build}
PROFILING_DIR=${PROFILING_DIR:-profiling}
NCU=${NCU:-ncu}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Enhanced GEMM Profiling Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if NCU is available
if ! command -v "$NCU" &> /dev/null; then
    echo -e "${RED}ERROR: Nsight Compute (ncu) not found${NC}"
    echo "Please ensure NCU is in your PATH or set NCU environment variable"
    exit 1
fi

# Check if benchmark executable exists
if [ ! -f "$BUILD_DIR/bench_gemm" ]; then
    echo -e "${RED}ERROR: $BUILD_DIR/bench_gemm not found${NC}"
    echo "Please build the benchmark first: make bench-gemm"
    exit 1
fi

# Create profiling directory
mkdir -p "$PROFILING_DIR"

echo -e "${GREEN}Using NCU: $NCU${NC}"
echo -e "${GREEN}Benchmark: $BUILD_DIR/bench_gemm${NC}"
echo -e "${GREEN}Output dir: $PROFILING_DIR${NC}"
echo ""

# Step 1: Run comprehensive profiling with all sections
echo -e "${YELLOW}Step 1: Running comprehensive profiling...${NC}"
echo "This will capture launch statistics, occupancy, and performance metrics"
echo ""

$NCU \
    --set full \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
    --metrics sm__sass_thread_inst_executed_op_fmul_pred_on.sum \
    --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
    --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
    --metrics sm__inst_executed_pipe_tensor.sum \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    --metrics achieved_occupancy \
    --metrics theoretical_occupancy \
    --export "$PROFILING_DIR/gemm_profile" \
    --force-overwrite \
    --target-processes all \
    --kernel-regex ".*gemm.*|.*Gemm.*|.*GEMM.*|.*sgemm.*|.*hgemm.*" \
    --print-gpu-trace \
    "$BUILD_DIR/bench_gemm" || {
        echo -e "${YELLOW}Warning: Profiling completed with some errors (this may be normal)${NC}"
        echo "Profile data should still be available"
    }

echo ""

# Step 2: Generate detailed reports
if [ -f "$PROFILING_DIR/gemm_profile.ncu-rep" ]; then
    echo -e "${YELLOW}Step 2: Generating detailed analysis reports...${NC}"
    
    # Generate summary with launch stats and occupancy
    echo "Generating launch statistics and occupancy summary..."
    $NCU --import "$PROFILING_DIR/gemm_profile.ncu-rep" \
        --page details \
        --print-summary per-kernel \
        > "$PROFILING_DIR/gemm_summary.txt" 2>&1 || true
    
    # Append launch statistics
    echo "" >> "$PROFILING_DIR/gemm_summary.txt"
    echo "=== LAUNCH STATISTICS ===" >> "$PROFILING_DIR/gemm_summary.txt"
    $NCU --import "$PROFILING_DIR/gemm_profile.ncu-rep" \
        --page LaunchStats \
        >> "$PROFILING_DIR/gemm_summary.txt" 2>&1 || true
    
    # Append occupancy analysis
    echo "" >> "$PROFILING_DIR/gemm_summary.txt"
    echo "=== OCCUPANCY ANALYSIS ===" >> "$PROFILING_DIR/gemm_summary.txt"
    $NCU --import "$PROFILING_DIR/gemm_profile.ncu-rep" \
        --page Occupancy \
        >> "$PROFILING_DIR/gemm_summary.txt" 2>&1 || true
    
    # Generate CSV metrics
    echo "Generating CSV metrics report..."
    $NCU --import "$PROFILING_DIR/gemm_profile.ncu-rep" \
        --page raw \
        --csv \
        > "$PROFILING_DIR/gemm_metrics.csv" 2>&1 || true
    
    # Generate GPU trace
    echo "Generating GPU trace summary..."
    $NCU --import "$PROFILING_DIR/gemm_profile.ncu-rep" \
        --page SpeedOfLight \
        > "$PROFILING_DIR/gemm_speedoflight.txt" 2>&1 || true
    
    echo -e "${GREEN}Step 3: Analysis complete!${NC}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Profiling Results${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Results saved to:${NC}"
    echo "  - NCU Report:         $PROFILING_DIR/gemm_profile.ncu-rep"
    echo "  - Summary & Stats:    $PROFILING_DIR/gemm_summary.txt"
    echo "  - Metrics CSV:        $PROFILING_DIR/gemm_metrics.csv"
    echo "  - Speed of Light:     $PROFILING_DIR/gemm_speedoflight.txt"
    echo ""
    echo -e "${GREEN}To view in Nsight Compute GUI:${NC}"
    echo "  ncu-ui $PROFILING_DIR/gemm_profile.ncu-rep"
    echo ""
    
    # Show quick summary if available
    if [ -f "$PROFILING_DIR/gemm_summary.txt" ]; then
        echo -e "${YELLOW}Quick Summary Preview:${NC}"
        echo "----------------------------------------"
        head -20 "$PROFILING_DIR/gemm_summary.txt" | grep -E "(Kernel|Launch|Block|Grid|Occupancy|Duration)" || true
        echo "----------------------------------------"
        echo "(See full details in $PROFILING_DIR/gemm_summary.txt)"
    fi
    
else
    echo -e "${RED}ERROR: Profile report not generated${NC}"
    echo "Check if the benchmark ran successfully and NCU has proper permissions"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"