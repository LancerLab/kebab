#!/usr/bin/env python3
"""
CuTeKernelLib Benchmark Report Generator

This script parses CSV benchmark results and generates a comprehensive
Markdown summary report with performance analysis.

Usage:
    python3 scripts/generate_report.py bench_results/*.csv > bench_results/summary.md
    python3 scripts/generate_report.py bench_results/elementwise_add_results.csv
"""

import sys
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse


class BenchmarkResult:
    """Represents a single benchmark result entry"""
    
    def __init__(self, operator: str, variant: str, batch_size: int, 
                 latency_ms: float, throughput: float, speedup: float):
        self.operator = operator
        self.variant = variant
        self.batch_size = batch_size
        self.latency_ms = latency_ms
        self.throughput = throughput
        self.speedup = speedup


class ReportGenerator:
    """Generates Markdown benchmark reports from CSV data"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.operators: set = set()
        
    def parse_csv_file(self, csv_path: str) -> None:
        """Parse a single CSV file and add results"""
        try:
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Handle different possible column names
                    operator = row.get('Operator', '').strip()
                    variant = row.get('Variant', '').strip()
                    
                    # Parse batch size
                    batch_size_str = row.get('BatchSize', '0')
                    try:
                        batch_size = int(batch_size_str)
                    except ValueError:
                        print(f"Warning: Invalid batch size '{batch_size_str}' in {csv_path}", 
                              file=sys.stderr)
                        continue
                    
                    # Parse latency
                    latency_str = row.get('Latency(ms)', '0')
                    try:
                        latency_ms = float(latency_str)
                    except ValueError:
                        print(f"Warning: Invalid latency '{latency_str}' in {csv_path}", 
                              file=sys.stderr)
                        continue
                    
                    # Parse throughput (could be GB/s or GFLOPS)
                    throughput_str = row.get('Throughput(GB/s)', 
                                           row.get('Throughput(GFLOPS)', '0'))
                    try:
                        throughput = float(throughput_str)
                    except ValueError:
                        print(f"Warning: Invalid throughput '{throughput_str}' in {csv_path}", 
                              file=sys.stderr)
                        continue
                    
                    # Parse speedup
                    speedup_str = row.get('Speedup', '1.0')
                    try:
                        speedup = float(speedup_str)
                    except ValueError:
                        print(f"Warning: Invalid speedup '{speedup_str}' in {csv_path}", 
                              file=sys.stderr)
                        continue
                    
                    # Skip empty rows
                    if not operator or not variant:
                        continue
                    
                    result = BenchmarkResult(operator, variant, batch_size, 
                                           latency_ms, throughput, speedup)
                    self.results.append(result)
                    self.operators.add(operator)
                    
        except FileNotFoundError:
            print(f"Error: File not found: {csv_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error parsing {csv_path}: {e}", file=sys.stderr)
    
    def parse_csv_files(self, csv_paths: List[str]) -> None:
        """Parse multiple CSV files"""
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                self.parse_csv_file(csv_path)
            else:
                print(f"Warning: File not found: {csv_path}", file=sys.stderr)
    
    def get_gpu_info(self) -> str:
        """Attempt to get GPU information"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_name, compute_cap = lines[0].split(', ')
                    return f"{gpu_name.strip()} (sm_{compute_cap.replace('.', '')})"
        except Exception:
            pass
        return "Unknown GPU"
    
    def generate_operator_table(self, operator: str) -> str:
        """Generate Markdown table for a specific operator"""
        # Filter results for this operator
        op_results = [r for r in self.results if r.operator == operator]
        if not op_results:
            return f"No results found for operator: {operator}\n"
        
        # Sort by batch size, then by variant (CUDA first for baseline)
        op_results.sort(key=lambda x: (x.batch_size, x.variant != "CUDA"))
        
        # Determine throughput unit
        throughput_unit = "GB/s"
        if operator.lower() in ["gemm", "conv2d"]:
            throughput_unit = "GFLOPS"
        
        # Generate table
        table = f"## {operator.title()} Performance\n\n"
        table += f"| Batch Size | Variant | Latency (ms) | Throughput ({throughput_unit}) | Speedup |\n"
        table += "|------------|---------|--------------|---------------------|----------|\n"
        
        for result in op_results:
            table += f"| {result.batch_size:,} | {result.variant} | "
            table += f"{result.latency_ms:.3f} | {result.throughput:.1f} | "
            table += f"{result.speedup:.2f}x |\n"
        
        # Add performance analysis
        cute_results = [r for r in op_results if r.variant == "CuTe"]
        cuda_results = [r for r in op_results if r.variant == "CUDA"]
        
        if cute_results and cuda_results:
            avg_speedup = sum(r.speedup for r in cute_results) / len(cute_results)
            max_speedup = max(r.speedup for r in cute_results)
            min_speedup = min(r.speedup for r in cute_results)
            
            table += f"\n**Performance Summary:**\n"
            table += f"- Average CuTe speedup: {avg_speedup:.2f}x\n"
            table += f"- Best CuTe speedup: {max_speedup:.2f}x\n"
            table += f"- Worst CuTe speedup: {min_speedup:.2f}x\n"
            
            if avg_speedup >= 1.0:
                table += f"- ✅ CuTe implementation meets or exceeds baseline performance\n"
            else:
                table += f"- ⚠️ CuTe implementation underperforms baseline by {(1.0/avg_speedup - 1)*100:.1f}%\n"
        
        table += "\n"
        return table
    
    def generate_summary_table(self) -> str:
        """Generate overall summary table"""
        if not self.results:
            return "No benchmark results found.\n"
        
        summary = "## Overall Performance Summary\n\n"
        summary += "| Operator | Best CuTe Speedup | Avg CuTe Speedup | Status |\n"
        summary += "|----------|-------------------|------------------|--------|\n"
        
        for operator in sorted(self.operators):
            cute_results = [r for r in self.results 
                           if r.operator == operator and r.variant == "CuTe"]
            
            if cute_results:
                speedups = [r.speedup for r in cute_results]
                avg_speedup = sum(speedups) / len(speedups)
                max_speedup = max(speedups)
                
                status = "✅ Good" if avg_speedup >= 1.0 else "⚠️ Needs optimization"
                
                summary += f"| {operator} | {max_speedup:.2f}x | {avg_speedup:.2f}x | {status} |\n"
        
        summary += "\n"
        return summary
    
    def generate_report(self) -> str:
        """Generate complete Markdown report"""
        if not self.results:
            return "# CuTeKernelLib Benchmark Results\n\nNo benchmark data found.\n"
        
        # Header
        report = "# CuTeKernelLib Benchmark Results\n\n"
        
        # Metadata
        gpu_info = self.get_gpu_info()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report += f"**GPU**: {gpu_info}\n"
        report += f"**Generated**: {timestamp}\n"
        report += f"**Operators tested**: {len(self.operators)}\n"
        report += f"**Total benchmarks**: {len(self.results)}\n\n"
        
        # Overall summary
        report += self.generate_summary_table()
        
        # Individual operator tables
        for operator in sorted(self.operators):
            report += self.generate_operator_table(operator)
        
        # Footer with methodology
        report += "---\n\n"
        report += "## Methodology\n\n"
        report += "- **Latency**: Average execution time measured with CUDA events\n"
        report += "- **Throughput**: Memory bandwidth (GB/s) or compute throughput (GFLOPS)\n"
        report += "- **Speedup**: CuTe performance relative to CUDA baseline (higher is better)\n"
        report += "- **Baseline**: Hand-optimized CUDA implementation for comparison\n\n"
        
        report += "*Generated by CuTeKernelLib benchmark report generator*\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark report from CSV files')
    parser.add_argument('csv_files', nargs='+', help='CSV files to process')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    csv_files = []
    for pattern in args.csv_files:
        if '*' in pattern:
            from glob import glob
            csv_files.extend(glob(pattern))
        else:
            csv_files.append(pattern)
    
    if args.verbose:
        print(f"Processing {len(csv_files)} CSV files:", file=sys.stderr)
        for f in csv_files:
            print(f"  - {f}", file=sys.stderr)
    
    # Generate report
    generator = ReportGenerator()
    generator.parse_csv_files(csv_files)
    
    if args.verbose:
        print(f"Parsed {len(generator.results)} benchmark results", file=sys.stderr)
        print(f"Found {len(generator.operators)} operators", file=sys.stderr)
    
    report = generator.generate_report()
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        if args.verbose:
            print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()