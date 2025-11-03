# CUDA Version Compatibility Fix

## Problem
The system has CUDA 13.0 installed as the default (`/usr/local/cuda` → `cuda-13.0`), but the NVIDIA driver only supports CUDA 12.8. This causes runtime errors:
```
CUDA driver version is insufficient for CUDA runtime version
```

## Automatic Solution (Implemented)
The Makefile now includes intelligent CUDA version detection and selection:

### How It Works
1. **Driver Detection**: Automatically detects your NVIDIA driver version using `nvidia-smi`
2. **Version Mapping**: Maps driver version to maximum supported CUDA version using official NVIDIA compatibility matrix
3. **Toolkit Discovery**: Scans for all available CUDA installations in standard locations
4. **Smart Selection**: Chooses the highest CUDA version that's compatible with your driver
5. **Fallback**: Falls back to traditional detection if automatic selection fails

### Driver-CUDA Compatibility Matrix
```
Driver Version  →  Max CUDA Version
580+           →  13.0
570+           →  12.8  ← Your system
560+           →  12.6
550+           →  12.4
535+           →  12.2
525+           →  12.0
515+           →  11.8
510+           →  11.6
495+           →  11.4
470+           →  11.2
460+           →  11.0
```

### Usage
Simply run make commands as normal - no manual configuration needed:
```bash
make build
make bench-all
make test
```

The Makefile will automatically output:
```
Detected driver supports CUDA version: 12.8
Selected compatible CUDA toolkit: /usr/local/cuda-12.8 (version 12.8)
```

### Manual Override (Optional)
You can still manually specify CUDA_PATH if needed:
```bash
export CUDA_PATH=/usr/local/cuda-12.8
make build
```

### Troubleshooting
If automatic detection fails, the Makefile will show:
- Your driver's supported CUDA version
- All available CUDA installations
- Suggestions for resolution

## Additional Fixes Applied
1. **Deprecated API**: Fixed `benchmarks/bench_elementwise_add.cu` to use `cudaDeviceGetAttribute()` instead of deprecated `cudaDeviceProp.memoryClockRate`
2. **Smart Detection**: Enhanced Makefile with automatic CUDA version compatibility checking
3. **Forward Compatibility**: Added `-cudart shared` flag for better runtime compatibility

## Verification
The system now works automatically:
```bash
make bench-all  # No manual CUDA_PATH needed!
```