# Testing Guide

## Running Tests

To run all tests:
```bash
make test
```

This will run the GEMM benchmark with verification enabled, which tests all supported modes (RR, RC, CR, CC).

## Test Status

### Current Tests
- **GEMM Verification**: Integrated into `make bench-gemm`
  - Tests all storage format combinations (RR, RC, CR, CC)
  - Verifies results against cuBLAS reference
  - Supports verbose mode for detailed matrix inspection

### Removed Tests
The following legacy test files have been removed as they were incompatible with the current RC/CR interface:
- `tests/test_gemm_simple.cu` - Used old TN/NT interface
- `tests/test_gemm_dispatch.cu` - Used old TN/NT interface
- `tests/test_init_methods.cu` - Used old TN/NT interface
- `tests/debug_cublas.cu` - Debug file no longer needed

## Verification

The benchmark system includes built-in verification:
- Each GEMM operation is verified against cuBLAS
- Results are checked with configurable tolerance
- Verbose mode shows detailed matrix comparisons

To enable verbose mode for detailed verification:
```yaml
# In config.yaml
operators:
  gemm:
    verbose: true
```

## Test Coverage

Current test coverage includes:
- ✅ RR mode (row-major × row-major) - **WORKING**
- ✅ RC mode (row-major × column-major) - **WORKING**
- ✅ CR mode (column-major × row-major) - **WORKING**
- ✅ CC mode (column-major × column-major) - **WORKING**
- ✅ Multiple matrix sizes (128, 256, 512, 1024, 2048)
- ✅ FP16 precision
- ✅ Verification against cuBLAS reference

## Adding New Tests

To add new tests, create test files in the `tests/` directory and add corresponding targets in the Makefile. Ensure tests use the current RC/CR interface format.
