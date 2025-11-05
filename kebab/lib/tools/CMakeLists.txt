# Development tools

# Install shell scripts
install(FILES
    diagnose_profiling.sh
    profile_gemm.sh
    test_cuda_auto_detection.sh
    test_cuda_detection.sh
    test_profiling_fix.sh
    test_simple_profiling.sh
    verify_cute_implementation.sh
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install Python scripts
install(FILES
    generate_report.py
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)