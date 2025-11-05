# Example programs

# Simple GEMM example
add_executable(simple_gemm
    simple_gemm.cu
)
target_include_directories(simple_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(simple_gemm kebab)