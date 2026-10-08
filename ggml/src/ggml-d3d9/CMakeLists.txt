cmake_minimum_required(VERSION 3.13)

# SM 3.0 pixel shaders, compiled at runtime (ps_3_0) with d3dcompiler_47.dll; no shader files to embed
ggml_add_backend_library(ggml-d3d9
    ggml-d3d9.cpp
    ../../include/ggml-d3d9.h
)

target_compile_features(ggml-d3d9 PRIVATE cxx_std_17)
target_link_libraries(ggml-d3d9 PRIVATE d3d9 d3dcompiler_47)
