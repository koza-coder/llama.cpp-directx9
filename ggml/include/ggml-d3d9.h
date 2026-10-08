#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_D3D9_NAME "D3D9"

GGML_BACKEND_API ggml_backend_t ggml_backend_d3d9_init(int device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_d3d9_reg(void);

#ifdef  __cplusplus
}
#endif
