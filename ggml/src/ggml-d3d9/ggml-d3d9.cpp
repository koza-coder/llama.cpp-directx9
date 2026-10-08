// Direct3D 9 backend: a mat-mul accelerator in the style of ggml-blas.
// The CPU buffer type is used (is_host), only MUL_MAT is offloaded; every other op stays on the CPU.
// Compute model: SM 3.0 pixel shaders, one output pixel = 4 outputs, inputs are float textures, split-K plus a reduce pass.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>
#include <d3dcompiler.h>

#include "ggml-d3d9.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>


static bool d3d9_debug() {
    static const bool v = getenv("GGML_D3D9_DEBUG") != nullptr;
    return v;
}

// ---------------------------------------------------------------- shaders

static const char * ps_vs_src = "float4 main(float4 p : POSITION) : POSITION { return p; }\n";

// c0.xy = 1/size of the weight texture, c0.zw = 1/size of the activation texture
static const char * ps_common =
    "sampler2D sW : register(s0);\n"
    "sampler2D sX : register(s1);\n"
    "float4 uInv : register(c0);\n"
    "float4 T(sampler2D s, float x, float y, float2 inv) {\n"
    "    return tex2Dlod(s, float4((x + 0.5) * inv.x, (y + 0.5) * inv.y, 0, 0));\n"
    "}\n";

// pixel (m4 * SPLIT + s, r) = outputs (r, 4*m4 .. 4*m4+3) summed over texel chunk s of the K axis
static const char * ps_matmul =
    "float4 main(float2 vpos : VPOS) : COLOR {\n"
    "    float2 p = floor(vpos + 0.25);\n"
    "    float m4 = floor((p.x + 0.5) / SPLIT);\n"
    "    float base = (p.x - m4 * SPLIT) * CHUNK;\n"
    "    float c = m4 * 4;\n"
    "    float4 acc = 0;\n"
    "    [loop] for (int o = 0; o < OUTER; o++) {\n"
    "        [loop] for (int i = 0; i < INNER; i++) {\n"
    "            float t = base + o * INNER + i;\n"
    "            float4 w = T(sW, t, p.y, uInv.xy);\n"
    "            acc += float4(dot(w, T(sX, t, c,     uInv.zw)), dot(w, T(sX, t, c + 1, uInv.zw)),\n"
    "                          dot(w, T(sX, t, c + 2, uInv.zw)), dot(w, T(sX, t, c + 3, uInv.zw)));\n"
    "        }\n"
    "    }\n"
    "    return acc;\n"
    "}\n";

// pixel (x, r) = sum of the partials (x * SPLIT + s, r); the partials are bound at s0, c0.xy = 1/size
static const char * ps_reduce =
    "float4 main(float2 vpos : VPOS) : COLOR {\n"
    "    float2 p = floor(vpos + 0.25);\n"
    "    float4 acc = 0;\n"
    "    [loop] for (int s = 0; s < SPLIT; s++) acc += T(sW, p.x * SPLIT + s, p.y, uInv.xy);\n"
    "    return acc;\n"
    "}\n";

// ---------------------------------------------------------------- device context

struct ggml_d3d9_adapter {
    UINT        index;
    std::string name;
    int         max_dim;   // largest texture side the device allows (GGML_D3D9_MAX_DIM can lower it, for tests)
    bool        half_tex;  // A16B16G16R16F textures work: F16 and quantised weights are stored as half floats (GGML_D3D9_W16=0 turns it off)
};

struct ggml_d3d9_rt {
    IDirect3DTexture9 * tex  = nullptr;
    IDirect3DSurface9 * surf = nullptr;
};

struct ggml_d3d9_context {
    int                    device_index = 0;
    int                    max_dim = 4096;
    bool                   half_tex = false;
    IDirect3D9Ex *         d3d  = nullptr;
    IDirect3DDevice9Ex *   dev  = nullptr;
    HWND                   hwnd = nullptr;
    IDirect3DVertexShader9 *      vs   = nullptr;
    IDirect3DVertexDeclaration9 * decl = nullptr;

    // key: split, inner, outer
    std::map<std::tuple<int, int, int>, IDirect3DPixelShader9 *> mm_ps;
    std::map<int, IDirect3DPixelShader9 *>                      red_ps;
    std::map<uint64_t, ggml_d3d9_rt>                             rts;      // render targets by (w, h)
    std::map<uint64_t, IDirect3DTexture9 *>                      stages;   // system-memory staging textures by (w, h)
    std::map<uint64_t, IDirect3DTexture9 *>                      stages16; // the same in A16B16G16R16F (weights)
    std::map<uint64_t, IDirect3DSurface9 *>                      readbacks;
    std::map<uint64_t, IDirect3DTexture9 *>                      xtex;     // activation textures by (w, h)
    std::map<std::tuple<const void *, size_t, int>, IDirect3DTexture9 *> weights;

    uint64_t n_calls = 0;
    uint64_t w_bytes = 0;   // video memory held by cached weight textures (printed with GGML_D3D9_STATS)
};

static std::vector<ggml_d3d9_adapter> & ggml_d3d9_adapters() {
    static std::vector<ggml_d3d9_adapter> v;
    static std::once_flag once;
    std::call_once(once, []() {
        if (getenv("GGML_D3D9_DISABLE")) {
            GGML_LOG_INFO("%s: GGML_D3D9_DISABLE set, no D3D9 devices\n", __func__);
            return;
        }
        IDirect3D9Ex * d3d = nullptr;
        if (FAILED(Direct3DCreate9Ex(D3D_SDK_VERSION, &d3d))) {
            GGML_LOG_INFO("%s: Direct3DCreate9Ex failed (no interactive session?), no D3D9 devices\n", __func__);
            return;
        }
        UINT n = d3d->GetAdapterCount();
        for (UINT a = 0; a < n; a++) {
            D3DADAPTER_IDENTIFIER9 id = {};
            D3DCAPS9 caps = {};
            if (FAILED(d3d->GetAdapterIdentifier(a, 0, &id)) || FAILED(d3d->GetDeviceCaps(a, D3DDEVTYPE_HAL, &caps))) {
                continue;
            }
            if (d3d9_debug()) {
                // GGML_D3D9_DEBUG=1: caps of every adapter, so design choices per GPU come from data
                GGML_LOG_INFO("D3D9 CAPS adapter %u: %s | vendor 0x%04lx device 0x%04lx driver %u.%u.%u.%u\n", a, id.Description,
                              (unsigned long) id.VendorId, (unsigned long) id.DeviceId,
                              (unsigned) HIWORD(id.DriverVersion.HighPart), (unsigned) LOWORD(id.DriverVersion.HighPart),
                              (unsigned) HIWORD(id.DriverVersion.LowPart), (unsigned) LOWORD(id.DriverVersion.LowPart));
                GGML_LOG_INFO("D3D9 CAPS adapter %u: ps %lu.%lu vs %lu.%lu maxtex %lux%lu rts %lu ps30_slots %lu ps_instr_exec %lu ps_temps %d ps_static_flow %d ps_dynamic_flow %d\n",
                              a, (unsigned long) D3DSHADER_VERSION_MAJOR(caps.PixelShaderVersion), (unsigned long) D3DSHADER_VERSION_MINOR(caps.PixelShaderVersion),
                              (unsigned long) D3DSHADER_VERSION_MAJOR(caps.VertexShaderVersion), (unsigned long) D3DSHADER_VERSION_MINOR(caps.VertexShaderVersion),
                              (unsigned long) caps.MaxTextureWidth, (unsigned long) caps.MaxTextureHeight, (unsigned long) caps.NumSimultaneousRTs,
                              (unsigned long) caps.MaxPixelShader30InstructionSlots, (unsigned long) caps.MaxPShaderInstructionsExecuted,
                              caps.PS20Caps.NumTemps, caps.PS20Caps.StaticFlowControlDepth, caps.PS20Caps.DynamicFlowControlDepth);
                static const struct { D3DFORMAT f; const char * n; } fmts[] = {
                    { D3DFMT_A32B32G32R32F, "A32B32G32R32F" }, { D3DFMT_R32F, "R32F" }, { D3DFMT_G32R32F, "G32R32F" },
                    { D3DFMT_A16B16G16R16F, "A16B16G16R16F" }, { D3DFMT_R16F, "R16F" }, { D3DFMT_L8, "L8" }, { D3DFMT_A8R8G8B8, "A8R8G8B8" },
                };
                for (const auto & f : fmts) {
                    const bool tex = SUCCEEDED(d3d->CheckDeviceFormat(a, D3DDEVTYPE_HAL, D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, f.f));
                    const bool rt  = SUCCEEDED(d3d->CheckDeviceFormat(a, D3DDEVTYPE_HAL, D3DFMT_X8R8G8B8, D3DUSAGE_RENDERTARGET, D3DRTYPE_TEXTURE, f.f));
                    const bool vtf = SUCCEEDED(d3d->CheckDeviceFormat(a, D3DDEVTYPE_HAL, D3DFMT_X8R8G8B8, D3DUSAGE_QUERY_VERTEXTEXTURE, D3DRTYPE_TEXTURE, f.f));
                    GGML_LOG_INFO("D3D9 CAPS adapter %u: format %-14s texture %d rendertarget %d vertextexture %d\n", a, f.n, (int) tex, (int) rt, (int) vtf);
                }
            }
            // needs ps_3_0 and float render targets
            if (caps.PixelShaderVersion < D3DPS_VERSION(3, 0)) {
                continue;
            }
            if (FAILED(d3d->CheckDeviceFormat(a, D3DDEVTYPE_HAL, D3DFMT_X8R8G8B8, D3DUSAGE_RENDERTARGET, D3DRTYPE_TEXTURE, D3DFMT_A32B32G32R32F))) {
                continue;
            }
            int max_dim = (int) std::min<DWORD>(caps.MaxTextureWidth, caps.MaxTextureHeight);
            // GGML_D3D9_MAX_DIM=<n> (tests only): lower the limit to exercise the row-chunk path
            if (getenv("GGML_D3D9_MAX_DIM") && atoi(getenv("GGML_D3D9_MAX_DIM")) >= 64) {
                max_dim = std::min(max_dim, atoi(getenv("GGML_D3D9_MAX_DIM")));
            }
            const bool half_tex = !(getenv("GGML_D3D9_W16") && atoi(getenv("GGML_D3D9_W16")) == 0) &&
                SUCCEEDED(d3d->CheckDeviceFormat(a, D3DDEVTYPE_HAL, D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, D3DFMT_A16B16G16R16F));
            v.push_back({ a, id.Description, max_dim, half_tex });
        }
        d3d->Release();
    });
    return v;
}

static uint64_t key2(int w, int h) { return ((uint64_t) w << 32) | (uint32_t) h; }

static std::string def_int(const char * name, int v) {
    return std::string("#define ") + name + " " + std::to_string(v) + "\n";
}

static IDirect3DPixelShader9 * ggml_d3d9_make_ps(ggml_d3d9_context * ctx, const std::string & defs, const char * body) {
    std::string src = defs + ps_common + body;
    ID3DBlob * code = nullptr, * err = nullptr;
    HRESULT hr = D3DCompile(src.data(), src.size(), "ggml-d3d9", nullptr, nullptr, "main", "ps_3_0",
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    if (FAILED(hr)) {
        GGML_LOG_ERROR("%s: D3DCompile failed 0x%08lx: %s\n", __func__, (unsigned long) hr, err ? (const char *) err->GetBufferPointer() : "");
        if (err) err->Release();
        return nullptr;
    }
    if (err) err->Release();
    IDirect3DPixelShader9 * ps = nullptr;
    hr = ctx->dev->CreatePixelShader((const DWORD *) code->GetBufferPointer(), &ps);
    code->Release();
    if (FAILED(hr)) {
        GGML_LOG_ERROR("%s: CreatePixelShader failed 0x%08lx\n", __func__, (unsigned long) hr);
        return nullptr;
    }
    return ps;
}

static bool ggml_d3d9_context_init(ggml_d3d9_context * ctx) {
    auto & ads = ggml_d3d9_adapters();
    if (ctx->device_index < 0 || (size_t) ctx->device_index >= ads.size()) {
        return false;
    }
    ctx->max_dim = ads[ctx->device_index].max_dim;
    ctx->half_tex = ads[ctx->device_index].half_tex;
    if (FAILED(Direct3DCreate9Ex(D3D_SDK_VERSION, &ctx->d3d))) {
        return false;
    }
    ctx->hwnd = CreateWindowA("STATIC", "ggml-d3d9", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, nullptr, nullptr, nullptr, nullptr);
    D3DPRESENT_PARAMETERS pp = {};
    pp.Windowed             = TRUE;
    pp.SwapEffect           = D3DSWAPEFFECT_DISCARD;
    pp.BackBufferWidth      = 1;
    pp.BackBufferHeight     = 1;
    pp.BackBufferFormat     = D3DFMT_UNKNOWN;
    pp.hDeviceWindow        = ctx->hwnd;
    DWORD flags = D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING;
    HRESULT hr = ctx->d3d->CreateDeviceEx(ads[ctx->device_index].index, D3DDEVTYPE_HAL, ctx->hwnd, flags, &pp, nullptr, &ctx->dev);
    if (FAILED(hr)) {
        GGML_LOG_ERROR("%s: CreateDeviceEx failed 0x%08lx\n", __func__, (unsigned long) hr);
        return false;
    }

    ID3DBlob * vcode = nullptr, * err = nullptr;
    hr = D3DCompile(ps_vs_src, strlen(ps_vs_src), "ggml-d3d9-vs", nullptr, nullptr, "main", "vs_3_0", 0, 0, &vcode, &err);
    if (err) err->Release();
    if (FAILED(hr)) {
        return false;
    }
    hr = ctx->dev->CreateVertexShader((const DWORD *) vcode->GetBufferPointer(), &ctx->vs);
    vcode->Release();
    const D3DVERTEXELEMENT9 el[] = { { 0, 0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 }, D3DDECL_END() };
    if (FAILED(hr) || FAILED(ctx->dev->CreateVertexDeclaration(el, &ctx->decl))) {
        return false;
    }
    ctx->dev->SetVertexShader(ctx->vs);
    ctx->dev->SetVertexDeclaration(ctx->decl);
    ctx->dev->SetRenderState(D3DRS_ZENABLE, FALSE);
    ctx->dev->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    ctx->dev->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    for (int i = 0; i < 2; i++) {
        ctx->dev->SetSamplerState(i, D3DSAMP_MINFILTER, D3DTEXF_POINT);
        ctx->dev->SetSamplerState(i, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
        ctx->dev->SetSamplerState(i, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        ctx->dev->SetSamplerState(i, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        ctx->dev->SetSamplerState(i, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
    }
    return true;
}

static void ggml_d3d9_context_free(ggml_d3d9_context * ctx) {
    for (auto & kv : ctx->mm_ps)    kv.second->Release();
    for (auto & kv : ctx->red_ps)   kv.second->Release();
    for (auto & kv : ctx->rts)      { kv.second.surf->Release(); kv.second.tex->Release(); }
    for (auto & kv : ctx->stages)   kv.second->Release();
    for (auto & kv : ctx->stages16) kv.second->Release();
    for (auto & kv : ctx->readbacks) kv.second->Release();
    for (auto & kv : ctx->xtex)     kv.second->Release();
    for (auto & kv : ctx->weights)  kv.second->Release();
    if (ctx->decl) ctx->decl->Release();
    if (ctx->vs)   ctx->vs->Release();
    if (ctx->dev)  ctx->dev->Release();
    if (ctx->d3d)  ctx->d3d->Release();
    if (ctx->hwnd) DestroyWindow(ctx->hwnd);
    delete ctx;
}

// ---------------------------------------------------------------- resources

static IDirect3DTexture9 * ggml_d3d9_stage(ggml_d3d9_context * ctx, int w, int h, bool half = false) {
    auto & m = half ? ctx->stages16 : ctx->stages;
    auto it = m.find(key2(w, h));
    if (it != m.end()) {
        return it->second;
    }
    IDirect3DTexture9 * t = nullptr;
    if (FAILED(ctx->dev->CreateTexture(w, h, 1, 0, half ? D3DFMT_A16B16G16R16F : D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM, &t, nullptr))) {
        return nullptr;
    }
    m[key2(w, h)] = t;
    return t;
}

// rows: h rows of w texels (16 bytes each); source row r is read through row_ptr(r), rows >= valid_rows are zero
template <typename F>
static bool ggml_d3d9_fill(ggml_d3d9_context * ctx, IDirect3DTexture9 * dst, int w, int h, int valid_rows, int row_floats, F row_ptr) {
    IDirect3DTexture9 * stage = ggml_d3d9_stage(ctx, w, h);
    if (!stage) {
        return false;
    }
    D3DLOCKED_RECT lr;
    if (FAILED(stage->LockRect(0, &lr, nullptr, 0))) {
        return false;
    }
    for (int y = 0; y < h; y++) {
        char * d = (char *) lr.pBits + (size_t) y * lr.Pitch;
        if (y < valid_rows) {
            memcpy(d, row_ptr(y), (size_t) row_floats * 4);
            memset(d + (size_t) row_floats * 4, 0, (size_t) w * 16 - (size_t) row_floats * 4);
        } else {
            memset(d, 0, (size_t) w * 16);
        }
    }
    stage->UnlockRect(0);
    return SUCCEEDED(ctx->dev->UpdateTexture(stage, dst));
}

static IDirect3DTexture9 * ggml_d3d9_new_tex(ggml_d3d9_context * ctx, int w, int h, bool half = false) {
    IDirect3DTexture9 * t = nullptr;
    if (FAILED(ctx->dev->CreateTexture(w, h, 1, 0, half ? D3DFMT_A16B16G16R16F : D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &t, nullptr))) {
        return nullptr;
    }
    return t;
}

static bool ggml_d3d9_rt_get(ggml_d3d9_context * ctx, int w, int h, ggml_d3d9_rt & out) {
    auto it = ctx->rts.find(key2(w, h));
    if (it != ctx->rts.end()) {
        out = it->second;
        return true;
    }
    ggml_d3d9_rt rt;
    if (FAILED(ctx->dev->CreateTexture(w, h, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &rt.tex, nullptr))) {
        return false;
    }
    rt.tex->GetSurfaceLevel(0, &rt.surf);
    ctx->rts[key2(w, h)] = rt;
    out = rt;
    return true;
}

static HRESULT ggml_d3d9_draw(ggml_d3d9_context * ctx, IDirect3DSurface9 * target, IDirect3DPixelShader9 * ps,
                              IDirect3DTexture9 * t0, IDirect3DTexture9 * t1, const float inv[4]) {
    static const float quad[16] = { -1, -1, 0, 1, -1, 1, 0, 1, 1, -1, 0, 1, 1, 1, 0, 1 };
    ctx->dev->SetRenderTarget(0, target);
    ctx->dev->SetTexture(0, t0);
    ctx->dev->SetTexture(1, t1);
    ctx->dev->SetPixelShader(ps);
    ctx->dev->SetPixelShaderConstantF(0, inv, 1);
    HRESULT hr = ctx->dev->BeginScene();
    if (SUCCEEDED(hr)) {
        hr = ctx->dev->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, 16);
        HRESULT hr_end = ctx->dev->EndScene();
        if (SUCCEEDED(hr)) {
            hr = hr_end;
        }
    }
    return hr;
}

// one clear line when a D3D9 call fails, so a TDR / lost device is not mistaken for a wrong result
static void ggml_d3d9_report(ggml_d3d9_context * ctx, const char * what, HRESULT hr, int K, int N, int M) {
    const HRESULT st = ctx->dev->CheckDeviceState(nullptr);
    GGML_LOG_ERROR("D3D9 %s failed 0x%08lx (device state 0x%08lx%s) during MUL_MAT K=%d N=%d M=%d\n", what, (unsigned long) hr, (unsigned long) st,
                   (hr == D3DERR_DEVICEHUNG || hr == D3DERR_DEVICEREMOVED || hr == D3DERR_DEVICELOST ||
                    st == D3DERR_DEVICEHUNG || st == D3DERR_DEVICEREMOVED || st == D3DERR_DEVICELOST) ? ", device hung/lost (TDR?)" : "", K, N, M);
}

// ---------------------------------------------------------------- mat-mul

static int ggml_d3d9_largest_divisor_le(int n, int cap) {
    for (int d = cap < n ? cap : n; d > 1; d--) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// split-K factor: enough pixels to fill the GPU, chunks of at least 8 texels, render target width within limits
static int ggml_d3d9_pick_split(int max_dim, int k4, int m4, int n) {
    static const int cand[] = { 16, 8, 4, 2 };
    if ((int64_t) m4 * n >= 65536) {
        return 1;
    }
    for (int s : cand) {
        if (k4 % s == 0 && k4 / s >= 8 && (int64_t) m4 * s <= max_dim) {
            return s;
        }
    }
    return 1;
}

// one weight row to F32 (the GPU only sees F32 texels)
static void ggml_d3d9_widen_row(ggml_type type, const void * src, float * dst, int64_t k) {
    if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, k);
    } else {
        ggml_get_type_traits(type)->to_float(src, dst, k);
    }
}

// element-wise read for the float types, used when a row is not dense (permuted tensors)
static void ggml_d3d9_gather_row(const ggml_tensor * t, const char * row, float * dst, int64_t k) {
    for (int64_t i = 0; i < k; i++) {
        const char * e = row + (size_t) i * t->nb[0];
        dst[i] = t->type == GGML_TYPE_F32 ? *(const float *) e :
                 t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*(const ggml_fp16_t *) e) : ggml_bf16_to_fp32(*(const ggml_bf16_t *) e);
    }
}

// float types may have any element stride (they are gathered); quantised types need whole blocks in a row
static bool ggml_d3d9_strides_ok(const ggml_tensor * t) {
    const bool flt = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
    return flt || t->nb[0] == ggml_type_size(t->type);
}

static bool ggml_d3d9_type_ok(ggml_type type, int64_t k) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) {
        return true;
    }
    const auto * tt = ggml_get_type_traits(type);
    return (type == GGML_TYPE_BF16 || ggml_is_quantized(type)) && tt->to_float != nullptr && k % ggml_blck_size(type) == 0;
}

static bool ggml_d3d9_mul_mat(ggml_d3d9_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (dst->ne[2] > 1 || dst->ne[3] > 1) {
        // batched / broadcast: one 2D product per (i2, i3); src0 is repeated r2 x r3 times over src1 (ggml broadcast rule)
        const int64_t r2 = src1->ne[2] / src0->ne[2], r3 = src1->ne[3] / src0->ne[3];
        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                ggml_tensor w = *src0, x = *src1, d = *dst;
                w.ne[2] = w.ne[3] = x.ne[2] = x.ne[3] = d.ne[2] = d.ne[3] = 1;
                w.data = (char *) src0->data + (size_t) (i2 / r2) * src0->nb[2] + (size_t) (i3 / r3) * src0->nb[3];
                x.data = (char *) src1->data + (size_t) i2 * src1->nb[2] + (size_t) i3 * src1->nb[3];
                d.data = (char *) dst->data + (size_t) i2 * dst->nb[2] + (size_t) i3 * dst->nb[3];
                d.src[0] = &w;
                d.src[1] = &x;
                if (!ggml_d3d9_mul_mat(ctx, &d)) {
                    return false;
                }
            }
        }
        return true;
    }

    if (src0->ne[1] > ctx->max_dim) {
        // more weight rows than the texture height allows: run max_dim-row chunks through the same path
        for (int64_t r0 = 0; r0 < src0->ne[1]; r0 += ctx->max_dim) {
            const int64_t n = std::min<int64_t>(ctx->max_dim, src0->ne[1] - r0);
            ggml_tensor w = *src0, d = *dst;
            w.ne[1] = n;
            w.data = (char *) src0->data + (size_t) r0 * src0->nb[1];
            d.ne[0] = n;
            d.data = (char *) dst->data + (size_t) r0 * dst->nb[0];
            d.src[0] = &w;
            d.src[1] = (ggml_tensor *) src1;
            if (!ggml_d3d9_mul_mat(ctx, &d)) {
                return false;
            }
        }
        return true;
    }

    const int K = (int) src0->ne[0], N = (int) src0->ne[1], M = (int) src1->ne[1];
    const int K4 = (K + 3) / 4, M4 = (M + 3) / 4;
    const int split = ggml_d3d9_pick_split(ctx->max_dim, K4, M4, N);
    const int chunk = K4 / split;
    const int inner = ggml_d3d9_largest_divisor_le(chunk, 128);
    const int outer = chunk / inner;

    // weights: cached only for buffers marked as weights (other buffers are reused for different data)
    // GGML_D3D9_CACHE_ALL=1 (benchmarks only): cache every weight tensor by pointer and size; wrong if a buffer is reused for different data
    static const bool cache_all = getenv("GGML_D3D9_CACHE_ALL") != nullptr;
    const bool cache_w = src0->nb[0] == ggml_type_size(src0->type) &&   // permuted (strided) tensors are never cached: the key has no strides
                         (cache_all || (src0->buffer && src0->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
    const auto wkey = std::make_tuple((const void *) src0->data, ggml_nbytes(src0), (int) src0->type);
    IDirect3DTexture9 * wtex = nullptr;
    bool wtemp = false;
    if (cache_w) {
        auto it = ctx->weights.find(wkey);
        if (it != ctx->weights.end()) {
            wtex = it->second;
        }
    }
    if (!wtex) {
        // F16 and quantised weights go up as half floats (half the video memory); F32 and BF16 stay F32. A value that does not
        // fit a half (|x| > 65000) sends the tensor back through the F32 path.
        bool half = ctx->half_tex && src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_BF16;
        bool ok = false;
        bool was_half = false;
        for (;;) {
            bool overflow = false;
            wtex = ggml_d3d9_new_tex(ctx, K4, N, half);
            if (!wtex) {
                return false;
            }
            if (src0->type == GGML_TYPE_F32) {
                std::vector<float> wrow;
                if (src0->nb[0] != 4) {
                    wrow.resize((size_t) K);
                }
                ok = ggml_d3d9_fill(ctx, wtex, K4, N, N, K, [&](int r) -> const void * {
                    const char * rp = (const char *) src0->data + (size_t) r * src0->nb[1];
                    if (src0->nb[0] == 4) {
                        return rp;
                    }
                    ggml_d3d9_gather_row(src0, rp, wrow.data(), K);   // permuted weights: strided rows
                    return wrow.data();
                });
            } else {
                std::vector<float> row((size_t) K);
                // F16, BF16 and quantised rows are widened one at a time while the staging texture is filled
                IDirect3DTexture9 * stage = ggml_d3d9_stage(ctx, K4, N, half);
                D3DLOCKED_RECT lr;
                ok = stage && SUCCEEDED(stage->LockRect(0, &lr, nullptr, 0));
                if (ok) {
                    const size_t esz = half ? 2 : 4;
                    for (int r = 0; r < N; r++) {
                        char * d = (char *) lr.pBits + (size_t) r * lr.Pitch;
                        const char * sp = (const char *) src0->data + (size_t) r * src0->nb[1];
                        if (half && src0->type == GGML_TYPE_F16 && src0->nb[0] == 2) {
                            memcpy(d, sp, (size_t) K * 2);
                        } else {
                            if (src0->nb[0] == ggml_type_size(src0->type)) {
                                ggml_d3d9_widen_row(src0->type, sp, row.data(), K);
                            } else {
                                ggml_d3d9_gather_row(src0, sp, row.data(), K);
                            }
                            if (half) {
                                for (int i = 0; i < K; i++) {
                                    overflow |= !(fabsf(row[i]) <= 65000.0f);
                                }
                                ggml_fp32_to_fp16_row(row.data(), (ggml_fp16_t *) d, K);
                            } else {
                                memcpy(d, row.data(), (size_t) K * 4);
                            }
                        }
                        memset(d + (size_t) K * esz, 0, (size_t) K4 * 4 * esz - (size_t) K * esz);
                    }
                    stage->UnlockRect(0);
                    ok = SUCCEEDED(ctx->dev->UpdateTexture(stage, wtex));
                }
            }
            was_half = half;
            if (ok && overflow) {
                wtex->Release();
                wtex = nullptr;
                half = false;
                continue;
            }
            break;
        }
        if (!ok) {
            wtex->Release();
            return false;
        }
        if (cache_w) {
            ctx->weights[wkey] = wtex;
            ctx->w_bytes += (uint64_t) K4 * N * (was_half ? 8 : 16);
        } else {
            wtemp = true;
        }
    }

    // activations: M rows of K floats, padded with zero rows up to a multiple of 4
    auto xit = ctx->xtex.find(key2(K4, M4 * 4));
    IDirect3DTexture9 * xt = nullptr;
    if (xit != ctx->xtex.end()) {
        xt = xit->second;
    } else {
        xt = ggml_d3d9_new_tex(ctx, K4, M4 * 4);
        if (!xt) {
            if (wtemp) wtex->Release();
            return false;
        }
        ctx->xtex[key2(K4, M4 * 4)] = xt;
    }
    // F16 / BF16 activations are widened on the CPU; a permuted src1 (element stride != size) is gathered
    const bool x_conv = src1->type != GGML_TYPE_F32 || src1->nb[0] != 4;
    std::vector<float> xrow;
    if (x_conv) {
        xrow.resize((size_t) K);
    }
    bool ok = ggml_d3d9_fill(ctx, xt, K4, M4 * 4, M, K, [&](int m) -> const void * {
        const char * p = (const char *) src1->data + (size_t) m * src1->nb[1];
        if (!x_conv) {
            return p;
        }
        if (src1->nb[0] != ggml_type_size(src1->type)) {
            ggml_d3d9_gather_row(src1, p, xrow.data(), K);
        } else if (src1->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) p, xrow.data(), K);
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) p, xrow.data(), K);
        }
        return xrow.data();
    });

    // shaders
    auto mkey = std::make_tuple(split, inner, outer);
    IDirect3DPixelShader9 * ps = nullptr;
    auto pit = ctx->mm_ps.find(mkey);
    if (pit != ctx->mm_ps.end()) {
        ps = pit->second;
    } else if (ok) {
        ps = ggml_d3d9_make_ps(ctx, def_int("SPLIT", split) + def_int("CHUNK", chunk) + def_int("INNER", inner) + def_int("OUTER", outer), ps_matmul);
        if (ps) ctx->mm_ps[mkey] = ps;
    }
    IDirect3DPixelShader9 * psr = nullptr;
    if (ok && ps && split > 1) {
        auto rit = ctx->red_ps.find(split);
        if (rit != ctx->red_ps.end()) {
            psr = rit->second;
        } else {
            psr = ggml_d3d9_make_ps(ctx, def_int("SPLIT", split), ps_reduce);
            if (psr) ctx->red_ps[split] = psr;
        }
    }

    ggml_d3d9_rt fin, part;
    ok = ok && ps && (split == 1 || psr) && ggml_d3d9_rt_get(ctx, M4, N, fin) && (split == 1 || ggml_d3d9_rt_get(ctx, M4 * split, N, part));
    IDirect3DSurface9 * sys = nullptr;
    if (ok) {
        auto sit = ctx->readbacks.find(key2(M4, N));
        if (sit != ctx->readbacks.end()) {
            sys = sit->second;
        } else if (SUCCEEDED(ctx->dev->CreateOffscreenPlainSurface(M4, N, D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM, &sys, nullptr))) {
            ctx->readbacks[key2(M4, N)] = sys;
        } else {
            ok = false;
        }
    }
    if (ok) {
        const float inv[4] = { 1.0f / K4, 1.0f / N, 1.0f / K4, 1.0f / (M4 * 4) };
        HRESULT hr;
        if (split == 1) {
            hr = ggml_d3d9_draw(ctx, fin.surf, ps, wtex, xt, inv);
        } else {
            hr = ggml_d3d9_draw(ctx, part.surf, ps, wtex, xt, inv);
            if (SUCCEEDED(hr)) {
                const float rinv[4] = { 1.0f / (M4 * split), 1.0f / N, 0, 0 };
                hr = ggml_d3d9_draw(ctx, fin.surf, psr, part.tex, nullptr, rinv);
            }
        }
        if (FAILED(hr)) {
            ggml_d3d9_report(ctx, "draw", hr, K, N, M);
            ok = false;
        }
        D3DLOCKED_RECT lr;
        if (ok) {
            hr = ctx->dev->GetRenderTargetData(fin.surf, sys);
            if (FAILED(hr)) {
                ggml_d3d9_report(ctx, "GetRenderTargetData", hr, K, N, M);
            } else {
                hr = sys->LockRect(&lr, nullptr, D3DLOCK_READONLY);
                if (FAILED(hr)) {
                    ggml_d3d9_report(ctx, "LockRect", hr, K, N, M);
                }
            }
            ok = SUCCEEDED(hr);
        }
        if (ok) {
            // pixel (m4, r) holds outputs for columns 4*m4 .. 4*m4+3 of row r
            for (int r = 0; r < N; r++) {
                const float * px = (const float *) ((const char *) lr.pBits + (size_t) r * lr.Pitch);
                for (int m = 0; m < M; m++) {
                    *(float *) ((char *) dst->data + (size_t) m * dst->nb[1] + (size_t) r * dst->nb[0]) = px[m];
                }
            }
            sys->UnlockRect();
        }
    }
    if (wtemp) {
        wtex->Release();
    }
    ctx->n_calls++;
    if (d3d9_debug()) {
        GGML_LOG_INFO("%s: K=%d N=%d M=%d split=%d inner=%d outer=%d -> %s\n", __func__, K, N, M, split, inner, outer, ok ? "ok" : "FAILED");
    }
    return ok;
}

// MUL_MAT_ID (mixture of experts): the CPU groups the (id, token) pairs by expert, then each used expert runs one 2D product on the
// GPU through the same path as MUL_MAT (the expert's weight slice is cached like any other weight) and the rows are scattered back
static bool ggml_d3d9_mul_mat_id(ggml_d3d9_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];   // experts: K x rows x n_as
    const ggml_tensor * src1 = dst->src[1];   // K x (n_ids or 1) x n_tokens
    const ggml_tensor * ids  = dst->src[2];   // n_ids x n_tokens, I32
    const int64_t K = src0->ne[0], R = src0->ne[1], n_as = src0->ne[2];
    const int64_t n_ids = ids->ne[0], n_tok = ids->ne[1];

    std::vector<std::vector<std::pair<int, int>>> groups((size_t) n_as);   // per expert: (id slot, token)
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t i = 0; i < n_ids; i++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + (size_t) t * ids->nb[1] + (size_t) i * ids->nb[0]);
            if (e < 0 || e >= n_as) {
                GGML_LOG_ERROR("%s: expert id %d out of range (%d experts)\n", __func__, (int) e, (int) n_as);
                return false;
            }
            groups[(size_t) e].push_back({ (int) i, (int) t });
        }
    }

    std::vector<float> xbuf, ybuf;
    for (int64_t e = 0; e < n_as; e++) {
        const auto & g = groups[(size_t) e];
        for (size_t c0 = 0; c0 < g.size(); c0 += (size_t) ctx->max_dim) {
            const size_t cnt = std::min<size_t>((size_t) ctx->max_dim, g.size() - c0);
            xbuf.resize(cnt * (size_t) K);
            ybuf.resize(cnt * (size_t) R);
            for (size_t j = 0; j < cnt; j++) {
                const char * p = (const char *) src1->data + (size_t) (g[c0 + j].first % src1->ne[1]) * src1->nb[1] + (size_t) g[c0 + j].second * src1->nb[2];
                if (src1->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) p, xbuf.data() + j * (size_t) K, K);
                } else {
                    memcpy(xbuf.data() + j * (size_t) K, p, (size_t) K * 4);
                }
            }
            ggml_tensor w = *src0, x = {}, d = {};
            w.ne[2] = w.ne[3] = 1;
            w.data = (char *) src0->data + (size_t) e * src0->nb[2];
            x.type = GGML_TYPE_F32;
            x.ne[0] = K; x.ne[1] = (int64_t) cnt; x.ne[2] = x.ne[3] = 1;
            x.nb[0] = 4; x.nb[1] = 4 * (size_t) K; x.nb[2] = x.nb[3] = x.nb[1] * cnt;
            x.data = xbuf.data();
            d.type = GGML_TYPE_F32;
            d.ne[0] = R; d.ne[1] = (int64_t) cnt; d.ne[2] = d.ne[3] = 1;
            d.nb[0] = 4; d.nb[1] = 4 * (size_t) R; d.nb[2] = d.nb[3] = d.nb[1] * cnt;
            d.data = ybuf.data();
            d.src[0] = &w;
            d.src[1] = &x;
            if (!ggml_d3d9_mul_mat(ctx, &d)) {
                return false;
            }
            for (size_t j = 0; j < cnt; j++) {
                memcpy((char *) dst->data + (size_t) g[c0 + j].first * dst->nb[1] + (size_t) g[c0 + j].second * dst->nb[2],
                       ybuf.data() + j * (size_t) R, (size_t) R * 4);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------- backend interface

static const char * ggml_backend_d3d9_get_name(ggml_backend_t backend) {
    return "D3D9";

    GGML_UNUSED(backend);
}

// GGML_D3D9_STATS=1: count offloaded MUL_MATs on stderr (diagnostic)
static const bool g_d3d9_stats = getenv("GGML_D3D9_STATS") != nullptr;
static int g_d3d9_calls = 0;

static void ggml_backend_d3d9_free(ggml_backend_t backend) {
    if (getenv("GGML_D3D9_STATS")) {
        fprintf(stderr, "D3D9 STATS: backend free, total MUL_MAT offloaded %d, cached weight textures %zu = %llu MB\n", g_d3d9_calls,
                ((ggml_d3d9_context *) backend->context)->weights.size(), (unsigned long long) (((ggml_d3d9_context *) backend->context)->w_bytes >> 20));
    }
    ggml_d3d9_context_free((ggml_d3d9_context *) backend->context);
    delete backend;
}

static enum ggml_status ggml_backend_d3d9_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_d3d9_context * ctx = (ggml_d3d9_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (g_d3d9_stats) {
                    if (++g_d3d9_calls == 1) fprintf(stderr, "D3D9 STATS: first MUL_MAT offloaded\n");
                    if (g_d3d9_calls % 500 == 0) fprintf(stderr, "D3D9 STATS: %d MUL_MAT offloaded\n", g_d3d9_calls);
                }
                if (!ggml_d3d9_mul_mat(ctx, node)) {
                    GGML_LOG_ERROR("%s: MUL_MAT failed\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                break;

            case GGML_OP_MUL_MAT_ID:
                if (!ggml_d3d9_mul_mat_id(ctx, node)) {
                    GGML_LOG_ERROR("%s: MUL_MAT_ID failed\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i d3d9_backend_i = {
    /* .get_name                = */ ggml_backend_d3d9_get_name,
    /* .free                    = */ ggml_backend_d3d9_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_d3d9_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_d3d9_guid(void) {
    static ggml_guid guid = { 0x39, 0xd9, 0x5a, 0x11, 0x7c, 0x2e, 0x4b, 0x80, 0xa3, 0x0f, 0x6d, 0x91, 0x04, 0xe7, 0x2c, 0x59 };
    return &guid;
}

// device interface

static const char * ggml_backend_d3d9_device_get_name(ggml_backend_dev_t dev) {
    static std::deque<std::string> names;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    size_t idx = (size_t) (intptr_t) dev->context;
    if (names.size() <= idx) {
        names.resize(idx + 1);
    }
    if (names[idx].empty()) {
        names[idx] = std::string(GGML_D3D9_NAME) + std::to_string(idx);
    }
    return names[idx].c_str();
}

static const char * ggml_backend_d3d9_device_get_description(ggml_backend_dev_t dev) {
    auto & ads = ggml_d3d9_adapters();
    size_t idx = (size_t) (intptr_t) dev->context;
    return idx < ads.size() ? ads[idx].name.c_str() : "Direct3D 9";
}

static void ggml_backend_d3d9_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // textures are cached per use; no device memory is reported (the CPU buffer type holds the tensors)
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_d3d9_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_d3d9_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_d3d9_device_get_name(dev);
    props->description = ggml_backend_d3d9_device_get_description(dev);
    props->type        = ggml_backend_d3d9_device_get_type(dev);
    ggml_backend_d3d9_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_d3d9_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_d3d9_context * ctx = new ggml_d3d9_context;
    ctx->device_index = (int) (intptr_t) dev->context;
    if (!ggml_d3d9_context_init(ctx)) {
        ggml_d3d9_context_free(ctx);
        return nullptr;
    }
    return new ggml_backend {
        /* .guid    = */ ggml_backend_d3d9_guid(),
        /* .iface   = */ d3d9_backend_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_d3d9_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_d3d9_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_d3d9_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];

            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }
            // batched and broadcast mat-muls run as one 2D product per slice (GGML_D3D9_BATCHED=1: each slice costs an upload and a
            // readback, so they slow token generation until tensors stay on the GPU; off by default); sizes within the texture limits
            static const bool batched = getenv("GGML_D3D9_BATCHED") != nullptr;
            const int max_dim = ggml_d3d9_adapters()[(size_t) (intptr_t) dev->context].max_dim;
            // GGML_D3D9_MIN_N=<n>: offload only products with at least n columns (n=1 token generation is slower on the GPU path)
            static const int64_t min_n = getenv("GGML_D3D9_MIN_N") ? atoll(getenv("GGML_D3D9_MIN_N")) : 1;
            // src0 rows must be dense inside (any row stride is fine); src1 may be F32 or F16
            return src1->ne[1] >= min_n && ggml_d3d9_type_ok(src0->type, src0->ne[0]) &&
                   (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16) && op->type == GGML_TYPE_F32 &&
                   ggml_d3d9_strides_ok(src0) && ggml_d3d9_strides_ok(src1) && ggml_is_contiguous(op) &&
                   (batched || (src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1)) &&
                   src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0 &&
                   (src0->ne[0] + 3) / 4 <= max_dim && src1->ne[1] <= max_dim;
        }

        case GGML_OP_MUL_MAT_ID:
        {
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];
            const ggml_tensor * ids  = op->src[2];
            // GGML_D3D9_MOE_MIN_TOKENS=<n>: offload only batches of at least n tokens (a token-generation step makes n_used tiny products)
            static const int64_t min_tok = getenv("GGML_D3D9_MOE_MIN_TOKENS") ? atoll(getenv("GGML_D3D9_MOE_MIN_TOKENS")) : 8;
            const int max_dim = ggml_d3d9_adapters()[(size_t) (intptr_t) dev->context].max_dim;
            return ids && ids->type == GGML_TYPE_I32 && ids->ne[1] >= min_tok && ggml_d3d9_type_ok(src0->type, src0->ne[0]) &&
                   (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) && ggml_is_contiguous(op) &&
                   src0->ne[3] == 1 && src1->ne[3] == 1 && (src0->ne[0] + 3) / 4 <= max_dim;
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_d3d9_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_d3d9_device_i = {
    /* .get_name             = */ ggml_backend_d3d9_device_get_name,
    /* .get_description      = */ ggml_backend_d3d9_device_get_description,
    /* .get_memory           = */ ggml_backend_d3d9_device_get_memory,
    /* .get_type             = */ ggml_backend_d3d9_device_get_type,
    /* .get_props            = */ ggml_backend_d3d9_device_get_props,
    /* .init_backend         = */ ggml_backend_d3d9_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_d3d9_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_d3d9_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_d3d9_device_supports_op,
    /* .supports_buft        = */ ggml_backend_d3d9_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_d3d9_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_D3D9_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_d3d9_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_d3d9_adapters().size();

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_d3d9_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    static std::vector<ggml_backend_device *> devs;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    GGML_ASSERT(index < ggml_d3d9_adapters().size());
    if (devs.size() <= index) {
        devs.resize(index + 1, nullptr);
    }
    if (!devs[index]) {
        devs[index] = new ggml_backend_device {
            /* .iface   = */ ggml_backend_d3d9_device_i,
            /* .reg     = */ reg,
            /* .context = */ (void *) (intptr_t) index,
        };
    }
    return devs[index];
}

static const struct ggml_backend_reg_i ggml_backend_d3d9_reg_i = {
    /* .get_name         = */ ggml_backend_d3d9_reg_get_name,
    /* .get_device_count = */ ggml_backend_d3d9_reg_get_device_count,
    /* .get_device       = */ ggml_backend_d3d9_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_d3d9_reg(void) {
    static struct ggml_backend_reg ggml_backend_d3d9_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_d3d9_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_d3d9_reg;
}

ggml_backend_t ggml_backend_d3d9_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_d3d9_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, (size_t) device), nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_d3d9_reg)
