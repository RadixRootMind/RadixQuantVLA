# foldquant_int4_per_row: W4A4 DiT macro plugins (scheme w4a4_sr).
#
# 4-bit weights AND activations on the DiT stack: rotated-INT4 GEMM +
# BF16-folded rotation + cuBLAS SDPA + INT4 weight-only AdaLN. Each DiT block
# collapses to ~2 plugin nodes.
#
# NEEDS_CUTLASS: dit_int4_rowwise_gemm_fused_cuda.cu includes <cutlass/cutlass.h>.
#   (This was previously declared CUTLASS-free, which broke the build; the
#   dependency now lives here, next to the source that has it.)
# NEEDS_CUBLAS: BF16 rotation + batched BF16 SDPA inside the W4A4 macros.
#
# sdpa_softmax_cuda.cu is compiled into this library as well as int8_per_row's:
# it is the one shared source between the two sets, and duplicating the object
# keeps each .so self-contained rather than introducing a link-order dependency
# between two independently loadable plugins.
foldquant_add_kernel(foldquant_int4_per_row
    DIR int4_per_row
    NEEDS_CUTLASS
    NEEDS_CUBLAS
    EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/int8_per_row/cuda
    SOURCES
        cuda/dit_int4_rowwise_gemm_fused_cuda.cu
        cuda/per_row_rotate_quant_int4_cuda.cu
        cuda/per_row_quant_int4_cuda.cu
        cuda/dit_int4_prologue_bf16_cuda.cu
        cuda/dit_int4_block_rotate_cublas.cu
        cuda/adaln_gemv_int4_cuda.cu
        ../int8_per_row/cuda/sdpa_softmax_cuda.cu
        plugin/fused_ffn_int4_plugin.cpp
        plugin/fused_selfattn_full_int4_plugin.cpp
        plugin/fused_crossattn_full_int4_plugin.cpp
        plugin/encoder_prequant_int4_plugin.cpp
        plugin/adaln_mod_int4_plugin.cpp
        plugin/per_row_int4_linear_residual_plugin.cpp
        cuda/rmsnorm_per_row_quant_int4_cuda.cu
        plugin/fused_rmsnorm_linear_int4_plugin.cpp
)
