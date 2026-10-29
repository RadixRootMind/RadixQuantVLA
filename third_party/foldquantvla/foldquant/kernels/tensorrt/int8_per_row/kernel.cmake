# foldquant_int8_per_row: INT8 dynamic per-row DiT/LLM macro plugins.
#
# Weights INT8 with a per-row scale; activation amax is computed at runtime
# inside each plugin, so this scheme needs no calibration.
#
# NEEDS_CUTLASS: dit_int8_rowwise_v2*_cuda.cu include <cutlass/...>.
# NEEDS_CUBLAS: batched BF16 SDPA (Q.K^T, scores.V) inside the v2 macro plugins.
foldquant_add_kernel(foldquant_int8_per_row
    DIR int8_per_row
    NEEDS_CUTLASS
    NEEDS_CUBLAS
    SOURCES
        cuda/dit_int8_rowwise_v2_cuda.cu
        cuda/dit_int8_rowwise_v2_fused_cuda.cu
        cuda/dit_int8_rowwise_v2_fused_t128_cuda.cu
        cuda/dit_int8_rowwise_v2_tiles_cuda.cu
        cuda/per_row_quant_cuda.cu
        cuda/rmsnorm_per_row_quant_cuda.cu
        cuda/gelu_quant_cuda.cu
        cuda/fused_adaln_quant_cuda.cu
        cuda/sdpa_softmax_cuda.cu
        plugin/encoder_prequant_plugin.cpp
        plugin/fused_selfattn_full_plugin.cpp
        plugin/fused_crossattn_full_plugin.cpp
        plugin/fused_crossattn_full_cached_plugin.cpp
        plugin/fused_cross_attn_prequantized_plugin.cpp
        plugin/fused_norm_projout_plugin.cpp
        plugin/fused_ffn_plugin.cpp
        plugin/fused_rmsnorm_linear_int8_plugin.cpp
        plugin/per_row_int8_linear_residual_plugin.cpp
)
