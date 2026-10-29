# foldquant_int4_groupwise: INT4 weight-only groupwise GEMM (W4A16, group 128).
#
# Consumes ModelOpt AWQ-quantized weights; the graph surgery that rewrites
# trt::DequantizeLinear into this plugin lives in the TensorRT target.
#
# The only CUTLASS-free kernel: its GEMM/GEMV are hand-written, so this target
# builds on a checkout without the submodule.
foldquant_add_kernel(foldquant_int4_groupwise
    DIR int4_groupwise
    SOURCES
        cuda/int4_woq_gemm_cuda.cu
        cuda/int4_woq_gemv_cuda.cu
        plugin/int4_groupwise_gemm_plugin.cpp
)
