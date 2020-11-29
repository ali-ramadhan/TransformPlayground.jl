module TransformPlayground

export dct_makhoul_1d, idct_makhoul_1d,
       dct_makhoul_2d, idct_makhoul_2d,
       benchmark_cpu_fft, benchmark_cpu_dct,
       benchmark_cpu_fft_y, benchmark_gpu_fft_y

using BenchmarkTools
using FFTW
using CUDA

"""
    ω(M, k)
Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

include("dct_makhoul.jl")
include("benchmark_ffts.jl")

function __init__()
    # This is a repo for playing around and debugging.
    # Who cares about performance!?
    CUDA.allowscalar(true)
end

end # module
