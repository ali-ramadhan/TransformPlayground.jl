module TransformPlayground

export dct_makhoul_1d, idct_makhoul_1d,
       benchmark_cpu_fft, benchmark_cpu_dct,
       benchmark_cpu_fft_y, benchmark_gpu_fft_y

using BenchmarkTools
using FFTW
using CUDA

include("dct_makhoul.jl")
include("benchmark_ffts.jl")

end # module
