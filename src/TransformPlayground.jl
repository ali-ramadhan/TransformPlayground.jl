module TransformPlayground

export dct_makhoul_1d, idct_makhoul_1d,
       benchmark_cpu_fft, benchmark_cpu_dct

using BenchmarkTools
using FFTW
using CUDA

include("dct_makhoul.jl")
include("benchmark_ffts.jl")

end # module
