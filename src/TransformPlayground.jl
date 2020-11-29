module TransformPlayground

export ω, dct_makhoul_1d

using CUDA

include("dct_makhoul.jl")
include("benchmark_ffts.jl")

end # module
