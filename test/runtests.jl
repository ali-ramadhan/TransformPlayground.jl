using Test
using TransformPlayground

@testset "TransformPlayground" begin
    include("test_1d_fft.jl")
    include("test_1d_dct.jl")
    include("test_2d_dct.jl")
    include("test_fft_y.jl")
end
