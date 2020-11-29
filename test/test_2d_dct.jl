using Random
using Test
using FFTW
using CUDA
using TransformPlayground

Random.seed!(0)

FT = Float64
ε = eps(FT)

for N in (4, 5)
    A_cpu = randn(FT, N, N)
    A_gpu = CuArray(A_cpu)

    B_cpu = FFTW.r2r(A_cpu, FFTW.REDFT10)
    B_gpu = TransformPlayground.dct_makhoul_2d(A_gpu)

    @test isapprox.(B_cpu, Array(B_gpu), atol=16ε) |> all

    C_cpu = FFTW.r2r(B_cpu, FFTW.REDFT01)
    C_cpu = C_cpu ./ (2N)^2

    C_gpu = TransformPlayground.idct_makhoul_2d(B_gpu)

    @test isapprox.(C_cpu, Array(C_gpu), atol=ε) |> any

    @test isapprox.(A_cpu, C_cpu, atol=4ε) |> all
    @test isapprox.(Array(A_gpu), Array(C_gpu), atol=4ε) |> all
end
