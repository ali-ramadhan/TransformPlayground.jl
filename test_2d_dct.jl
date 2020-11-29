using Random
using Test
using FFTW
using CUDA
using TransformPlayground

Random.seed!(0)

FT = Float64
ε = eps(FT)

N = 4

A_cpu = randn(Float64, N, N)
A_gpu = CuArray(A_cpu)

B_cpu = FFTW.r2r(A_cpu, FFTW.REDFT10)
B_gpu = TransformPlayground.dct_makhoul_2d(A_gpu)

B_cpu ./ Array(B_gpu)

#=
@test isapprox.(B_cpu, Vector(B_gpu), atol=2ε) |> all

C_cpu = FFTW.r2r(B_cpu, FFTW.REDFT01)
C_cpu = C_cpu ./ 2N

C_gpu = TransformPlayground.idct_makhoul_1d(B_gpu)

@test isapprox.(C_cpu, Vector(C_gpu), atol=ε) |> any

@test isapprox.(A_cpu, C_cpu, atol=ε) |> all
@test isapprox.(Vector(A_gpu), Vector(C_gpu), atol=ε) |> all
=#