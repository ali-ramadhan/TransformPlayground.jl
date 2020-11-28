using Random
using Test
using FFTW
using CUDA

Random.seed!(0)

A_cpu = randn(Float64, 5)
A_gpu = CuArray(A_cpu)

B_cpu = FFTW.fft(A_cpu)
B_gpu = CUDA.CUFFT.fft(A_gpu)

@test isapprox.(B_cpu, Array(B_gpu), atol=eps(Float64)) |> all

C_cpu = FFTW.ifft(B_cpu)
C_gpu = CUDA.CUFFT.ifft(B_gpu)

@test isapprox.(C_cpu, Array(C_gpu), atol=eps(Float64)) |> all
