using Random
using Test
using FFTW
using CUDA

Random.seed!(0)

FT = Float64
ε = eps(FT)

N = 4

A_cpu = rand(Float64, (N, N, N))
A_gpu = CuArray(A_cpu)

B_cpu = FFTW.fft(A_cpu, 2)

B_gpu = permutedims(A_gpu, (2, 1, 3))
B_gpu = CUDA.CUFFT.fft(B_gpu, 1)
B_gpu = permutedims(B_gpu, (2, 1, 3))