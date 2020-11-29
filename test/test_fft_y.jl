using Random
using Test
using FFTW
using CUDA

Random.seed!(0)

FT = Float64
ε = eps(FT)

for N in (4, 5)
    A_cpu = rand(FT, (N, N, N))
    A_gpu = CuArray(A_cpu)

    B_cpu = FFTW.fft(A_cpu, 2)

    B_gpu = permutedims(A_gpu, (2, 1, 3))
    B_gpu = CUDA.CUFFT.fft(B_gpu, 1)
    B_gpu = permutedims(B_gpu, (2, 1, 3))

    @test isapprox.(B_cpu, Array(B_gpu), atol=4ε) |> all

    C_cpu = FFTW.ifft(B_cpu, 2)

    C_gpu = permutedims(B_gpu, (2, 1, 3))
    C_gpu = CUDA.CUFFT.ifft(C_gpu, 1)
    C_gpu = permutedims(C_gpu, (2, 1, 3))

    @test isapprox.(C_cpu, Array(C_gpu), atol=4ε) |> all

    @test isapprox.(C_cpu, A_cpu, atol=ε) |> all
    @test isapprox.(Array(C_gpu), Array(A_gpu), atol=ε) |> all
end