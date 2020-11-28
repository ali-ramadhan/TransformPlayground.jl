using Random
using Test
using FFTW
using CUDA

Random.seed!(0)

FT = Float64
ε = eps(FT)

# Test both even and odd sizes
for N in (4, 5)
    A_cpu = rand(Float64, N)
    A_gpu = CuArray(A_cpu)

    B_cpu = FFTW.fft(A_cpu)
    B_gpu = CUDA.CUFFT.fft(A_gpu)

    @test isapprox.(B_cpu, Vector(B_gpu), atol=ε) |> all

    C_cpu = FFTW.ifft(B_cpu)
    C_gpu = CUDA.CUFFT.ifft(B_gpu)

    @test isapprox.(C_cpu, Vector(C_gpu), atol=ε) |> all

    @test isapprox.(A_cpu, C_cpu, atol=ε) |> all
    @test isapprox.(Vector(A_gpu), Vector(C_gpu), atol=ε) |> all
end