function benchmark_cpu_fft(size, dims; FT=Float64, planner_flag=FFTW.PATIENT)
    A = zeros(complex(FT), size...)
    FFT! = FFTW.plan_fft!(A, dims, flags=planner_flag)
    trial = @benchmark ($FFT! * $A) samples=10
    return trial
end

# 3x1D: 61+111+98 = 270 ms
# 3D: 245 ms

function benchmark_cpu_dct(size, dims; FT=Float64, planner_flag=FFTW.PATIENT)
    A = zeros(FT, size...)
    DCT! = FFTW.plan_r2r!(A, FFTW.REDFT10, dims, flags=planner_flag)
    trial = @benchmark ($DCT! * $A) samples=10
    return trial
end

# 3x1D: 86+122+143=351 ms
# 3D: 328 ms

function benchmark_cpu_fft_y(N; FT=Float64, planner_flag=FFTW.PATIENT)
    A = zeros(complex(FT), (N, N, N))
    B = similar(A)
    FFT! = FFTW.plan_fft!(A, 1, flags=planner_flag)

    trial = @benchmark begin
        permutedims!($B, $A, (2, 1, 3))
        $FFT! * $B
        permutedims!($A, $B, (2, 1, 3))
    end

    return trial
end

# 250 ms vs. 111 ms

function benchmark_gpu_fft_y(N; FT=Float64)
    A = zeros(complex(FT), (N, N, N)) |> CuArray
    B = similar(A)
    FFT! = CUDA.CUFFT.plan_fft!(A, 1)

    trial = @benchmark begin
        CUDA.@sync begin
            permutedims!($B, $A, (2, 1, 3))
            $FFT! * $B
            permutedims!($A, $B, (2, 1, 3))
        end
    end

    return trial
end

# 5 ms vs. 3 ms
