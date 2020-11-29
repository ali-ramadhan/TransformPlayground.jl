function benchmark_cpu_fft(size, dims; FT=Float64, planner_flag=FFTW.PATIENT)
    A = zeros(complex(FT), size...)
    FFT! = FFTW.plan_fft!(A, dims, flags=planner_flag)
    trial = @benchmark ($FFT! * $A) samples=10
    return trial
end

# 61+111+98 = 270
# 245

function benchmark_cpu_dct(size, dims; FT=Float64, planner_flag=FFTW.PATIENT)
    A = zeros(FT, size...)
    DCT! = FFTW.plan_r2r!(A, FFTW.REDFT10, dims, flags=planner_flag)
    trial = @benchmark ($DCT! * $A) samples=10
    return trial
end

# 86+122+143=351
# 328
