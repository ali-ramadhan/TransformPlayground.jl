module TransformPlayground

export ω, dct_makhoul_1d

using CUDA

"""
    ω(M, k)
Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

@inline permute(i, N) = isodd(i) ? floor(Int, i/2) + 1 : N - floor(Int, (i-1)/2)

@inline unpermute(i, N) = i <= ceil(N/2) ? 2i-1 : 2(N-i+1)

function dct_makhoul_1d(A::CuArray)
    B = similar(A)
    N = length(A)

    for k in 1:N
        B[permute(k, N)] = A[k]
    end

    B = CUDA.CUFFT.fft(B)

    for k in 1:N
        B[k] = 2 * ω(4N, k-1) * B[k]
    end

    return real(B)
end

function idct_makhoul_1d(A::CuArray)
    B = similar(A, complex(eltype(A)))
    N = length(A)

    B[1] = 1/2 * ω(4N, 0) * A[1]
    for k in 2:N
        B[k] = ω(4N, 1-k) * A[k]
    end

    B = CUDA.CUFFT.ifft(B)

    C = similar(A)
    for k in 1:N
        C[unpermute(k, N)] = real(B[k])
    end

    return C
end

end # module
