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

function dct_makhoul_2d(A::CuArray)
    Nx, Ny = size(A)

    # DCT along dimension 1

    B = similar(A)

    for j in 1:Ny, i in 1:Nx
        B[permute(i, Nx), j] = A[i, j]
    end

    B = CUDA.CUFFT.fft(B, 1)

    for j in 1:Ny, i in 1:Nx
        B[i, j] = 2 * ω(4Nx, i-1) * B[i, j]
    end

    B = real(B)

    # DCT along dimension 2

    C = similar(A)

    for j in 1:Ny, i in 1:Nx
        C[i, permute(j, Ny)] = B[i, j]
    end

    C = CUDA.CUFFT.fft(C, 2)

    for j in 1:Ny, i in 1:Nx
        C[i, j] = 2 * ω(4Ny, j-1) * C[i, j]
    end

    return real(C)
end

function dct_makhoul_3d(A::CuArray)
    B = similar(A)
    Nx, Ny, Nz = size(A)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        i′ = permute(i, Nx)
        j′ = permute(j, Ny)
        k′ = permute(k, Nz)
        B[i′, j′, k′] = A[i, j, k]
    end

    B = CUDA.CUFFT.fft(B)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        B[i, j, k] = 2 * ω(4Nx, i-1) * B[i, j, k]
        B[i, j, k] = 2 * ω(4Ny, j-1) * B[i, j, k]
        B[i, j, k] = 2 * ω(4Nz, k-1) * B[i, j, k]
    end

    return real(B)
end
