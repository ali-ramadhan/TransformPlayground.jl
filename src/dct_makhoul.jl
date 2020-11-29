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

function idct_makhoul_2d(A::CuArray)
    Nx, Ny = size(A)

    # IDCT along dimension 1

    B = similar(A, complex(eltype(A)))

    for j in 1:Ny
        B[1, j] = 1/2 * ω(4Nx, 0) * A[1, j]
    end

    for j in 1:Ny, i in 2:Nx
        B[i, j] = ω(4Nx, 1-i) * A[i, j]
    end

    B = CUDA.CUFFT.ifft(B, 1)

    C = similar(A)
    for j in 1:Ny, i in 1:Nx
        C[unpermute(i, Nx), j] = real(B[i, j])
    end

    # IDCT along dimension 2

    D = similar(A, complex(eltype(A)))

    for i in 1:Nx
        D[i, 1] = 1/2 * ω(4Ny, 0) * C[i, 1]
    end

    for j in 2:Ny, i in 1:Nx
        D[i, j] = ω(4Ny, 1-j) * C[i, j]
    end

    D = CUDA.CUFFT.ifft(D, 2)

    E = similar(A)
    for j in 1:Ny, i in 1:Nx
        E[i, unpermute(j, Ny)] = real(D[i, j])
    end

    return E
end