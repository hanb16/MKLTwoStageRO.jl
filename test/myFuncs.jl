using MKLTwoStageRO
using LinearAlgebra: Diagonal
using Test: @test
using SparseArrays: sparse, spdiagm, spzeros


## --- Case-4: Two-Stage Robust Location-Transportation Problem ---
"Convert the general form Tow-Stage Robust Location-Transport Problem (TSRLTP) parameters to the standard form Two-Stage Robust Optimization parameter format."
function TSRLTP_standardize(f, a, c, K)
    m = size(c, 1) # number of potential facilities
    n = size(c, 2) # number of customers

    b = c[:]
    c = [f; a]
    A = [Diagonal(K) Diagonal(fill(-1, m))]
    d = zeros(m)
    Gu = []
    G = zeros(n, m*n)
    for i in 1:n
        G[i, (i-1)*m+1:i*m] .= 1
    end
    G = vcat(repeat(Diagonal(fill(-1,m)),1,n), G)
    h = zeros(m+n)
    E = zeros(m+n, 2m)
    E[1:m,m+1:end] = Diagonal(fill(1,m))
    M = zeros(m+n, n)
    M[m+1:end,:] = Diagonal(fill(-1,n))

    Sy = [ZeroOne(m); Rplus(m)]
    Sx = [Rplus(m*n)]

    return c, b, A, d, G, h, E, M, Sy, Sx
end


"Recover the result of the original Tow-Stage Robust Location-Transport Problem (TSRLTP) from the standard form Two-Stage Robust Optimization result."
function TSRLTP_recover(y)
    n = length(y)
    z = y[n÷2+1:end]
    y = y[1:n÷2]
    return y, z
end
## ----------------------------------------------------------------




## --- Case-5: Two-Stage Robust Rostering Problem ---
"Convert the general form Tow-Stage Robust Rostering Problem (TSRTP) parameters to the standard form Two-Stage Robust Optimization parameter format."
function TSRTP_standardize(I, J, N, T, c, f, h, M, l, u, a, b)
    m_ = I * T; m′_ = 0
    p_ = (J + 1) * T; n_ = J * T
    c_ = c[:]'
    d_ = [h[:]; M]'
    g_ = f[:]'
    
    A_ = []
    for t = 1:T-2
        push!(A_, [repeat(spzeros(I,I), 1, t-1) repeat(spdiagm(-ones(I)), 1, 3) repeat(spzeros(I,I), 1, T-2-t)])
    end
    A_ = vcat(A_...)
    A_ = [A_; repeat(spdiagm(ones(I)), 1, T); repeat(spdiagm(-ones(I)), 1, T)]
    b_ = [-2 .* ones(I*(T-2)); l; -u]
    
    E_ = spzeros(J*(T-1), p_)
    G_ = []
    for t = 1:T-1
        push!(G_, [repeat(spzeros(J,J), 1, t-1) repeat(spdiagm(-ones(J)), 1, 2) repeat(spzeros(J,J), 1, T-1-t)])
    end
    G_ = vcat(G_...)
    f_ = - ones(J*(T-1))
    R_ = spzeros(J*(T-1), T)
    D_ = spzeros(J*(T-1), m_)
    
    E_ = [E_; spzeros(2J, p_)]
    G_ = [G_; repeat(spdiagm(ones(J)), 1, T); -repeat(spdiagm(ones(J)), 1, T)]
    f_ = [f_; a; -b]
    R_ = [R_; spzeros(2J, T)]
    D_ = [D_; spzeros(2J, m_)]
    
    for t = 1:T
        E_ = [E_; repeat(spzeros(J,J), 1, t-1) spdiagm(-ones(J)) repeat(spzeros(J,J), 1, T-t) spzeros(J, T)]
    end
    G_ = [G_; spdiagm(N*ones(n_))]
    f_ = [f_; spzeros(n_)]
    R_ = [R_; spzeros(n_, T)]
    D_ = [D_; spzeros(n_, m_)]
    
    for t = 1:T
        E_ = [E_; repeat(spzeros(1,J), 1, t-1) ones(1, J) repeat(spzeros(1,J), 1, T-t) spzeros(1, T)]
    end
    E_[end-T+1:end, end-T+1:end] = spdiagm(ones(T))
    G_ = [G_; spzeros(T, n_)]
    f_ = [f_; spzeros(T)]
    R_ = [R_; spdiagm(-ones(T))]
    for t = 1:T
        D_ = [D_; repeat(spzeros(1,I), 1, t-1) ones(1, I) repeat(spzeros(1,I), 1, T-t)]
    end
    
    Sy_ = [ZeroOne(m_)]
    Sx_ = [Rplus(p_)]
    Sz_ = [ZeroOne(n_)]
    
    c = c_'
    b = [d_'; g_']
    A = A_
    d = b_
    G = [E_ G_]
    h = f_
    E = D_
    M = R_
    
    Sy = Sy_
    Sx = [Sx_; Sz_]
    
    return c, b, A, d, G, h, E, M, Sy, Sx
end

"Recover the result of the original Tow-Stage Robust Rostering Problem (TSRTP) from the standard form Two-Stage Robust Optimization result."
function TSRTP_recover(y, I, T)
    x = reshape(y, I, T)
    x = sparse(x)
    return x
end
## --------------------------------------------------