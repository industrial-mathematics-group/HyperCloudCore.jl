# Standalone, StaticArrays + LinearAlgebra only. Expected: N symmetric and value ≈ 3.3828 (the data are a quadratic).
using StaticArrays, LinearAlgebra
D_ = [parse.(Float32, split(l)) for l in eachline(joinpath(@__DIR__, "data.txt"))]
x = SVector{3,Float32}(D_[1][1:3]); invL = D_[1][4]
xs = [SVector{3,Float32}(r[1:3]) for r in D_[2:end]]; ys = [SVector{1,Float32}(r[4]) for r in D_[2:end]]
idx = collect(2:length(xs)); isb = zeros(Bool, length(xs)); L = zero(x); Li = zero(x)
@inline basis(d::SVector{3,T}) where {T} = SVector{10,T}(one(T), d[1], d[2], d[3], d[1]^2/2, d[2]^2/2, d[3]^2/2, d[1]*d[2], d[1]*d[3], d[2]*d[3])
@inline wfun(d2::T) where {T} = (y = d2 * T(16); one(T) / (one(T) + y * (one(T) + y * (T(0.5) + y * (T(1/6) + y * T(1/24))))))
@inline min_image(dx::SVector{D, T}, L::SVector{D, T}, Li::SVector{D, T}) where {D, T} = SVector{D, T}(ntuple(d -> dx[d] - L[d] * round(dx[d] * Li[d]), Val(D)))
@inline function pivots_ok(C, N_s::SMatrix{B, B, T}) where {B, T}
    LinearAlgebra.issuccess(C) || return false
    U = C.factors
    @inbounds for k in 1:B
        U[k, k]^2 > T(10) * eps(T) * N_s[k, k] || return false
    end
    return true
end
@inline function cholesky_moment(N_s::SMatrix{B, B, T}) where {B, T}
    C = cholesky(Symmetric(N_s), check = false)
    pivots_ok(C, N_s) && return C, true
    eps_reg = T(3.45e-4) * tr(N_s) / B
    C = cholesky(Symmetric(N_s + eps_reg * I), check = false)
    return C, LinearAlgebra.issuccess(C)
end
function trial(j, x, buf, rg, indices, pos, isb, L, Li, invL)
    N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
    r = min_image(pos[j] - x, L, Li); p = basis(r*invL); w = wfun(sum(abs2, r))
    N_s += w*(p*p'); b_s += w*(p*buf[j]')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = min_image(pos[m] - x, L, Li); p = basis(r*invL); w = wfun(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[m]')
    end
    C, ok = cholesky_moment(N_s)
    (C \ b_s)[1], N_s
end
v, N = trial(1, x, ys, 1:length(idx), idx, xs, isb, L, Li, invL)
println(VERSION, " ", Sys.CPU_NAME, ": value ", v, " (expected ≈ 3.3828), N symmetric: ", N == N')
