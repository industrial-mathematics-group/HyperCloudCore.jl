# Standalone (StaticArrays only): a weighted 10x10 normal matrix accumulated in a loop, then Cholesky-solved.
using StaticArrays, LinearAlgebra
D_ = [parse.(Float32, split(l)) for l in eachline("/tmp/brev_b/repro/data.txt")]; x0 = D_[1][1:3]; invL = D_[1][4]; xs0 = [r[1:3] for r in D_[2:end]]; ys0 = [r[4] for r in D_[2:end]]
xs = [SVector{3,Float32}(p) for p in xs0]; ys = [SVector{1,Float32}(y) for y in ys0]; x = SVector{3,Float32}(x0)
@inline basis(d::SVector{3,T}) where {T} = SVector{10,T}(one(T), d[1], d[2], d[3], d[1]^2/2, d[2]^2/2, d[3]^2/2, d[1]*d[2], d[1]*d[3], d[2]*d[3])
@inline wfun(d2::T) where {T} = (y = d2 * T(16); one(T) / (one(T) + y * (one(T) + y * (T(0.5) + y * (T(1/6) + y * T(1/24))))))
@inline minimg(d, L, Li) = d - L .* round.(d .* Li)
function mine(j, x, buf, rg, indices, pos, isb, L, Li, invL)
    N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
    r = minimg(pos[j] - x, L, Li); p = basis(r*invL); w = wfun(sum(abs2, r))
    N_s += w*(p*p'); b_s += w*(p*buf[j]')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = minimg(pos[m] - x, L, Li); p = basis(r*invL); w = wfun(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[m]')
    end
    C = cholesky(Symmetric(N_s), check = false)
    (C \ b_s)[1], N_s
end
function onlyN(j, x, rg, indices, pos, isb, L, Li, invL)
    N_s = zero(SMatrix{10,10,Float32,100})
    r = minimg(pos[j] - x, L, Li); p = basis(r*invL); w = wfun(sum(abs2, r)); N_s += w*(p*p')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = minimg(pos[m] - x, L, Li); p = basis(r*invL); w = wfun(sum(abs2, r)); N_s += w*(p*p')
    end
    N_s
end
idx = collect(2:length(xs)); isb = zeros(Bool, length(xs)); L = zero(x); Li = zero(x)
v, N = mine(1, x, ys, 1:length(idx), idx, xs, isb, L, Li, invL)
Nref = onlyN(1, x, 1:length(idx), idx, xs, isb, L, Li, invL)
println(VERSION, ": N from the solve function symmetric: ", N == N', ", equal to the plain accumulation: ", N == Nref,
        ", max |N - Nref| = ", maximum(abs, N - Nref))
