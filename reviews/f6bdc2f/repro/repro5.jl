using HyperCloudCore, StaticArrays, LinearAlgebra
const H = HyperCloudCore
D_ = [parse.(Float32, split(l)) for l in eachline("/tmp/brev_b/repro/data.txt")]
x = SVector{3,Float32}(D_[1][1:3]); invL = D_[1][4]
xs = [SVector{3,Float32}(r[1:3]) for r in D_[2:end]]; ys = [SVector{1,Float32}(r[4]) for r in D_[2:end]]
wf = ExponentialWeightFunction(1.0f0, 0.25f0)
L = zero(x); Li = zero(x); idx = collect(2:length(xs)); isb = zeros(Bool, length(xs))
@inline basis(d::SVector{3,T}) where {T} = SVector{10,T}(one(T), d[1], d[2], d[3], d[1]^2/2, d[2]^2/2, d[3]^2/2, d[1]*d[2], d[1]*d[3], d[2]*d[3])
@inline wfun(d2::T) where {T} = (y = d2 * T(16); one(T) / (one(T) + y * (one(T) + y * (T(0.5) + y * (T(1/6) + y * T(1/24))))))
for (label, MI, BA, WF, CH) in (("all library pieces", H._min_image, d -> H._target_basis(Val(2), d), wf, H._cholesky_moment),
                              ("plain cholesky", H._min_image, d -> H._target_basis(Val(2), d), wf, N -> (cholesky(Symmetric(N), check = false), true)),
                              ("own basis", H._min_image, basis, wf, H._cholesky_moment),
                              ("own weight", H._min_image, d -> H._target_basis(Val(2), d), wfun, H._cholesky_moment),
                              ("no min image", (d, L, Li) -> d, d -> H._target_basis(Val(2), d), wf, H._cholesky_moment))
    @eval function trial(j, x, buf, rg, indices, pos, isb, L, Li, invL)
        N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
        r = $MI(pos[j] - x, L, Li); p = $BA(r*invL); w = $WF(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[j]')
        for k in rg
            m = indices[k]; isb[m] && continue
            r = $MI(pos[m] - x, L, Li); p = $BA(r*invL); w = $WF(sum(abs2, r))
            N_s += w*(p*p'); b_s += w*(p*buf[m]')
        end
        C, ok = $CH(N_s)
        (C \ b_s)[1], N_s
    end
    v, N = Base.invokelatest(trial, 1, x, ys, 1:length(idx), idx, xs, isb, L, Li, invL)
    println(rpad(label, 22), " value ", v, "  N symmetric: ", N == N')
end
