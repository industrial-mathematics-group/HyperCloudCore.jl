using HyperCloudCore, StaticArrays, LinearAlgebra, Serialization
const H = HyperCloudCore
xs, ys, x, invL, wf = deserialize("/tmp/brev_b/repro/data.jls")
L = zero(x); Li = zero(x)
idx = collect(2:length(xs)); isb = zeros(Bool, length(xs))
function mine(j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)
    N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
    r = H._min_image(pos[j] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
    N_s += w*(p*p'); b_s += w*(p*buf[j]')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[m]')
    end
    C, ok = H._cholesky_moment(N_s)
    (C \ b_s)[1], N_s, b_s
end
v, N, b = mine(1, x, ys, 1:length(idx), idx, xs, isb, L, Li, wf, invL)
println("value ", v, "   N symmetric: ", N == N', "   F64 solve of the same data: ",
        (Float64.(Matrix(N + N') / 2) \ Float64.(Matrix(b)))[1])
U, ok = H._fit_degree_b(Val(9), Val(2), 1, x, ys, 1:length(idx), idx, xs, isb, L, Li, wf, invL)
println("library _fit_degree_b: ", U[1], " ok=", ok)
