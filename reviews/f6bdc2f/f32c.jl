include("common.jl")
const H = HyperCloudCore
T = Float32; q = 2; n = 10; dx = T(1/n)
geom = get_rectangular_domain(T, (0.,0.,0.), (1.,1.,1.); bc_map = BCD(t => ExtrapolationBC(q) for t in 1:6))
pg = ParticleGrid(geom, (dx,dx,dx), ExponentialWeightFunction(T(1), T(2.5)*dx), 1; randomness = (T(0.2)*dx, T(0.2)*dx, T(0.2)*dx), rng = MersenneTwister(1))
f(x) = 1 + x[1] + x[2] + x[3] + x[1]^2 - x[2]*x[3]
for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(T(f(Float64.(pg.core.positions[i])))); end
nb = pg.neighbor; isb = pg.core.is_boundary; pos = pg.core.positions
L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
i = 713; x = pos[i]
j, best = H._nearest_interior(i, x, nb.ranges, nb.indices, isb, pos, L, Li)
@show i j
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
invL = one(T)/minimum(pg.meta.dx)
u1, N1, b1 = mine(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
U, ok = H._fit_degree_b(Val(9), Val(2), j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
@show u1 U ok
C, ok2 = H._cholesky_moment(N1)
@show (C \ b1)[1]
for _ in 1:3
    j2, b2_ = H._nearest_interior(j, x, nb.ranges, nb.indices, isb, pos, L, Li)
    (j2 != 0 && b2_ < best) || break; global j, best = j2, b2_
end
invL = one(T)/minimum(pg.meta.dx)
@inline mybasis(r::SVector{3,T}) where {T} = SVector{10,T}(one(T), r[1], r[2], r[3], r[1]^2/2, r[2]^2/2, r[3]^2/2, r[1]*r[2], r[1]*r[3], r[2]*r[3])
function generic(bf, j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)
    N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
    r = H._min_image(pos[j] - x, L, Li); p = bf(r*invL); w = wf(sum(abs2, r))
    N_s += w*(p*p'); b_s += w*(p*buf[j]')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = bf(r*invL); w = wf(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[m]')
    end
    C, ok = H._cholesky_moment(N_s)
    (C \ b_s)[1], maximum(abs, N_s - N_s'), ok
end
println("lib basis: ", generic(r -> H._target_basis(Val(2), r), j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL))
println("my basis : ", generic(mybasis, j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL))
xs = vcat([pos[j]], [pos[nb.indices[k]] for k in nb.ranges[j] if !isb[nb.indices[k]]])
ys = vcat([pg.rhos[j]], [pg.rhos[nb.indices[k]] for k in nb.ranges[j] if !isb[nb.indices[k]]])
idx = collect(2:length(xs)); isb2 = falses(length(xs)); 
println("plain arrays, lib basis: ", generic(r -> H._target_basis(Val(2), r), 1, x, ys, 1:length(idx), idx, xs, isb2, L, Li, nb.weight_func, invL))
println("plain arrays, my basis: ", generic(mybasis, 1, x, ys, 1:length(idx), idx, xs, isb2, L, Li, nb.weight_func, invL))
using Serialization; serialize("/tmp/brev_b/repro/data.jls", (xs, ys, x, invL, nb.weight_func))
@show j
println("generic lib (this j): ", generic(r -> H._target_basis(Val(2), r), j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL))
println("generic lib (716): ", generic(r -> H._target_basis(Val(2), r), 716, x, pg.rhos, nb.ranges[716], nb.indices, pos, isb, L, Li, nb.weight_func, invL))
println("mine (716): ", mine(716, x, pg.rhos, nb.ranges[716], nb.indices, pos, isb, L, Li, nb.weight_func, invL)[1])
println("mine (this j): ", mine(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)[1])
println("_fit_degree_b (this j): ", H._fit_degree_b(Val(9), Val(2), j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL))
println("exact ", f(Float64.(x)), "  dist donors/dx: ", norm(pos[716]-x)/dx, " ", norm(pos[j]-x)/dx)
println("_support_count 716: ", H._support_count(716, nb.ranges, nb.indices, isb), "  j: ", H._support_count(j, nb.ranges, nb.indices, isb))
