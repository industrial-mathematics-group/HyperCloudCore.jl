include("common.jl")
const H = HyperCloudCore
T = Float32; D = 3; q = 2; n = 10; dx = T(1/n)
geom = get_rectangular_domain(T, (0.,0.,0.), (1.,1.,1.); bc_map = BCD(t => ExtrapolationBC(q) for t in 1:6))
pg = ParticleGrid(geom, (dx,dx,dx), ExponentialWeightFunction(T(1), T(2.5)*dx), 1; randomness = (T(0.2)*dx, T(0.2)*dx, T(0.2)*dx), rng = MersenneTwister(1))
f(x) = 1 + x[1] + x[2] + x[3] + x[1]^2 - x[2]*x[3]
for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(T(f(Float64.(pg.core.positions[i])))); end
buf = applybc(pg, 1, T)
g = ghosts(pg)
errs = [abs(buf[i][1] - f(Float64.(pg.core.positions[i]))) for i in g]
p = sortperm(errs; rev = true)
@printf("ghosts %d, #err>1e-2: %d, max %.3g\n", length(g), count(>(1e-2), errs), errs[p[1]])
nb = pg.neighbor; isb = pg.core.is_boundary; pos = pg.core.positions
L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
for a in p[1:4]
    i = g[a]; x = pos[i]
    j, best = H._nearest_interior(i, x, nb.ranges, nb.indices, isb, pos, L, Li)
    if j == 0
        for k in nb.ranges[i]
            j2, b2 = H._nearest_interior(nb.indices[k], x, nb.ranges, nb.indices, isb, pos, L, Li)
            (j2 != 0 && b2 < best) && (j = j2; best = b2)
        end
    end
    for _ in 1:3
        j2, b2 = H._nearest_interior(j, x, nb.ranges, nb.indices, isb, pos, L, Li)
        (j2 != 0 && b2 < best) || break; j, best = j2, b2
    end
    nsup = H._support_count(j, nb.ranges, nb.indices, isb)
    # build N_s in Float32 and Float64
    function Ns(TT)
        B = 10; Nm = zeros(TT, B, B)
        for m in vcat([j], [nb.indices[k] for k in nb.ranges[j] if !isb[nb.indices[k]]])
            r = TT.(pos[m] - x) / TT(dx)
            pb = TT[1, r[1], r[2], r[3], r[1]^2, r[1]*r[2], r[1]*r[3], r[2]^2, r[2]*r[3], r[3]^2]
            w = TT(pg.neighbor.weight_func(sum(abs2, pos[m] - x)))
            Nm .+= w .* (pb * pb')
        end
        Nm
    end
    N64 = Ns(Float64)
    N32s = SMatrix{10,10,Float32}(Ns(Float32))
    C, ok = H._cholesky_moment(N32s)
    piv = H._pivots_ok(cholesky(Symmetric(N32s), check=false), N32s)
    @printf("ghost %d: φ/dx=%.2f err=%.3g donor dist/dx=%.2f nsup=%d needed=%d cond64=%.2e  chol ok=%s pivots_ok=%s\n",
            i, pg.core.wall_distances[i]/dx, errs[a], sqrt(best)/dx, nsup, H._fit_support(3, q), cond(N64), ok, piv)
end
println("L=", L, " Li=", Li)
for a in p[1:4]
    i = g[a]; x = pos[i]
    # replicate donor quickly via kernel call on a single particle
    b2 = copy(pg.rhos)
    H._ghost_fit_body!(i, b2, pos, isb, pg.core.tags, pg.core.normals, pg.core.wall_distances, pg.core.particle_orders,
        nb.ranges, nb.indices, nb.weight_func, L, Li, one(T)/minimum(pg.meta.dx), pg.core.tags[i], Val(q), nothing, nothing, Val(false), Val(false), Val(3))
    @printf("ghost %d serial kernel value %.6g, buffer value %.6g, exact %.6g  rhos[i]=%.4g\n", i, b2[i][1], buf[i][1], f(Float64.(x)), pg.rhos[i][1])
end
println("--- direct fit")
for a in p[1:2]
    i = g[a]; x = pos[i]
    j, best = H._nearest_interior(i, x, nb.ranges, nb.indices, isb, pos, L, Li)
    if j == 0
        for k in nb.ranges[i]
            j2, b2_ = H._nearest_interior(nb.indices[k], x, nb.ranges, nb.indices, isb, pos, L, Li)
            (j2 != 0 && b2_ < best) && (j = j2; best = b2_)
        end
    end
    for _ in 1:3
        j2, b2_ = H._nearest_interior(j, x, nb.ranges, nb.indices, isb, pos, L, Li)
        (j2 != 0 && b2_ < best) || break; j, best = j2, b2_
    end
    invL = one(T)/minimum(pg.meta.dx)
    for bb in (buf, pg.rhos)
        U, ok = H._fit_degree(Val(q), j, x, bb, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
        @printf("  _fit_degree U=%.6g ok=%s\n", U[1], ok)
    end
    r = H._min_image(pos[j] - x, L, Li) * invL
    @show H._target_basis(Val(q), r)
    # my LS
    for TT in (Float32, Float64)
        rows = []; ws = TT[]; ys = TT[]
        for m in vcat([j], [nb.indices[k] for k in nb.ranges[j] if !isb[nb.indices[k]]])
            rr = TT.(pos[m] - x) .* TT(invL)
            push!(rows, TT[1, rr[1], rr[2], rr[3], rr[1]^2, rr[1]*rr[2], rr[1]*rr[3], rr[2]^2, rr[2]*rr[3], rr[3]^2])
            push!(ws, TT(nb.weight_func(sum(abs2, pos[m]-x)))); push!(ys, TT(pg.rhos[m][1]))
        end
        A = reduce(vcat, permutedims.(rows)); W = Diagonal(ws)
        c = (A'*W*A) \ (A'*W*ys)
        @printf("  my LS %s: %.6g   (exact %.6g)\n", TT, c[1], f(Float64.(x)))
    end
end
println("--- static cholesky")
let a = p[1]
    i = g[a]; x = pos[i]
    j = 0; best = T(Inf)
    j, best = H._nearest_interior(i, x, nb.ranges, nb.indices, isb, pos, L, Li)
    if j == 0
        for k in nb.ranges[i]
            j2, b2_ = H._nearest_interior(nb.indices[k], x, nb.ranges, nb.indices, isb, pos, L, Li)
            (j2 != 0 && b2_ < best) && (j = j2; best = b2_)
        end
    end
    for _ in 1:3
        j2, b2_ = H._nearest_interior(j, x, nb.ranges, nb.indices, isb, pos, L, Li)
        (j2 != 0 && b2_ < best) || break; j, best = j2, b2_
    end
    invL = one(T)/minimum(pg.meta.dx)
    N_s = zero(SMatrix{10,10,T}); b_s = zero(SMatrix{10,1,T})
    for m in vcat([j], [nb.indices[k] for k in nb.ranges[j] if !isb[nb.indices[k]]])
        r = H._min_image(pos[m] - x, L, Li); pb = H._target_basis(Val(q), r*invL); w = nb.weight_func(sum(abs2, r))
        N_s += w*(pb*pb'); b_s += w*(pb*pg.rhos[m]')
    end
    C, ok = H._cholesky_moment(N_s)
    @show ok (C \ b_s)[1] (Matrix(N_s) \ Matrix(b_s))[1] (Float64.(Matrix(N_s)) \ Float64.(Matrix(b_s)))[1]
    @show typeof(C) maximum(abs, Matrix(C.U)'*Matrix(C.U) - Matrix(N_s)) / maximum(abs, N_s)
    @show H._mls_reg(T) H._pivot_tol(T)
    C2 = cholesky(Symmetric(N_s), check=false)
    @show LinearAlgebra.issuccess(C2) (C2 \ b_s)[1]
    @show minimum(eigvals(Symmetric(Float64.(Matrix(N_s))))) maximum(eigvals(Symmetric(Float64.(Matrix(N_s)))))
end
