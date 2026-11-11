include("common.jl")
# exactness of ExtrapolationBC / MirrorBC(bounded=false) for random polynomials of the degree
function run_case(T, D, q, Rf, jit, kind; seed = 1, n = D == 3 ? 10 : (D == 2 ? 20 : 40), periodic = false)
    rng = MersenneTwister(seed)
    dx = T(1 / n)
    bc = kind == :ext ? ExtrapolationBC(q) : MirrorBC(; degree = q, bounded = false)
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D);
            is_periodic = ntuple(d -> periodic && d == 1, D), bc_map = BCD(t => bc for t in 1:2D))
    pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(T(1), T(Rf) * dx), 1;
                      randomness = ntuple(_ -> T(jit) * dx, D), rng = rng)
    f = randpoly(D, q, rng)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(T(f(Float64.(pg.core.positions[i])))); end
    buf = applybc(pg, 1, T)
    scale = maximum(abs(pg.rhos[i][1]) for i in interior(pg))
    errs = Float64[]
    for i in ghosts(pg)
        xt = kind == :ext ? Float64.(pg.core.positions[i]) : Float64.(mirrorpt(pg, i))
        push!(errs, abs(buf[i][1] - f(xt)) / scale)
    end
    tol = T == Float64 ? 1e-8 : 1e-3
    nbad = count(>(tol), errs)
    @printf("%-8s D=%d q=%d R=%.1f jit=%.2f %s per=%d: ghosts=%5d  max rel err=%.2e  #>%g: %d\n", T, D, q, Rf, jit, kind, periodic, length(errs), maximum(errs), tol, nbad)
    return pg, errs, buf, f
end
for D in (2, 3), kind in (:ext, :mir)
    for q in (D == 3 ? (1, 2) : (2, 3))
        run_case(Float32, D, q, q >= 3 ? 3.5 : 2.5, 0.2, kind)
    end
end
