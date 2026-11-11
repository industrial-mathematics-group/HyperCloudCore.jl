include("common.jl")
function grid(T, D, Rf, jit, bc; n = D == 3 ? 10 : (D == 2 ? 20 : 40), seed = 1, periodic = false, wf = :exp)
    dx = T(1 / n)
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D);
            is_periodic = ntuple(d -> periodic && d == 1, D), bc_map = BCD(t => bc for t in (periodic ? (3:2D) : (1:2D))))
    w = wf == :exp ? ExponentialWeightFunction(T(1), T(Rf) * dx) : InverseWeightFunction(T(1), T(Rf) * dx)
    ParticleGrid(geom, ntuple(_ -> dx, D), w, 1; randomness = ntuple(_ -> T(jit) * dx, D), rng = MersenneTwister(seed)), dx
end
# effective degree per ghost: largest k with all-monomials-up-to-k exact
function effdeg(pg, T, D, Q, kind; skipx = false)
    rng = MersenneTwister(5)
    ng = length(ghosts(pg)); eff = fill(-1, ng); errs = zeros(ng, Q+1)
    for k in 0:Q
        f0 = randpoly(D - (skipx ? 1 : 0), k, rng)
        f = skipx ? (x -> f0(x[2:end])) : f0
        for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(T(f(Float64.(pg.core.positions[i])))); end
        buf = applybc(pg, 1, T)
        sc = maximum(abs(pg.rhos[i][1]) for i in interior(pg))
        for (a, i) in enumerate(ghosts(pg))
            xt = kind == :ext ? Float64.(pg.core.positions[i]) : Float64.(mirrorpt(pg, i))
            errs[a, k+1] = abs(buf[i][1] - f(xt)) / sc
        end
    end
    errs
end
tolT(T) = T == Float64 ? 1e-8 : 1e-3
println("== Float32 extrapolation: error vs ghost wall distance (q=2, 3-D; q=3, 2-D)")
for (D, q, Rf) in ((3, 2, 2.5), (2, 3, 3.5), (2, 2, 2.5))
    pg, dx = grid(Float32, D, Rf, 0.2, ExtrapolationBC(q))
    e = effdeg(pg, Float32, D, q, :ext)
    g = ghosts(pg); φ = [pg.core.wall_distances[i] / dx for i in g]
    for band in ((0, 1), (1, 2), (2, 3), (3, 10))
        sel = findall(x -> band[1] < x <= band[2], φ)
        isempty(sel) && continue
        @printf("D=%d q=%d φ/dx in (%d,%d]: n=%4d  max err at degree q: %.2e  median %.2e; at degree q-1: %.2e\n", D, q, band..., length(sel),
                maximum(e[sel, q+1]), sort(e[sel, q+1])[(length(sel)+1)÷2], maximum(e[sel, q]))
    end
    pg64, _ = grid(Float64, D, Rf, 0.2, ExtrapolationBC(q))
    e64 = effdeg(pg64, Float64, D, q, :ext)
    @printf("  same in Float64: max err %.2e\n", maximum(e64[:, q+1]))
end
println("== degree actually reached in 1-D (exactness by degree), ExtrapolationBC/MirrorBC(Q), R/dx sweep")
for Q in (3, 4, 5), Rf in (2.5, 3.5, 4.5, 5.5), kind in (:ext, :mir)
    bc = kind == :ext ? ExtrapolationBC(Q) : MirrorBC(; degree = Q, bounded = false)
    pg, dx = grid(Float64, 1, Rf, 0.2, bc)
    e = effdeg(pg, Float64, 1, Q, kind)
    reached = [something(findlast(k -> all(e[a, 1:k+1] .< 1e-8), 0:Q), 0) - 1 for a in axes(e, 1)]
    @printf("1-D Q=%d R=%.1fdx %s: degrees reached per ghost %s\n", Q, Rf, kind, string(reached))
end
println("== periodic channel (poly in y,z only)")
for D in (2, 3), kind in (:ext, :mir)
    bc = kind == :ext ? ExtrapolationBC(2) : MirrorBC(; degree = 2, bounded = false)
    pg, dx = grid(Float64, D, 2.5, 0.2, bc; periodic = true)
    e = effdeg(pg, Float64, D, 2, kind; skipx = true)
    @printf("D=%d %s periodic: max err degree 2: %.2e\n", D, kind, maximum(e[:, 3]))
end
println("== InverseWeightFunction, unjittered (mirror point on a particle)")
for T in (Float64, Float32), kind in (:ext, :mir)
    bc = kind == :ext ? ExtrapolationBC(2) : MirrorBC(; degree = 2, bounded = false)
    pg, dx = grid(T, 2, 2.5, 0.0, bc; wf = :inv)
    e = effdeg(pg, T, 2, 2, kind)
    @printf("%s %s inverse weight unjittered: max err degree 2: %.2e\n", T, kind, maximum(e[:, 3]))
    pg, dx = grid(T, 2, 2.5, 0.2, bc; wf = :inv)
    e = effdeg(pg, T, 2, 2, kind)
    @printf("%s %s inverse weight jittered: max err degree 2: %.2e\n", T, kind, maximum(e[:, 3]))
end
