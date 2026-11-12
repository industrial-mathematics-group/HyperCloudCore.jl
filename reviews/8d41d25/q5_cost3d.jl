# Cost of the sweep's rounds: a plane piston in 3-D (and 2-D) advancing one row per sweep; time per sweep, neighbour search
include("common.jl")
f3(x) = 1 + 0.5 * sin(2π * x[1]) * cos(2π * x[2])
function piston(D, N; w = -1.0)
    h = 1 / N
    nrm = ntuple(d -> d == 1 ? 1.0 : 0.0, D)
    pc = Moving(Tagged(HalfSpace(ntuple(d -> d == 1 ? 1.0 : 0.0, D), nrm), 2D + 1), translating(ntuple(d -> d == 1 ? w : 0.0, D)))
    hi = ntuple(d -> d == 1 ? 2.0 : 1.0, D)
    box = Cuboid(ntuple(_ -> 0.0, D), hi; open = ntuple(d -> d > 1, D))
    geom = GeometricDomain(intersect(box, pc); is_periodic = ntuple(d -> d > 1, D), mins = ntuple(_ -> 0.0, D), maxs = hi,
                           bc_map = BCD(t => ExtrapolationBC(1) for t in 1:2D+1))
    pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f3(pg.core.positions[i])); end
    pg, h
end
for (D, N) in ((2, 40), (2, 80), (3, 12), (3, 16), (3, 24))
    pg, h = piston(D, N)
    tn = @elapsed pg.neighbor(pg)
    tn = @elapsed pg.neighbor(pg)
    t = 0.0
    # warm-up (compilation): a sweep without removal
    place_ghosts!(pg, pg.core.positions, 0.0, 0.0); sweep_boundaries!(pg, 0.0, 0.0, 3)
    tot, nrem, alloc, tmax = 0.0, 0, 0, 0.0
    nsweep = 10
    for k in 1:nsweep
        t += 0.25h
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        local r
        a = @allocated (el = @elapsed (r = sweep_boundaries!(pg, 0.0, t, 3)))
        tot += el; nrem += r.removed; alloc += a; tmax = max(tmax, el)
    end
    @printf("D=%d N=%d particles %d: neighbour search %.2f ms; %d sweeps (one row = %d particles every 4): %.1f ms mean, %.1f ms max, %d removed, %.1f MB per sweep\n",
            D, N, pg.meta.N, 1e3tn, nsweep, N^(D-1), 1e3tot / nsweep, 1e3tmax, nrem, alloc / nsweep / 2^20)
end
