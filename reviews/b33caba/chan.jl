# boundary_conditions.md Stability item 0: jittered channel (0.3h) t = 0.5 and off-lattice wall (0.1h, no jitter) t = 1,
# MUSCL3 unlimited, RK3, half the own dt; MirrorBC(degree 2) vs exact ghosts, N = 20/40/80
include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
struct ExactG{F} <: AbstractBoundaryCondition; u::F; end
function (b::ExactG)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = State{1, Float64}(b.u(pg.core.positions[i], t)))
    end
end
function run(N, which, exact; seed = 7, jit = 0.3)
    h = 1 / N
    if which == :jit
        u = (x, t) -> exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
        bc = exact ? ExactG(u) : MirrorBC(; degree = 2)
        geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
        pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit * h, jit * h), rng = MersenneTwister(seed))
        tend = 0.5
    else
        y0 = 0.1h
        u = (x, t) -> exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + (x[2] - y0)^2) / 0.02)
        bc = exact ? ExactG(u) : MirrorBC(; degree = 2)
        shape = intersect(Tagged(Cuboid((0.0, 0.0), (1.0, 1.0); open = (true, false)), 4), Tagged(HalfSpace((0.0, y0), (0.0, -1.0)), 3))
        pg = ParticleGrid(GeometricDomain(shape; is_periodic = (true, false), mins = (0.0, 0.0), maxs = (1.0, 1.0), bc_map = BCD(3 => bc, 4 => bc)),
                          (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
        tend = 1.0
    end
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
    maximum(abs(pg.rhos[i][1] - u(pg.core.positions[i], t)) for i in interior(pg))
end
for (which, seeds) in ((:jit, (7, 1, 2)), (:off, (0,)))
    for seed in seeds, exact in (false, true)
        e = [run(N, which, exact; seed) for N in (20, 40, 80)]
        @printf("%-4s seed %d %-6s %.3g / %.3g / %.3g\n", which, seed, exact ? "exact" : "mirror", e...)
        flush(stdout)
    end
end
