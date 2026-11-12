# M2: _sync_boundaries! — does it fire on ordinary steps (Float32, times computed as n*dt), substep counts
include("common.jl")
plate_shape32 = Tagged(Cuboid((-0.2f0, -0.05f0), (0.2f0, 0.05f0)), t -> t + 4)
function mk32(h)
    chan  = Cuboid((-1f0, -0.5f0), (1f0, 0.5f0); open = (true, false)) - Moving(plate_shape32, translating((1f0, 0f0)))
    geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1f0, -0.5f0), maxs = (1f0, 0.5f0),
                            bc_map = BCD(t => MirrorBC(; degree = 1) for t in (3, 4, 5, 6, 7, 8)))
    pg = ParticleGrid(geom, (Float32(h), Float32(h)), ExponentialWeightFunction(1f0, Float32(2.5h)), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float32}(1f0); end
    pg
end
h = 1/20
eq = LinearAdvection(((1f0,), (0f0,)), Float32)
interp = MUSCL(Float32, 2, 1, 3, NoLimiter(), RusanovFlux())
# count sync triggers by checking the condition before each step, as the stepper does
function would_sync(pg, t)
    T = Float32; t = T(t)
    tol = 8 * eps(T) * max(one(T), abs(t))
    !(abs(pg.meta.boundary_time - t) <= tol && abs(pg.meta.sweep_time - t) <= tol)
end
for mode in (:accumulate64, :ndt64, :accumulate32)
    pg = mk32(h)
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float32))
    dt = Float32(0.1h); fired = 0; t64 = 0.0; t32 = 0f0
    for n in 1:150
        t = mode == :accumulate64 ? t64 : mode == :ndt64 ? (n - 1) * Float64(dt) : t32
        fired += would_sync(pg, t)
        ts(eq, pg, t, dt)
        t64 += dt; t32 += dt
    end
    @printf("Float32 plate, %s: sync would fire on %d of 150 steps; final boundary_time %.7f, ΣV %.6f (1.96)\n", mode, fired, pg.meta.boundary_time, sumV(pg))
end
# a start at t0 = 0.3 then 0.2 more vs a start at 0: same final state?
eq64 = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = 1 + exp(-((mod(x[1] - t + 1, 2.0) - 0.5)^2 + (x[2] - 0.2)^2) / 0.02)
plate_shape = Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4)
function run(N, t0, tend)
    h = 1 / N
    chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - Moving(plate_shape, translating((1.0, 0.0)))
    geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                            bc_map = BCD(t => MirrorBC(; degree = 1) for t in (3, 4, 5, 6, 7, 8)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], t0)); end
    it = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq64, it, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eq64, it); t = t0
    el = @elapsed ts(eq64, pg, t, min(dt, tend - t))
    t += min(dt, tend - t)
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq64, pg, t, d); t += d; end
    inner = interior(pg)
    e = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner)
    @printf("  N=%d start t0=%.2f -> %.2f: first step %.2f s, max err %.4g, ΣV %.6f\n", N, t0, tend, el, e, sumV(pg))
end
for N in (20, 40); run(N, 0.0, 0.5); run(N, 0.3, 0.5); end
# substep counts of a start far from t = 0 for a slowly spinning small ball (region invariant)
ball = Moving(Tagged(Ball((0.0, 0.0), 0.1), 5), rotating(2.0))
geom = GeometricDomain(Cuboid((-1.0, -1.0), (1.0, 1.0)) - ball; mins = (-1.0, -1.0), maxs = (1.0, 1.0), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
pg = ParticleGrid(geom, (1/40, 1/40), ExponentialWeightFunction(1.0, 2.5/40), 1)
for t in (0.1, 1.0, 5.0)
    @printf("spinning ball (ω = 2, r = 0.1), start at t0 = %.1f: %d sync sub-steps (the ball itself moves %.3f spacings per unit... at its rim)\n",
            t, H._sync_substeps(pg, 0.0, t), 2 * 0.1 * 40)
end
