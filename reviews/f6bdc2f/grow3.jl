include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
struct Hybrid{B} <: HyperCloudCore.AbstractBoundaryCondition; bc::B; thr::Float64; near_exact::Bool; end
function (h::Hybrid)(pg, buf, tag, ts, eq, t)
    h.bc(pg, buf, tag, ts, eq, t)
    dx = pg.meta.dx[1]
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) || continue
        near = pg.core.wall_distances[i] < h.thr * dx
        (near == h.near_exact) && (buf[i] = State{1,Float64}(u0(pg.core.positions[i], t)))
    end
end
function run(name, bc, N, tend, jit; order = 3)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(7))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
    e = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg))
    @printf("%-34s N=%d jit=%.2f t=%.2f: max err %.3e\n", name, N, jit, t, e)
end
m2 = MirrorBC(; degree = 2)
run("Mirror2, exact for φ < 0.5h", Hybrid(m2, 0.5, true), 80, 0.5, 0.2)
run("exact, Mirror2 only for φ < 0.5h", Hybrid(m2, 0.5, false), 80, 0.5, 0.2)
run("Mirror2, exact for φ < 1.5h", Hybrid(m2, 1.5, true), 80, 0.5, 0.2)
run("exact, Mirror2 only for φ >= 1.5h", Hybrid(m2, 1.5, true), 80, 0.5, 0.2)
run("Mirror0", MirrorBC(; degree = 0), 80, 0.5, 0.2)
run("Extrap2 (as wall)", ExtrapolationBC(2), 80, 0.5, 0.2)
