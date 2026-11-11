include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
struct ExactBC <: HyperCloudCore.AbstractBoundaryCondition end
function (::ExactBC)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = State{1,Float64}(u0(pg.core.positions[i], t)))
    end
end
function run(bc, N, order, tend, jit)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(7))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
    e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg)]
    y = [pg.core.positions[i][2] for i in interior(pg)]
    maximum(e), maximum(e[k] for k in eachindex(e) if y[k] > 4h), sum(e) / length(e)
end
for jit in (0.0, 0.2), order in (3,)
    println("== MUSCL$order jitter $jit, t = 0.25: max err | max err for y > 4h | mean err")
    for (name, bc) in (("Exact", ExactBC()), ("Outflow", OutflowBC()), ("Mirror1 b", MirrorBC(; degree = 1)), ("Mirror2 b", MirrorBC(; degree = 2)), ("Mirror2 u", MirrorBC(; degree = 2, bounded = false)))
        r = [run(bc, N, order, 0.25, jit) for N in (20, 40, 80)]
        @printf("  %-10s max %s | far %s | mean %s\n", name, join([@sprintf("%.2e", x[1]) for x in r], " "), join([@sprintf("%.2e", x[2]) for x in r], " "), join([@sprintf("%.2e", x[3]) for x in r], " "))
    end
end
