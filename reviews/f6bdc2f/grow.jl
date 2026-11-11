include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
struct ExactBC <: HyperCloudCore.AbstractBoundaryCondition end
function (::ExactBC)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = State{1,Float64}(u0(pg.core.positions[i], t)))
    end
end
function run(name, bc, N, order, tend, jit)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(7))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    out = String[]
    for tk in (0.25, 0.5, 1.0)
        tk > tend && break
        while t < tk - 1e-12; d = min(dt, tk - t); ts(eqx, pg, t, d); t += d; end
        e = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg))
        m = maximum(abs(pg.rhos[i][1]) for i in interior(pg))
        eg = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in ghosts(pg) if pg.core.positions[i][2] < 0.5)
        push!(out, @sprintf("t=%.2f err %.2e max %.3f ghost err %.2e", t, e, m, eg))
    end
    println(rpad("$name N=$N", 18), join(out, " | "))
end
N = parse(Int, ARGS[1])
for (name, bc) in (("Exact", ExactBC()), ("Outflow", OutflowBC()), ("Mirror1 b", MirrorBC(; degree = 1)), ("Mirror2 b", MirrorBC(; degree = 2)))
    run(name, bc, N, 3, 1.0, 0.2)
end
