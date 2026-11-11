include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
function run(bc, N, order, tend; jit = 0.2)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(7))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12
        d = min(dt, tend - t); ts(eqx, pg, t, d); t += d
        maximum(abs(pg.rhos[i][1]) for i in 1:pg.meta.N) > 1e6 && break
    end
    err = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg))
    err, maximum(abs(pg.rhos[i][1]) for i in interior(pg))
end
mode = ARGS[1]
if mode == "stab"
    for N in (20, 40), order in (3, 5)
        for (name, bc) in (("Outflow", OutflowBC()), ("Mirror1 b", MirrorBC(; degree = 1)), ("Mirror2 b", MirrorBC(; degree = 2)), ("Mirror3 b", MirrorBC(; degree = 3)),
                           ("Mirror1 u", MirrorBC(; degree = 1, bounded = false)), ("Mirror2 u", MirrorBC(; degree = 2, bounded = false)), ("Extrap1", ExtrapolationBC(1)))
            e, m = run(bc, N, order, 2.0)
            @printf("N=%d MUSCL%d %-10s t=2: err %.3e  max|u| %.3f\n", N, order, name, e, m)
        end
    end
elseif mode == "conv"   # boundary-driven accuracy at t = 0.25 (bump on the wall), MUSCL3 / MUSCL5
    for order in (3, 5)
        for (name, bc) in (("Outflow", OutflowBC()), ("Mirror1 b", MirrorBC(; degree = 1)), ("Mirror2 b", MirrorBC(; degree = 2)), ("Mirror2 u", MirrorBC(; degree = 2, bounded = false)), ("Mirror3 b", MirrorBC(; degree = 3)))
            es = [run(bc, N, order, 0.25)[1] for N in (20, 40, 80)]
            @printf("MUSCL%d %-10s errs %s EOC %.2f %.2f\n", order, name, join([@sprintf("%.2e", e) for e in es], " "), log2(es[1]/es[2]), log2(es[2]/es[3]))
        end
    end
end
