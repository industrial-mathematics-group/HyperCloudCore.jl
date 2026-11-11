include("common.jl")
# 1-D: a bump leaving through the right wall (t = 0.6); left wall FixedDirichlet (zero inflow)
u0(x, t) = exp(-(x - 0.5 - t)^2 / 0.01)
function run1d(bc, N, order, Rf; jit = 0.0, T = Float64)
    h = 1 / N
    geom = get_rectangular_domain(T, (0.0,), (1.0,); bc_map = BCD(1 => FixedDirichlet(), 2 => bc))
    pg = ParticleGrid(geom, (T(h),), ExponentialWeightFunction(T(1), T(Rf * h)), 1; randomness = (T(jit * h),), rng = MersenneTwister(3))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(T(u0(pg.core.positions[i][1], 0.0))); end
    eq = LinearAdvection(((1.0,),), T)
    interp = MUSCL(T, 1, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK4_SSP_Tableau(T))
    dt = T(0.5) * get_time_step(pg, eq, interp); t = zero(T); tend = T(0.6)
    while t < tend - 1e-12
        d = min(dt, tend - t); ts(eq, pg, t, d); t += d
    end
    maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i][1], t)) for i in interior(pg))
end
for Rf in (3.5, 4.5), jit in (0.0, 0.2)
    println("== MUSCL5, R = $(Rf)dx, jitter $jit")
    for (name, bc) in (("OutflowBC", OutflowBC()), ("Extrap(1)", ExtrapolationBC(1)), ("Extrap(3)", ExtrapolationBC(3)), ("Extrap(4)", ExtrapolationBC(4)))
        es = [run1d(bc, N, 5, Rf; jit = jit) for N in (50, 100, 200, 400)]
        eoc = [log2(es[k] / es[k+1]) for k in 1:3]
        @printf("  %-10s errs %s  EOC %s\n", name, join([@sprintf("%.2e", e) for e in es], " "), join([@sprintf("%.2f", e) for e in eoc], " "))
    end
end
