include("common.jl")
eq = LinearAdvection(((1.0,), (0.3,)), Float64)
u0(x, t) = exp(-((x[1] - 0.5 - t)^2 + (x[2] - 0.5 - 0.3t)^2) / 0.02)
function run(bc, N, order, tend)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); bc_map = BCD(1 => FixedDirichlet(), 3 => FixedDirichlet(), 2 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (0.2h, 0.2h), rng = MersenneTwister(7))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eq, interp); t = 0.0
    while t < tend - 1e-12
        d = min(dt, tend - t); ts(eq, pg, t, d); t += d
        maximum(abs(pg.rhos[i][1]) for i in 1:pg.meta.N) > 1e20 && break
    end
    maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg))
end
for N in (20, 40), order in (3, 5)
    r = [(name, run(bc, N, order, 2.0)) for (name, bc) in (("Outflow", OutflowBC()), ("Extrap1", ExtrapolationBC(1)), ("Extrap2", ExtrapolationBC(2)))]
    println("N=$N MUSCL$order t=2 max err: ", join([@sprintf("%s %.2e", a, b) for (a, b) in r], ", "))
end
