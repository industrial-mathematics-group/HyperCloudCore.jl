# KA CPU backend vs the default with ghosts in the gather (new): channel with MirrorBC, MUSCL3 + BJ limiter + MOODu2, 2-D, t = 0.2
include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x) = exp(-((x[1] - 0.5)^2 + x[2]^2) / 0.02)
function run(N)
    h = 1 / N
    bc = MirrorBC(; degree = 2)
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (0.2h, 0.2h), rng = MersenneTwister(2))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i])); end
    interp = MUSCL(Float64, 2, 1, 3, BarthJespersenLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(MOODu2(1e-3)), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < 0.2 - 1e-12; d = min(dt, 0.2 - t); ts(eqx, pg, t, d); t += d; end
    [pg.rhos[i][1] for i in 1:pg.meta.N], copy(interp.gradients[1:pg.meta.N]), pg
end
a, ga, pg = run(40)
set_backend!(HyperCloudCore.KA.CPU())
b, gb, _ = run(40)
G = ghosts(pg)
@printf("max |U_KA - U_default| = %.2e ; ghost gradients: max |g| %.2e, max diff %.2e\n", maximum(abs.(a .- b)),
        maximum(maximum(abs, reduce(vcat, ga[i])) for i in G), maximum(maximum(abs, reduce(vcat, ga[i] .- gb[i])) for i in G))
