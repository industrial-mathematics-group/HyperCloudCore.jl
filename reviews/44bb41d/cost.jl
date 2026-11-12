# Cost per step with the ghosts in the gather (old = fbb54d1 vs mid = a56944c); MUSCL3/5, 2-D and 3-D boxes with walls
include("common.jl")
function bench(D, N, order; reps = 20)
    h = 1 / N
    bc = OutflowBC()
    geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = BCD((k => bc for k in 1:2D)...))
    pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, (order == 5 && D == 3 ? 3.0 : 2.5) * h), 1)
    for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{1,Float64}(exp(-sum(abs2, x .- 0.5) / 0.02)); end
    eq = LinearAdvection(ntuple(d -> (1.0,), D), Float64)
    interp = MUSCL(Float64, D, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.2 * get_time_step(pg, eq, interp)
    for _ in 1:3; ts(eq, pg, 0.0, dt); end
    tm = minimum(begin t0 = time_ns(); ts(eq, pg, 0.0, dt); (time_ns() - t0) / 1e6 end for _ in 1:reps)
    ng = count(view(pg.core.is_boundary, 1:pg.meta.N))
    @printf("%dD N=%3d MUSCL%d  particles %6d ghosts %5d (%.0f %%)  %.2f ms/step\n", D, N, order, pg.meta.N, ng, 100ng / pg.meta.N, tm)
    flush(stdout)
end
println("threads: ", Threads.nthreads())
bench(2, 80, 3); bench(2, 160, 3); bench(2, 80, 5); bench(3, 20, 3); bench(3, 32, 3); bench(3, 16, 5)
