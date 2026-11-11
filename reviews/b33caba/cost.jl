# ms per step (min of reps) with walls: old (ghosts not gathered) / new (all ghosts) / mask (only ghosts with an interior neighbour)
include("common.jl")
function bench(D, N, order; reps = 30)
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
    tmed = sort([begin t0 = time_ns(); ts(eq, pg, 0.0, dt); (time_ns() - t0) / 1e6 end for _ in 1:reps])[reps ÷ 2]
    ng = count(view(pg.core.is_boundary, 1:pg.meta.N))
    used = falses(pg.meta.N)
    for i in interior(pg), k in pg.neighbor.ranges[i]; used[pg.neighbor.indices[k]] = true; end
    nu = count(i -> !used[i], ghosts(pg))
    chk = sum(i -> pg.rhos[i][1], interior(pg))
    @printf("%dD N=%3d MUSCL%d  particles %6d ghosts %5d (%.0f %%, unread %d)  min %.2f med %.2f ms/step  chk %.15e\n", D, N, order, pg.meta.N, ng, 100ng / pg.meta.N, nu, tm, tmed, chk)
    flush(stdout)
end
println("threads: ", Threads.nthreads(), "  tree: ", pathof(HyperCloudCore))
bench(2, 160, 3); bench(2, 80, 5); bench(3, 20, 3); bench(3, 32, 3); bench(3, 16, 5)
