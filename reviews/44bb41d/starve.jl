# Ghosts whose order the gather lowers (starved neighbourhood) and that are neighbours of interior particles (new)
include("common.jl")
function starve(D, N, order, Rf)
    h = 1 / N
    bc = OutflowBC()
    geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = BCD((k => bc for k in 1:2D)...))
    pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, Rf * h), 1)
    for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{1,Float64}(exp(-sum(abs2, x .- 0.5) / 0.02)); end
    eq = LinearAdvection(ntuple(d -> (1.0,), D), Float64)
    interp = MUSCL(Float64, D, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    ts(eq, pg, 0.0, 0.1 * get_time_step(pg, eq, interp))
    G = ghosts(pg); o = pg.core.particle_orders
    nbint = falses(pg.meta.N)
    for i in interior(pg), k in pg.neighbor.ranges[i]; nbint[pg.neighbor.indices[k]] = true; end
    low = [i for i in G if o[i] < order]
    lowint = [i for i in low if nbint[i]]
    @printf("%dD N=%d MUSCL%d R=%.1fh: ghosts %d, lowered %d (orders %s), of which neighbours of interior particles %d; interior lowered %d\n",
            D, N, order, Rf, length(G), length(low), string(sort(unique(o[low]))), length(lowint), count(i -> o[i] < order, interior(pg)))
end
starve(1, 80, 5, 2.5); starve(1, 80, 5, 3.5); starve(1, 80, 3, 2.5); starve(2, 40, 5, 2.5); starve(2, 40, 3, 2.5); starve(3, 16, 3, 2.5)
