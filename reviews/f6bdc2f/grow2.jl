include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
function run(name, bc, N, tend, jit; order = 3, show = false)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(7))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
    I = interior(pg)
    e = [pg.rhos[i][1] - u0(pg.core.positions[i], t) for i in I]
    k = argmax(abs.(e))
    @printf("%-10s N=%d jit=%.2f t=%.2f: max err %.3e at x=%s (value %.3f, exact %.3f)\n", name, N, jit, t, abs(e[k]), string(round.(pg.core.positions[I[k]], digits = 4)), pg.rhos[I[k]][1], u0(pg.core.positions[I[k]], t))
    if show   # error by row distance from the wall
        for (a, b) in ((0, 0.5), (0.5, 1.5), (1.5, 3), (3, 6), (6, 40))
            sel = [abs(e[m]) for m in eachindex(I) if a*h <= pg.core.positions[I[m]][2] < b*h]
            isempty(sel) || @printf("    y/h in [%.1f, %.1f): max err %.3e  (n=%d)\n", a, b, maximum(sel), length(sel))
        end
        # the ghosts with smallest wall distance and their neighbours
        g = ghosts(pg); φ = [pg.core.wall_distances[i] / h for i in g]
        @printf("    ghost φ/h: min %.3f, #<0.1: %d; interior min y/h: %.3f\n", minimum(φ), count(<(0.1), φ), minimum(pg.core.positions[i][2] for i in I) / h)
    end
end
run("Mirror2 b", MirrorBC(; degree = 2), 80, 1.0, 0.0)
run("Mirror2 b", MirrorBC(; degree = 2), 80, 0.5, 0.2; show = true)
run("Mirror2 b", MirrorBC(; degree = 2), 80, 0.5, 0.05; show = true)
run("Mirror2 b", MirrorBC(; degree = 2), 80, 0.5, 0.1)
run("Outflow", OutflowBC(), 80, 0.5, 0.2; show = true)
