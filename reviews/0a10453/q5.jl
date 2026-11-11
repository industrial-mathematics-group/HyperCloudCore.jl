include("/tmp/brev_b/probes/common.jl")
T = Float64; dx = 1/40
for seed in (1, 2, 3), Rf in (5.5,)
    geom = get_rectangular_domain(T, (0.0,), (1.0,); bc_map = BCD(1 => ExtrapolationBC(5), 2 => ExtrapolationBC(5)))
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, Rf * dx), 1; randomness = (0.2dx,), rng = MersenneTwister(seed))
    g = ghosts(pg)
    out = String[]
    for k in 0:5
        f(x) = (x[1] - 0.3)^k
        for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(f(pg.core.positions[i])); end
        buf = applybc(pg, 1, T)
        e = maximum(abs(buf[i][1] - f(pg.core.positions[i])) for i in g)
        push!(out, @sprintf("k=%d %.1e", k, e))
    end
    println("seed $seed R=$Rf: ", join(out, "  "))
end
