include("common.jl")
h = 1/40; W = ExponentialWeightFunction(1.0, 2.5h)
for wrap in (true, false)
    inner = Moving(Tagged(Cuboid((-0.2, -0.04), (0.2, 0.04)), t -> t + 4), rotating(1.0))
    piece = wrap ? Translated(inner, (0.3, 0.0)) : inner
    chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - piece
    geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => MirrorBC(; degree = 1) for t in 3:8))
    pg = ParticleGrid(geom, (h, h), W, 1); gh = ghosts(pg)
    sd0 = [signed_distance(at_time(chan, 0.0), pg.core.positions[i]) for i in gh]
    X = copy(view(pg.core.positions, 1:pg.meta.N)); place_ghosts!(pg, X, 0.0, 0.5)
    println("Translated wrapper = $wrap: max |sd(t=0.5, placed ghost) - sd(t=0, ghost)| = ", maximum(abs.([signed_distance(at_time(chan, 0.5), X[i]) for i in gh] .- sd0)))
end
