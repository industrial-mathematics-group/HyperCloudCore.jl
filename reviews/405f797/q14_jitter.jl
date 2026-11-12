# Insertions far from the piece: a jittered cloud (or one with holes) inside a piece's box gets filled by the sweep
include("common.jl")
using Random
for jit in (0.2, 0.35, 0.45)
    h = 1/40
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-0.5, 0.0)))
    geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, true)), piston); is_periodic = (false, true), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit * h, jit * h), rng = MersenneTwister(1))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0 + pg.core.positions[i][1]); end
    n0 = count(i -> !pg.core.is_boundary[i], 1:pg.meta.N)
    t = 0.05h
    place_ghosts!(pg, pg.core.positions, 0.0, t); pg.neighbor(pg)
    P0 = Set(pg.core.positions[i] for i in 1:pg.meta.N)
    r = sweep_boundaries!(pg, 0.0, t, 3)
    newx = [pg.core.positions[i][1] for i in 1:pg.meta.N if !pg.core.is_boundary[i] && !(pg.core.positions[i] in P0)]
    @printf("jitter %.2f h: interior %d, one sweep moving the face by 0.025 h: +%d -%d; inserted particles at distances from the face (in h) %s\n",
            jit, n0, r.inserted, r.removed, isempty(newx) ? "-" : string(round.(extrema((1 - 0.5t) .- newx) ./ h, digits = 1)))
end
