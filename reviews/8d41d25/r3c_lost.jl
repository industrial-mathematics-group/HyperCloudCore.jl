include("common.jl")
f2(x) = 1.0
h = 1/40
w = -0.5
H5 = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
upper = Tagged(Cuboid((-1.0, 0.5), (3.0, 2.0)), 6)
shape = intersect(Cuboid((0.0, 0.0), (2.0, 1.0)), union(H5, upper))
geom = GeometricDomain(shape; mins = (0.0, 0.0), maxs = (2.0, 1.0), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:6))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
P0 = Set(pg.core.positions[i] for i in interior(pg) if pg.core.positions[i][2] > 0.5 + 3h)
t = 0.0
for k in 1:40
    global t; t += 0.2h / abs(w)
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
    sweep_boundaries!(pg, 0.0, t, 3)
end
P1 = Set(pg.core.positions[i] for i in interior(pg) if pg.core.positions[i][2] > 0.5 + 3h)
println("lost in the upper chamber: ", sort([Tuple(x) for x in setdiff(P0, P1)]))
println("new there: ", sort([Tuple(x) for x in setdiff(P1, P0)]))
