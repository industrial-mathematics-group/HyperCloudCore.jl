include("common.jl")
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(2π * x[2])
h = 1/40
w = -0.5
H5 = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
upper = Tagged(Cuboid((-1.0, 0.5), (3.0, 2.0)), 6)
shape = intersect(Cuboid((0.0, 0.0), (2.0, 1.0)), union(H5, upper))
geom = GeometricDomain(shape; mins = (0.0, 0.0), maxs = (2.0, 1.0), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:6))
println("pieces: ", typeof.(geom.pieces))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
println("bounding box of the piece at 0: ", bounding_box(at_time(H5, 0.0)))
t = 0.0; dt = 0.2h / abs(w)
for k in 1:10
    global t
    t += dt
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
    face = 1 + w * t
    posed = at_time(H5, t)
    cand = [i for i in interior(pg) if abs(signed_distance(posed, pg.core.positions[i])) < 0.25h]
    r = sweep_boundaries!(pg, 0.0, t, 3)
    left = [i for i in interior(pg) if abs(signed_distance(posed, pg.core.positions[i])) < 0.25h]
    @printf("sweep %d face %.4f: interior within 0.25h of the plane before %d (upper %d), after %d (y/h of those: %s); -%d +%d\n", k, face, length(cand),
            count(i -> pg.core.positions[i][2] > 0.5, cand), length(left), string(round.([pg.core.positions[i][2] / h for i in left], digits = 2))[1:min(end, 120)], r.removed, r.inserted)
end
