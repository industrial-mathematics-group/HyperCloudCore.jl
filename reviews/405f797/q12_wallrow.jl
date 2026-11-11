# Wall-row particles (on a static wall) next to an advancing piston face: removed at the clearance, or only once covered?
include("common.jl")
h = 1/20
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-0.5, 0.0)))
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
t = 0.0; dt = 0.2h / 0.5            # |w| dt = 0.2 h < clearance
closest_wall, closest_inner, inside = Inf, Inf, 0
for k in 1:40
    global t, closest_wall, closest_inner, inside
    t += dt
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
    # before the sweep: interior particles inside the piece (covered before removal)
    face = 1 - 0.5t
    inside += count(i -> !pg.core.is_boundary[i] && pg.core.positions[i][1] > face, 1:pg.meta.N)
    sweep_boundaries!(pg, 0.0, t, 3)
    face = 1 - 0.5t
    for i in 1:pg.meta.N
        pg.core.is_boundary[i] && continue
        x = pg.core.positions[i]; d = (face - x[1]) / h
        onwall = x[2] == 0.0 || x[2] == 0.5
        onwall ? (closest_wall = min(closest_wall, d)) : (closest_inner = min(closest_inner, d))
    end
end
@printf("after the sweeps (|w|dt = 0.2h): closest wall-row particle to the face %.3f h, closest other particle %.3f h; interior particles found inside the piece before a sweep: %d\n",
        closest_wall, closest_inner, inside)
