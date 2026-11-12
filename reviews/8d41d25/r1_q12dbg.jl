# q12 follow-up: which interior particles are inside the advancing piece before a sweep at |w|dt = 0.2h?
include("common.jl")
h = 1/20
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-0.5, 0.0)))
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
t = 0.0; dt = 0.2h / 0.5
face0 = 1.0
println("t=0: interior particles with x > 1 - 0.6h: ", [(pg.core.positions[i] .- (1.0, 0.0)) ./ h for i in 1:pg.meta.N if !pg.core.is_boundary[i] && pg.core.positions[i][1] > 1 - 0.6h])
for k in 1:40
    global t
    told = t
    t += dt
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
    face = 1 - 0.5t
    ins = [i for i in 1:pg.meta.N if !pg.core.is_boundary[i] && pg.core.positions[i][1] > face]
    isempty(ins) || println("before sweep $k (face $(round(face, digits=4))): inside ", [(round((pg.core.positions[i][1]-face)/h, digits=3), pg.core.positions[i][2]/h) for i in ins])
    r = sweep_boundaries!(pg, 0.0, t, 3)
    near = [(round((face - pg.core.positions[i][1])/h, digits=3), round(pg.core.positions[i][2]/h, digits=3)) for i in 1:pg.meta.N if !pg.core.is_boundary[i] && face - pg.core.positions[i][1] < 0.3h]
    (k <= 3 || !isempty(ins)) && println("  after sweep $k: removed $(r.removed) inserted $(r.inserted); interior within 0.3h of the face (d/h, y/h): ", near)
end
