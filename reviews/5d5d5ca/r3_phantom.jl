# Removal by the pieces' own distances: a piece whose surface lies partly inside the region (a piston in the lower
# half of a box with an open upper chamber: fluid = box ∩ (H ∪ upper)). The plane x = face continues through the
# upper chamber, where it is no wall.
include("common.jl")
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(2π * x[2])
h = 1/40
for (lbl, w) in (("advancing", -0.5), ("receding", 0.5))
    H5 = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
    upper = Tagged(Cuboid((-1.0, 0.5), (3.0, 2.0)), 6)
    shape = intersect(Cuboid((0.0, 0.0), (2.0, 1.0)), union(H5, upper))
    geom = GeometricDomain(shape; mins = (0.0, 0.0), maxs = (2.0, 1.0), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:6))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
    upperset(p) = [i for i in interior(p) if p.core.positions[i][2] > 0.5 + 3h]
    n0 = length(upperset(pg)); V0 = sumV(pg)
    t = 0.0; dt = 0.2h / abs(w)
    remU, insU, totr, toti = 0, 0, 0, 0
    for k in 1:40
        t += dt
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        before = Set(pg.core.positions[i] for i in upperset(pg))
        r = sweep_boundaries!(pg, 0.0, t, 3)
        after = Set(pg.core.positions[i] for i in upperset(pg))
        remU += length(setdiff(before, after)); insU += length(setdiff(after, before)); totr += r.removed; toti += r.inserted
    end
    face = 1 + w * t
    up = upperset(pg)
    near = [i for i in up if abs(pg.core.positions[i][1] - face) < 0.25h]
    err = maximum(abs(pg.rhos[i][1] - f2(pg.core.positions[i])) for i in up)
    @printf("%s piston, 40 sweeps of 0.2 h: upper chamber (y > 0.5+3h) interior %d -> %d; removed there %d, inserted there %d (all: -%d +%d); particles within 0.25h of the plane there: %d; max |U - f| there %.3g; ΣV %.6f (exact %.6f)\n",
            lbl, n0, length(up), remU, insU, totr, toti, length(near), err, sumV(pg), 1.0 + 0.5 * face)
end
