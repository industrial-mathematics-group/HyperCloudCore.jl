# _refresh_static_ghosts!: static-wall ghosts (tags 3, 4) of an oscillating piston along walls. Their stored normal and
# wall distance vs the static wall's own (n = (0, ∓1), φ = distance to the wall line), and vs a cloud generated at the pose.
include("common.jl")
h = 1/20
bcs = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => ExtrapolationBC(1))
osc = t -> RigidMotion(SVector(0.3sin(2π * t), 0.0), SMatrix{2,2,Float64,4}(1, 0, 0, 1), SVector(0.6π * cos(2π * t), 0.0), zero(SMatrix{2,2,Float64,4}))
pc = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), osc)
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), pc); mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = bcs)
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
t = 0.0; dt = 0.1h / (0.6π)
nst = round(Int, 2 / dt)
for k in 1:nst
    global t; t += dt
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg); sweep_boundaries!(pg, 0.0, t, 3)
    if k % round(Int, 0.25 / dt) == 0
        face = 1 + 0.3sin(2π * t)
        st = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] in (3, 4)]
        # static ghosts inside the piston (x > face) and their normals
        inside = [i for i in st if pg.core.positions[i][1] > face]
        nx = [i for i in st if abs(pg.core.normals[i][1]) > 0.5]
        # fresh cloud at the pose: what are these positions there?
        gF = GeometricDomain(at_time(geom.shape, t); mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = bcs)
        pF = ParticleGrid(gF, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
        key(x) = round.(Tuple(x) ./ (h / 1000))
        F = Dict(key(pF.core.positions[i]) => i for i in 1:pF.meta.N)
        dn, dd, tagdiff, nofresh = 0.0, 0.0, 0, 0
        for i in st
            j = get(F, key(pg.core.positions[i]), 0)
            j == 0 && (nofresh += 1; continue)
            pF.core.tags[j] != pg.core.tags[i] && (tagdiff += 1)
            dn = max(dn, norm(pg.core.normals[i] - pF.core.normals[j])); dd = max(dd, abs(pg.core.wall_distances[i] - pF.core.wall_distances[j]))
        end
        @printf("t=%.2f face %.3f: static ghosts %d, inside the piston %d, with |n_x| > 0.5: %d; vs fresh: not in fresh %d, other tag in fresh %d, max |Δn| %.2f, max |Δφ| %.3f h\n",
                t, face, length(st), length(inside), length(nx), nofresh, tagdiff, dn, dd / h)
    end
end
