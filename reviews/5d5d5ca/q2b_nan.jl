# Where does the NaN of a fast piston sweep (|w|dt = 1.5 h) come from?
include("common.jl")
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
h = 1/40
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-1.0, 0.0)))
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, true)), piston); is_periodic = (false, true), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
hasnan(pg) = any(i -> any(isnan, pg.rhos[i]) || isnan(pg.core.volumes[i]), 1:pg.meta.N)
t = 0.0
for k in 1:6
    global t
    t_old = t; t += 1.5h
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
    face = 1 - t
    # replicate the region pass on a copy to see zero volumes
    pc = deepcopy(pg)
    cd, R, dx = pc.domain, pc.meta.R, minimum(pc.meta.dx)
    boxes = map(pc.geometry.pieces) do piece
        lo0, hi0 = bounding_box(at_time(piece, pc.meta.sweep_time)); lo1, hi1 = bounding_box(at_time(piece, t))
        (max.(min.(lo0, lo1) .- (R + 2dx), cd.canvas_mins), min.(max.(hi0, hi1) .+ (R + 2dx), cd.canvas_maxs))
    end
    H._update_region!(pc, pc.geometry, pc.meta.sweep_time, t, boxes, 2)
    zs = [i for i in 1:pc.meta.N if !pc.core.is_boundary[i] && pc.core.volumes[i] == 0]
    @printf("sweep %d (face %.4f): after the region pass: %d interior particles with V = 0 (x - face in h: %s), NaN %s\n", k, face, length(zs),
            string(unique(round.([(pc.core.positions[i][1] - face) / h for i in zs], digits = 2))), hasnan(pc))
    sweep_boundaries!(pg, 0.0, t, 3)
    if hasnan(pg)
        bad = [i for i in 1:pg.meta.N if isnan(pg.rhos[i][1]) || isnan(pg.core.volumes[i])]
        @printf("   NaN after the sweep at %d particles, x - face in h: %s\n", length(bad), string(sort(unique(round.([(pg.core.positions[i][1] - face) / h for i in bad], digits = 1)))[1:min(end, 12)]))
        break
    end
end
