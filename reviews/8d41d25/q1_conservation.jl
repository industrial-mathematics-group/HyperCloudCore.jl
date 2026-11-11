# Conservation of the sweep: the change of interior_mass over a sweep vs the change from the region pass alone
# (the remaps should be exactly conservative), and Σ V vs the region volume.
include("common.jl")
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
function boxes_of(pg, t_old, t)
    geom, cd, R, dx = pg.geometry, pg.domain, pg.meta.R, minimum(pg.meta.dx)
    map(geom.pieces) do piece
        lo0, hi0 = bounding_box(at_time(piece, t_old)); lo1, hi1 = bounding_box(at_time(piece, t))
        (max.(min.(lo0, lo1) .- (R + 2dx), cd.canvas_mins), min.(max.(hi0, hi1) .+ (R + 2dx), cd.canvas_maxs))
    end
end
function check(mk, ts_list; label = "")
    println("== ", label)
    pgA = mk(); pgB = mk()
    t = 0.0
    for t1 in ts_list
        for pg in (pgA, pgB)
            place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t1); pg.neighbor(pg)
        end
        # A: region pass alone (on a copy state identical to B)
        mA0, VA0 = interior_mass(pgA)[1], sumV(pgA)
        H._update_region!(pgA, pgA.geometry, pgA.meta.sweep_time, t1, boxes_of(pgA, pgA.meta.sweep_time, t1), 2)
        dmA, dVA = interior_mass(pgA)[1] - mA0, sumV(pgA) - VA0
        mB0, VB0 = interior_mass(pgB)[1], sumV(pgB)
        r = sweep_boundaries!(pgB, 0.0, t1, 3)
        dmB, dVB = interior_mass(pgB)[1] - mB0, sumV(pgB) - VB0
        @printf("  t=%.4f +%d -%d: Δmass region %.6e, sweep %.6e (diff %.2e, residual %.2e); ΔV region %.6e sweep %.6e (diff %.2e); NaN %s\n",
                t1, r.inserted, r.removed, dmA, dmB, dmB - dmA, r.mass_residual[1] - dmB, dVA, dVB, dVB - dVA,
                any(i -> any(isnan, pgB.rhos[i]) || isnan(pgB.core.volumes[i]), 1:pgB.meta.N))
        # bring A to B's state for the next step
        pgA = deepcopy(pgB)
    end
    return pgB
end
h = 1/40
mkpiston(w; periodic = true) = () -> begin
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, periodic))
    geom = GeometricDomain(intersect(box, piston); is_periodic = (false, periodic), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
    pg
end
pg = check(mkpiston(-0.5), [0.2h * k for k in 1:8]; label = "2-D plane piston advancing, periodic y")
@printf("  final ΣV %.6f exact %.6f\n", sumV(pg), 0.5 * (1 - 0.5 * 1.6h))
pg = check(mkpiston(1.0; periodic = false), [0.2h * k for k in 1:8]; label = "2-D plane piston receding, walls")
@printf("  final ΣV %.6f exact %.6f\n", sumV(pg), 0.5 * (1 + 1.6h))
mkball() = begin
    ballm = Moving(Tagged(Ball((0.5, 0.25), 0.1), 5), translating((0.5, 0.2)))
    geom = GeometricDomain(Cuboid((0.0, 0.0), (1.0, 0.5)) - ballm; mins = (0.0, 0.0), maxs = (1.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
    pg
end
pg = check(mkball, [0.01 * k for k in 1:6]; label = "2-D ball")
@printf("  final ΣV %.6f exact %.6f\n", sumV(pg), 0.5 - π * 0.01)
