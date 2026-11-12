include("common.jl")
f(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
function onesweep(geom, h, t1; label = "")
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end
    U0 = copy(pg.rhos[1:pg.meta.N]); P0 = copy(pg.core.positions[1:pg.meta.N]); V0 = copy(pg.core.volumes[1:pg.meta.N])
    m0 = interior_mass(pg)[1]
    place_ghosts!(pg, pg.core.positions, 0.0, t1)
    pg.neighbor(pg)
    r = sweep_boundaries!(pg, 0.0, t1, 3)
    # match survivors by position
    d = Dict(P0[i] => i for i in 1:length(P0))
    sh = at_time(geom.shape, t1)
    rows = Tuple{Float64, Float64, Float64}[]
    for i in 1:pg.meta.N
        pg.core.is_boundary[i] && continue
        j = get(d, pg.core.positions[i], 0); j == 0 && continue
        push!(rows, (-signed_distance(sh, pg.core.positions[i]) / h, abs(pg.rhos[i][1] - U0[j][1]), pg.core.volumes[i] / V0[j]))
    end
    @printf("  %s: inserted %d removed %d; ΔΣVU=%.3g\n", label, r.inserted, r.removed, interior_mass(pg)[1] - m0)
    for (a, b) in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 100))
        sel = [x for x in rows if a <= x[1] < b]
        isempty(sel) && continue
        @printf("     distance to the wall in [%2d,%3d)h: n=%4d max|ΔU|=%.3g  V/V0 in [%.3f, %.3f]\n", a, b, length(sel), maximum(x[2] for x in sel), minimum(x[3] for x in sel), maximum(x[3] for x in sel))
    end
end
h = 1/40
for periodic in (false, true)
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-0.5, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, periodic))
    geom = GeometricDomain(intersect(box, piston); is_periodic = (false, periodic), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(t => FixedDirichlet() for t in 1:5))
    onesweep(geom, h, 0.04; label = "plane piston, one row removed (periodic y = $periodic)")
end
# a small ball approaching one particle: one removal
ballm = Moving(Tagged(Ball((0.5 + 0.1 + 0.3/40, 0.25 + 0.0), 0.1), 5), translating((-0.5, 0.0)))
box = Cuboid((0.0, 0.0), (1.0, 0.5))
geom = GeometricDomain(box - ballm; mins = (0.0, 0.0), maxs = (1.0, 0.5), bc_map = BCD(t => FixedDirichlet() for t in 1:5))
for t1 in (0.01, 0.03); onesweep(geom, h, t1; label = "ball, t1 = $t1"); end
