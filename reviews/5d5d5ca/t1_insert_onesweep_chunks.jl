# One sweep behind a receding plane piston (fluid at rest, no PDE): what the insertions (now one remap round) do to
# the survivors (|ΔU|, V/V0 by distance to the face) and the inserted particles (V/h² against a cloud generated at the
# pose, error against the smooth data), walls at y = 0, 0.5 or periodic in y.
include("common.jl"); include("o_chunks.jl")
f(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
mkgeom(x0, per) = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, per)),
                       Moving(Tagged(HalfSpace((x0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0))));
                       is_periodic = (false, per), mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = BCD(t => FixedDirichlet() for t in 1:5))
h = 1 / 40
for per in (false, true), cols in (0.6, 1.6, 2.6)
    t1 = cols * h
    geom = mkgeom(1.0, per)
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end
    P0 = copy(pg.core.positions[1:pg.meta.N]); U0 = copy(pg.rhos[1:pg.meta.N]); V0 = copy(pg.core.volumes[1:pg.meta.N])
    d0 = Dict(P0[i] => i for i in 1:length(P0) if !pg.core.is_boundary[i])
    place_ghosts!(pg, pg.core.positions, 0.0, t1); pg.neighbor(pg)
    r = sweep_boundaries!(pg, 0.0, t1, 3)
    # generated at the pose
    gg = ParticleGrid(mkgeom(1.0 + t1, per), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    dg = Dict(round.(Tuple(gg.core.positions[i]), digits = 9) => gg.core.volumes[i] for i in 1:gg.meta.N if !gg.core.is_boundary[i])
    face = 1 + t1
    @printf("periodic=%s, sweep of %.1f h: inserted %d removed %d, ΣV %.6f (region %.6f), interior %d (generated %d)\n",
            per, cols, r.inserted, r.removed, sumV(pg), 0.5 * face, length(interior(pg)), length(interior(gg)))
    rows = Tuple{Float64, Float64, Float64}[]; ins = Tuple{Float64, Float64, Float64, Float64}[]
    for i in interior(pg)
        x = pg.core.positions[i]
        j = get(d0, x, 0)
        if j == 0
            vg = get(dg, round.(Tuple(x), digits = 9), NaN)
            push!(ins, (x[2], pg.core.volumes[i] / h^2, vg / h^2, pg.rhos[i][1] - f(x)))
        else
            push!(rows, ((face - x[1]) / h, abs(pg.rhos[i][1] - U0[j][1]), pg.core.volumes[i] / V0[j]))
        end
    end
    for (a, b) in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 100))
        sel = [x for x in rows if a <= x[1] < b]; isempty(sel) && continue
        @printf("   survivors [%2d,%3d)h from the face: n=%4d max|ΔU|=%.3g  V/V0 in [%.3f, %.3f]\n", a, b, length(sel),
                maximum(x[2] for x in sel), minimum(x[3] for x in sel), maximum(x[3] for x in sel))
    end
    if !isempty(ins)
        wall = [x for x in ins if x[1] == 0.0 || x[1] == 0.5]; mid = [x for x in ins if 0 < x[1] < 0.5]
        for (lab, s) in (("wall-row", wall), ("inner", mid))
            isempty(s) && continue
            @printf("   inserted %-8s n=%3d V/h² in [%.3f, %.3f] (generated [%.3f, %.3f]), error in [%+.4f, %+.4f]\n", lab, length(s),
                    minimum(x[2] for x in s), maximum(x[2] for x in s), minimum(x[3] for x in s), maximum(x[3] for x in s),
                    minimum(x[4] for x in s), maximum(x[4] for x in s))
        end
    end
    pts = [pg.core.positions[i] for i in 1:pg.meta.N]
    dmin = minimum(norm(H._wrapped_diff(pts[a], pts[b], pg.domain.L_wrap, pg.domain.invL_wrap)) for a in 1:length(pts) for b in a+1:length(pts) if abs(pts[a][1] - pts[b][1]) < 2h)
    @printf("   min pair distance near the face %.3f h\n", dmin / h)
end
