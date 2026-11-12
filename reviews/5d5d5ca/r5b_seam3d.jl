include("common.jl")
for (lbl, mk, t) in (("3-D ball r=0.1 spinning, edge 0.03 from the seam", () -> Moving(Tagged(Ball((0.37, 0.0, 0.0), 0.1), 7), rotating((0.0, 0.0, 1.0), 1.0; center = (0.37, 0.0, 0.0))), π / 4),
                     ("3-D plate crossing the seam by 0.1", () -> Moving(Tagged(Cuboid((0.2, -0.1, -0.1), (0.35, 0.1, 0.1)), 7), translating((1.0, 0.0, 0.0))), 0.25),
                     ("3-D cube rotated 45° about z, edge 0.01 across", () -> (c = 0.5 - 0.1*sqrt(2) + 0.01; Moving(Tagged(Rotated(Cuboid((c-0.1, -0.1, -0.1), (c+0.1, 0.1, 0.1)), (0.0, 0.0, 1.0), π/4; center = (c, 0.0, 0.0)), 7), translating((1.0, 0.0, 0.0)))), 0.0))
    piece = try mk() catch e; println(lbl, ": construction failed ", first(sprint(showerror, e), 100)); continue end
    box = Cuboid((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5); open = (true, false, false)) - piece
    geom = GeometricDomain(box; is_periodic = (true, false, false), mins = (-0.5, -0.5, -0.5), maxs = (0.5, 0.5, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:7))
    h3 = 1/24
    pg = ParticleGrid(geom, (h3, h3, h3), ExponentialWeightFunction(1.0, 2.5h3), 1)
    r = try H._check_periodic_pieces(pg, geom, t); "accepted" catch e; "refused" end
    tm = @elapsed (try H._check_periodic_pieces(pg, geom, t) catch end)
    lo, hi = bounding_box(at_time(piece, t))
    @printf("%s: box x [%.3f, %.3f] -> %s (check %.3f ms)\n", lbl, lo[1], hi[1], r, 1e3 * tm)
end
# 2-D: a square rotated 45° whose corner pokes across the seam by δ
h = 1/40
for δ in (0.1h, 0.2h, 0.3h, 0.6h)
    c = 1.0 - 0.1 * sqrt(2) + δ
    sq = Moving(Tagged(Rotated(Cuboid((c - 0.1, -0.1), (c + 0.1, 0.1)), π / 4; center = (c, 0.0)), 5), translating((0.0, 0.0)))
    chan = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - sq
    geom = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    r = try H._check_periodic_pieces(pg, geom, 0.0); "accepted" catch e; "refused" end
    @printf("2-D square at 45°, corner %.2f h across the seam -> %s\n", δ / h, r)
end
