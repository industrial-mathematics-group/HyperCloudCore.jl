# _check_periodic_pieces after 8d41d25: thin pieces, a piece past the seam, 3-D, cost
include("common.jl")
function trysweep(geom, h, t; D = 2)
    pg = try ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1) catch e; return "grid refused: " * first(sprint(showerror, e), 60), nothing end
    pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    place_ghosts!(pg, pg.core.positions, 0.0, t); pg.neighbor(pg)
    try
        sweep_boundaries!(pg, 0.0, t, 3); return "accepted", pg
    catch e
        return (e isa ArgumentError ? "refused" : "ERROR $(typeof(e)): $(sprint(showerror, e)[1:min(end,150)])"), pg
    end
end
h = 1/40
# thin horizontal plate (thickness th) translating across the periodic end x = 1 (half of it beyond the seam at t)
for th in (0.3h, 0.45h, 0.6h, 1.2h, 4h)
    for y0 in (0.0, 0.3h * 0.37)     # plate's lower edge on / off the sample lattice
        plate = Moving(Tagged(Cuboid((0.7, y0), (0.9, y0 + th)), 5), translating((1.0, 0.0)))
        chan = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
        geom = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:5))
        r, _ = trysweep(geom, h, 0.2)             # plate at [0.9, 1.1]: crosses the seam
        @printf("thin plate thickness %.2f h, lower edge y0 = %.4f: crossing at t = 0.2 -> %s\n", th / h, y0, r)
    end
end
# thin plate rotated by 30 degrees crossing
for th in (0.3h, 0.6h)
    body = Rotated(Cuboid((-0.1, -th / 2), (0.1, th / 2)), π / 6)
    plate = Moving(Tagged(Translated(body, (0.8, 0.0)), 5), translating((1.0, 0.0)))
    chan = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
    geom = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:5))
    r, _ = trysweep(geom, h, 0.2)
    @printf("rotated thin plate thickness %.2f h crossing -> %s\n", th / h, r)
end
# a piece entirely beyond the periodic end (jumped over the seam in one sweep): plate [0.9, 0.95] -> [1.02, 1.07]
plate = Moving(Tagged(Cuboid((0.9, -0.05), (0.95, 0.05)), 5), translating((1.0, 0.0)))
chan = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
geom = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:5))
r, pg = trysweep(geom, h, 0.12)
@printf("plate entirely past the seam after one sweep (box [1.02, 1.07]) -> %s; interior particles inside the plate's wrapped image: %d, ΣV %.4f (region %.4f)\n", r,
        count(i -> -0.98 <= pg.core.positions[i][1] <= -0.93 && abs(pg.core.positions[i][2]) <= 0.05, interior(pg)), sumV(pg), 2.0 - 0.05 * 0.1)
# 3-D: a ball spinning about its centre near the seam (accepted?), a plate crossing (refused?), cost of the check
for (lbl, mk, t) in (("3-D ball r=0.1 spinning, edge 0.03 from the seam", () -> Moving(Tagged(Ball((0.37, 0.0, 0.0), 0.1), 7), rotating((0.0, 0.0, 1.0), 1.0; center = (0.37, 0.0, 0.0))), π / 4),
                     ("3-D plate crossing the seam", () -> Moving(Tagged(Cuboid((0.2, -0.1, -0.1), (0.35, 0.1, 0.1)), 7), translating((1.0, 0.0, 0.0))), 0.1))
    piece = mk()
    box = Cuboid((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5); open = (true, false, false)) - piece
    geom = GeometricDomain(box; is_periodic = (true, false, false), mins = (-0.5, -0.5, -0.5), maxs = (0.5, 0.5, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:7))
    h3 = 1/24
    pg = ParticleGrid(geom, (h3, h3, h3), ExponentialWeightFunction(1.0, 2.5h3), 1)
    place_ghosts!(pg, pg.core.positions, 0.0, t); pg.neighbor(pg)
    H._check_periodic_pieces(pg, geom, t) === nothing || nothing
    tm = @elapsed (r = try H._check_periodic_pieces(pg, geom, t); "accepted" catch e; "refused" end)
    tm = @elapsed (try H._check_periodic_pieces(pg, geom, t) catch end)
    @printf("%s -> %s (check %.2f ms)\n", lbl, r, 1e3 * tm)
end
