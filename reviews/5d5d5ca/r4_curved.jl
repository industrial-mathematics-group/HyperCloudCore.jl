# Insertion `signed_distance(shape_t, x) <= -gap`: behind a piece receding along a curved static wall, the lattice
# points less than half a spacing inside the wall are no longer inserted (a generated cloud has them). Hole size and
# near-wall particles in the uncovered part, swept vs generated at the same pose.
include("common.jl")
function holes(p, region_pts)
    P = [p.core.positions[i] for i in interior(p)]
    maximum(y -> minimum(x -> norm(x - y), P), region_pts)
end
for h in (1/20, 1/40)
    rad = 0.8
    discp(mov) = (pp = Tagged(HalfSpace((-0.3, 0.0), (1.0, 0.0)), 5); intersect(Tagged(Ball((0.0, 0.0), rad), 1), mov ? Moving(pp, translating((1.0, 0.0))) : pp))
    bcs = BCD(1 => MirrorBC(; degree = 1), 5 => ExtrapolationBC(1))
    geom = GeometricDomain(discp(true); mins = (-rad, -rad), maxs = (rad, rad), bc_map = bcs)
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    t = 0.0
    while t < 0.4 - 1e-12
        t += 0.2h
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        sweep_boundaries!(pg, 0.0, t, 3)
    end
    face = -0.3 + t
    gF = GeometricDomain(at_time(geom.shape, t); mins = (-rad, -rad), maxs = (rad, rad), bc_map = bcs)
    pF = ParticleGrid(gF, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    # uncovered part: -0.3 + 3h < x < face - 3h; sample points of the arc and of a band inside it
    xs = range(-0.3 + 3h, face - 3h; length = 200)
    arc = [Space{2, Float64}(x, s * sqrt(rad^2 - x^2)) for x in xs for s in (-1, 1)]
    unc(x) = -0.3 + 3h < x[1] < face - 3h
    depth(x) = rad - norm(x)
    for (lbl, p) in (("swept", pg), ("generated", pF))
        inn = [i for i in interior(p) if unc(p.core.positions[i])]
        d = [depth(p.core.positions[i]) for i in inn]
        nearw = count(<(0.5h), d)
        # per arc point, the distance to the nearest interior particle
        P = [p.core.positions[i] for i in interior(p)]
        hmax = maximum(y -> minimum(x -> norm(x - y), P), arc)
        @printf("  h=1/%d %-9s: interior in the uncovered part %d, with depth < 0.5h %d, min depth %.3f h; largest distance arc → nearest particle %.3f h; ΣV %.6f\n",
                round(Int, 1/h), lbl, length(inn), nearw, minimum(d) / h, hmax / h, sumV(p))
    end
    # initial (never swept) part for reference: x < -0.3 - 3h
    xs0 = range(-rad + 0.1, -0.3 - 3h; length = 200)
    arc0 = [Space{2, Float64}(x, s * sqrt(rad^2 - x^2)) for x in xs0 for s in (-1, 1)]
    P = [pg.core.positions[i] for i in interior(pg)]
    @printf("  h=1/%d initial part: largest distance arc → nearest particle %.3f h\n", round(Int, 1/h), maximum(y -> minimum(x -> norm(x - y), P), arc0) / h)
end
