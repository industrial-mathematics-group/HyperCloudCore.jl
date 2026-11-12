# M3 check: a ball rotating about its own centre near a periodic end (the ball never crosses the seam)
include("common.jl")
h = 1/40
r = 0.1
for gapfrac in (1.2, 1.3, 1.45)
    xc = 1.0 - gapfrac * r                       # the ball's right edge at 1 - (gapfrac-1) r from the seam x = 1
    ball = Moving(Tagged(Ball((xc, 0.0), r), 5), rotating(1.0; center = (xc, 0.0)))
    chan = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - ball
    geom = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    t1 = π / 4
    lo, hi = bounding_box(at_time(ball, t1))
    place_ghosts!(pg, pg.core.positions, 0.0, t1); pg.neighbor(pg)
    msg = try
        sweep_boundaries!(pg, 0.0, t1, 3); "ok"
    catch e
        "refused"
    end
    @printf("ball edge %.3f from the seam, box at t=π/4 hi=%.4f: %s\n", (gapfrac - 1) * r, hi[1], msg)
end
# a static Translated wrapper around a translating motion (frames cancel) — refused now?
ball = Tagged(Ball((0.0, 0.0), 0.1), 5)
try
    GeometricDomain(Cuboid((-1.0, -0.5), (1.0, 0.5)) - Translated(Moving(ball, translating((1.0, 0.0))), (0.3, 0.0)))
    println("Translated(Moving(translating)): accepted")
catch e
    println("Translated(Moving(translating)): refused")
end
# nested wrappers
for (lbl, s) in (("Complement(Rotated(Tagged(Moving)))", complement(Rotated(Tagged(Moving(Ball((0.0, 0.0), 0.1), rotating(1.0)), 5), π/3))),
                 ("Translated(Union(Moving, Ball))", Translated(union(Moving(Ball((0.0, 0.0), 0.1), rotating(1.0)), Ball((0.5, 0.0), 0.1)), (0.1, 0.0))))
    try
        GeometricDomain(intersect(Cuboid((-1.0, -0.5), (1.0, 0.5)), s); mins = (-1.0, -0.5), maxs = (1.0, 0.5))
        println(lbl, ": accepted")
    catch e
        println(lbl, ": ", e isa ArgumentError ? "refused (ArgumentError)" : "error $(typeof(e))")
    end
end
