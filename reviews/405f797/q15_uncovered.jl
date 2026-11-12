# Where is the error in the uncovered part behind a receding co-moving piston next to walls (consistent data)?
include("common.jl")
uc(x, t) = 1 + 0.5 * cos(2π * (x[1] - t - 1)) * cos(2π * x[2])
for N in (20, 40, 80)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
    geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => MirrorBC(; degree = 1)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(uc(pg.core.positions[i], 0.0)); end
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4h; t = 0.0
    while t < 0.5 - 1e-12; d = min(dt, 0.5 - t); ts(eq, pg, t, d); t += d; end
    face = 1 + t
    rows = Dict{Int, Float64}(); xs = Dict{Int, Float64}()
    for i in interior(pg)
        x = pg.core.positions[i]
        (x[1] > t + 0.2 && x[1] < face - 3h) || continue
        e = abs(pg.rhos[i][1] - uc(x, t))
        r = round(Int, min(x[2], 0.5 - x[2]) / h)          # row index from the nearest wall
        if e > get(rows, r, 0.0); rows[r] = e; xs[r] = x[1]; end
    end
    print("N=$N max err by wall row (row: err @ x): ")
    for r in sort(collect(keys(rows)))[1:min(end, 6)]; @printf("%d: %.3g @ %.3f  ", r, rows[r], xs[r]); end
    # split by x: initial part (x < 1) vs uncovered part (x > 1)
    e_old = maximum(abs(pg.rhos[i][1] - uc(pg.core.positions[i], t)) for i in interior(pg) if t + 0.2 < pg.core.positions[i][1] < 1 - 3h)
    e_new = maximum(abs(pg.rhos[i][1] - uc(pg.core.positions[i], t)) for i in interior(pg) if 1 + 3h < pg.core.positions[i][1] < face - 3h)
    @printf("\n   x in (t+0.2, 1-3h): %.4g;  uncovered (1+3h, face-3h): %.4g\n", e_old, e_new)
end
