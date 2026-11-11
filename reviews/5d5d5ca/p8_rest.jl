include("common.jl")
# fluid at rest: u must stay u0 everywhere that is still fluid (velocity 0)
function rest1d(N, w; tend = 0.5, f = x -> 1 + 0.5 * sin(2π * x), bc = ExtrapolationBC(1), sweeps_only = false)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((w,)))
    geom   = GeometricDomain(intersect(Tagged(Cuboid((0.0,), (2.0,)), 1), piston); mins = (0.0,), maxs = (2.0,),
                             bc_map = BCD(1 => FixedDirichlet(), 2 => bc))
    pg = ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i][1])); end
    eq0 = LinearAdvection(((0.0,),), Float64)
    ts  = GeneralRKTimeStepper(eq0, MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    t = 0.0; dt = 0.4h
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq0, pg, t, d); t += d; end
    inner = interior(pg)
    e = [abs(pg.rhos[i][1] - f(pg.core.positions[i][1])) for i in inner]
    k = argmax(e)
    x = pg.core.positions[inner[k]][1]
    @printf("  1-D N=%4d w=%5.2f max err %.4g at x=%.4f (face %.4f, (face-x)/h=%.2f); err beyond 3h of face %.3g\n", N, w, e[k], x, 1 + w * tend, (1 + w*tend - x)/h,
            maximum(e[j] for j in eachindex(inner) if pg.core.positions[inner[j]][1] < 1 + w*tend - 3h; init = 0.0))
end
println("== 1-D fluid at rest, piston advancing (w = -0.5) and receding (w = 0.5), t = 0.5")
for w in (-0.5, 0.5), N in (50, 100, 200, 400); rest1d(N, w); end
