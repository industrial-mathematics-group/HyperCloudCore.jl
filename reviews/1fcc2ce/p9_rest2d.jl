include("common.jl")
function rest2d(N, w; tend = 0.5, periodic = true, f = x -> 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2]))
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, periodic))
    geom = GeometricDomain(intersect(box, piston); is_periodic = (false, periodic), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(t => (t == 5 ? ExtrapolationBC(1) : FixedDirichlet()) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end
    eq0 = LinearAdvection(((0.0,), (0.0,)), Float64)
    ts  = GeneralRKTimeStepper(eq0, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    t = 0.0; dt = 0.4h; step = 0
    while t < tend - 1e-12
        d = min(dt, tend - t); r = ts(eq0, pg, t, d); t += d; step += 1
        inner = interior(pg)
        e = [abs(pg.rhos[i][1] - f(pg.core.positions[i])) for i in inner]
        if step <= 3 || maximum(e) > 0.1
            k = argmax(e); x = pg.core.positions[inner[k]]
            @printf("    step %d t=%.3f max err %.3g at (%.4f, %.4f) face %.4f, U=%.4f f=%.4f V/h²=%.3f\n", step, t, e[k], x[1], x[2], 1 + w*t, pg.rhos[inner[k]][1], f(x), pg.core.volumes[inner[k]]/h^2)
            maximum(e) > 0.1 && break
        end
    end
    inner = interior(pg)
    e = [abs(pg.rhos[i][1] - f(pg.core.positions[i])) for i in inner]
    @printf("  N=%d periodic=%s: max err %.4g ΣV %.5f exact %.5f\n", N, periodic, maximum(e), sumV(pg), 0.5 * (1 + w * t))
end
for p in (true, false), N in (20, 40); rest2d(N, -0.5; periodic = p); end
