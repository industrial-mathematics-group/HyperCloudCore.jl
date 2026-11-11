include("common.jl")
# 2-D tube [0,2]x[0,0.5] periodic in y, piston plane at x = 1 moving with w
function run2d(N, w, a, u0, tend; bc = MirrorBC(; degree = 1), jitter = 0.0)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, true))
    geom = GeometricDomain(intersect(box, piston); is_periodic = (false, true), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 5 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jitter*h, jitter*h))
    eq = LinearAdvection(((a,), (0.0,)), Float64)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4 * h; t = 0.0
    m0 = interior_mass(pg)[1]
    while t < tend - 1e-12
        d = min(dt, tend - t); ts(eq, pg, t, d); t += d
    end
    inner = interior(pg)
    V = [pg.core.volumes[i] / h^2 for i in inner]
    err = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner)
    face = 1 + w * t
    @printf("  N=%3d w=%5.2f a=%4.1f jitter=%.2f err=%.4g ΣV=%.6f |Ω|=%.6f V/h² in [%.3f, %.3f] (%d < 0.5, %d > 1.5) n=%d minU=%.4f maxU=%.4f\n",
            N, w, a, jitter, err, sumV(pg), face * 0.5, minimum(V), maximum(V), count(<(0.5), V), count(>(1.5), V), length(inner),
            minimum(pg.rhos[i][1] for i in inner), maximum(pg.rhos[i][1] for i in inner))
    return pg
end
u0(x, t) = 1 + exp(-((x[1] - 0.8 - t)^2 + (x[2] - 0.25)^2) / 0.02)
println("== 2-D receding plane piston, co-moving flow (exact u0(x - t)), t = 0.5")
for N in (20, 40, 80); run2d(N, 1.0, 1.0, u0, 0.5); end
println("== jittered (0.2h)")
for N in (20, 40); run2d(N, 1.0, 1.0, u0, 0.5; jitter = 0.2); end
println("== 2-D advancing plane piston, fluid at rest, smooth data (u should stay u0 where still fluid), t = 0.5")
us(x, t) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
for N in (20, 40, 80); run2d(N, -0.5, 0.0, us, 0.5); end
println("== 2-D receding, fluid at rest, smooth data, t = 0.5 (inserted points get interpolated data)")
for N in (20, 40, 80); run2d(N, 0.5, 0.0, us, 0.5); end
