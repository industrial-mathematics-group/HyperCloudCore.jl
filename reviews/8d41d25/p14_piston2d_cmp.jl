include("common.jl")
SEQ = get(ENV, "SEQ", "0") == "1"
SEQ && include("seq_sweep.jl")
function run2d(N, w, a, u0, tend; periodic = false)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, periodic))
    geom = GeometricDomain(intersect(box, piston); is_periodic = (false, periodic), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => MirrorBC(; degree = 1)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    eq = LinearAdvection(((a,), (0.0,)), Float64)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4 * h; t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq, pg, t, d); t += d; end
    inner = interior(pg)
    face = 1 + w * t
    inner = [i for i in inner if pg.core.positions[i][1] > a * t + 0.2]; e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner]
    xs = [pg.core.positions[i][1] for i in inner]
    @printf("  SEQ=%s N=%3d err=%.4g (beyond 3h of the face %.4g) ΣV=%.6f |Ω|=%.6f\n", SEQ, N, maximum(e),
            maximum(e[k] for k in eachindex(e) if xs[k] < face - 3h), sumV(pg), face * 0.5)
end
u0(x, t) = 1 + 0.5 * sin(2π * (x[1] - t)) * cos(2π * x[2])     # even about the walls y = 0, 0.5
println("== 2-D receding plane piston co-moving (w = a = 1), walls y = 0, 0.5 MirrorBC, t = 0.5")
for N in (20, 40, 80); run2d(N, 1.0, 1.0, u0, 0.5); end
println("== 2-D advancing plane piston w = -0.5, a = 1 (outflow through the face), t = 0.5")
for N in (20, 40, 80); run2d(N, -0.5, 1.0, u0, 0.5); end
