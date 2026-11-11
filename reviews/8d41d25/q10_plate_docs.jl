# docs: the plate with 1 + ½ cos 5π(x − t) (zero normal derivative on the plate's faces)
include("common.jl")
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
plate_shape = Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4)
function run(N, tend, u0)
    h = 1 / N
    chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - Moving(plate_shape, translating((1.0, 0.0)))
    geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                            bc_map = BCD(t => MirrorBC(; degree = 1) for t in (3, 4, 5, 6, 7, 8)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eq, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq, pg, t, d); t += d; end
    inner = interior(pg)
    dpl(x) = signed_distance(plate_shape, x - SVector(t, 0.0))
    e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner]
    dd = [dpl(pg.core.positions[i]) for i in inner]
    @printf("  N=%3d max err %.4g; within 12h of the plate %.4g; beyond 12h %.4g\n", N, maximum(e),
            maximum(e[k] for k in eachindex(e) if dd[k] < 12h), maximum(e[k] for k in eachindex(e) if dd[k] >= 12h; init=0.0))
end
uc(x, t) = 1 + 0.5 * cos(5π * (x[1] - t))
println("== plate, u0 = 1 + ½ cos 5π(x - t), t = 0.5")
for N in (20, 40, 80); run(N, 0.5, uc); end
