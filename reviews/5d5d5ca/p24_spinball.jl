include("common.jl")
h = 1/40
res = Dict()
for spin in (false, true)
    b = Tagged(Ball((0.0, 0.0), 0.15), 5)
    piece = spin ? Moving(b, rotating(20.0)) : b       # tip speed 3
    geom = GeometricDomain(Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - piece; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                           bc_map = BCD(t => MirrorBC(; degree = 1) for t in 3:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    u0(x) = 1 + 0.5 * sin(π * x[1]) * cos(π * x[2])
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i])); end
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    V0 = sumV(pg); N0 = pg.meta.N
    t = 0.0; dt = 0.4h
    for _ in 1:50; ts(eq, pg, t, dt); t += dt; end
    res[spin] = Dict(pg.core.positions[i] => pg.rhos[i][1] for i in interior(pg))
    @printf("spin=%s: N %d -> %d, ΣV %.6f -> %.6f\n", spin, N0, pg.meta.N, V0, sumV(pg))
end
common = intersect(keys(res[false]), keys(res[true]))
@printf("interior particles in common %d of %d; max |U_spin - U_static| = %.3g\n", length(common), length(res[false]), maximum(abs(res[true][k] - res[false][k]) for k in common))
