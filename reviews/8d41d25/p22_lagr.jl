include("common.jl")
for mover in (NoGridMover(), LagrangianMover())
    N = 50; h = 1/N
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((1.0,)))
    geom   = GeometricDomain(intersect(Tagged(Cuboid((0.0,), (2.0,)), 1), piston); mins = (0.0,), maxs = (2.0,),
                             bc_map = BCD(1 => FixedDirichlet(), 2 => MirrorBC(; degree = 1)))
    pg = ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, 2.5h), 1; mover = mover)
    u0(x) = 1 + exp(-((x - 0.8) / 0.08)^2)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i][1])); end
    eq = LinearAdvection(((1.0,),), Float64)
    interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    t = 0.0; dt = 0.4h
    try
        for _ in 1:round(Int, 0.5/dt); ts(eq, pg, t, dt); t += dt; end
        inner = interior(pg)
        err = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i][1] - t)) for i in inner)
        @printf("%s: t=%.2f ΣV=%.4f (|Ω(t)| = %.4f) n=%d err=%.3g max x=%.3f\n", typeof(mover), t, sumV(pg), 1 + t, length(inner), err, maximum(pg.core.positions[i][1] for i in inner))
    catch e
        println(typeof(mover), ": ERR ", sprint(showerror, e)[1:min(end, 300)])
    end
end
