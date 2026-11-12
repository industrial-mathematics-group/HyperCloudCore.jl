include("common.jl")
oscill(A, ω) = t -> RigidMotion(SVector(A * sin(ω * t)), SMatrix{1,1}(1.0), SVector(A * ω * cos(ω * t)), SMatrix{1,1}(0.0))
u0o(x, t) = 1 + exp(-((x - 0.2 - t) / 0.08)^2)
for N in (50, 100)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), oscill(0.1, 8π))
    geom   = GeometricDomain(intersect(Tagged(Cuboid((0.0,), (2.0,)), 1), piston); mins = (0.0,), maxs = (2.0,),
                             bc_map = BCD(1 => FixedDirichlet(), 2 => ExtrapolationBC(1)))
    pg = ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0o(pg.core.positions[i][1], 0.0)); end
    eq = LinearAdvection(((1.0,),), Float64)
    interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4 * get_time_step(pg, eq, interp); t = 0.0
    face(t) = 1 + 0.1sin(8π * t)
    @printf("N=%d dt=%.4g (dt/h = %.3f)\n", N, dt, dt / h)
    k = 0; prev = 0.0
    while t < 0.8 - 1e-12
        d = min(dt, 0.8 - t)
        sync = !(pg.meta.boundary_time == t && pg.meta.sweep_time == t)
        ts(eq, pg, t, d); t += d; k += 1
        dev = (sumV(pg) - face(t)) / h
        if abs(dev - prev) > 0.3
            xs = sort([pg.core.positions[i][1] for i in interior(pg)])
            @printf("  step %d t=%.4f: ΣV-|Ω| = %.2f h (was %.2f), sync before %s, face %.4f, last particle %.4f, w=%.3f\n", k, t, dev, prev, sync, face(t), xs[end], 0.8π * cos(8π * t))
        end
        prev = dev
    end
end
