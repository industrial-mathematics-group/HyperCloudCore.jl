include("common.jl")
function tube(N, motion, bc_face; bc_left = FixedDirichlet(), L = 2.0)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), motion)
    geom   = GeometricDomain(intersect(Tagged(Cuboid((0.0,), (L,)), 1), piston); mins = (0.0,), maxs = (L,),
                             bc_map = BCD(1 => bc_left, 2 => bc_face))
    return ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, 2.5h), 1), h
end
function oscill(A, ω)
    t -> RigidMotion(SVector(A * sin(ω * t)), SMatrix{1,1}(1.0), SVector(A * ω * cos(ω * t)), SMatrix{1,1}(0.0))
end
function scalar_run(N, motion, bc, u0, tend; face = t -> 1.0, cfl = 0.4)
    eq = LinearAdvection(((1.0,),), Float64)
    pg, h = tube(N, motion, bc)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i][1], 0.0)); end
    interp = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = cfl * get_time_step(pg, eq, interp); t = 0.0
    m0 = interior_mass(pg)[1]; nins = 0; nrem = 0
    while t < tend - 1e-12
        d = min(dt, tend - t); ts(eq, pg, t, d); t += d
    end
    inner = interior(pg)
    xs = sort([pg.core.positions[i][1] for i in inner])
    err = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i][1], t)) for i in inner)
    l1 = sum(pg.core.volumes[i] * abs(pg.rhos[i][1] - u0(pg.core.positions[i][1], t)) for i in inner)
    @printf("  N=%4d err=%.4g L1=%.3g ΣV=%.6f face=%.6f Δm=%.4f n=%d gap_last=%.3fh maxgap=%.2fh mingap=%.2fh\n", N, err, l1, sumV(pg), face(t),
            interior_mass(pg)[1] - m0, length(inner), (face(t) - xs[end]) / h, maximum(diff(xs)) / h, minimum(diff(xs)) / h)
    return err
end
println("== receding piston w = 1 (commit: 0.19/0.060/0.014), t = 0.5, MirrorBC(1)")
u0r(x, t) = 1 + exp(-((x - 0.8 - t) / 0.08)^2)
for N in (50, 100, 200); scalar_run(N, translating((1.0,)), MirrorBC(; degree = 1), u0r, 0.5; face = t -> 1 + t); end
println("== advancing piston w = -0.5 against the flow a = 1 (outflow through the moving face), t = 0.5, pulse exits")
for bc in (ExtrapolationBC(1), MirrorBC(; degree = 1))
    println(" ", typeof(bc))
    for N in (50, 100, 200); scalar_run(N, translating((-0.5,)), bc, u0r, 0.5; face = t -> 1 - 0.5t); end
end
println("== oscillating piston x_p = 1 + 0.05 sin(4π t) (|w| ≤ 0.63 < a), pulse through the face, t = 0.8, ExtrapolationBC(1)")
u0o(x, t) = 1 + exp(-((x - 0.5 - t) / 0.08)^2)
for N in (50, 100, 200); scalar_run(N, oscill(0.05, 4π), ExtrapolationBC(1), u0o, 0.8; face = t -> 1 + 0.05sin(4π * t)); end
println("== same with a fast piston (|w| dt > dx/4): A = 0.1, ω = 8π, |w|max = 2.5, CFL 0.4")
for N in (50, 100); scalar_run(N, oscill(0.1, 8π), ExtrapolationBC(1), u0o, 0.8; face = t -> 1 + 0.1sin(8π * t)); end
