include("common.jl")
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = 1 + exp(-((mod(x[1] - t + 1, 2.0) - 0.5)^2 + (x[2] - 0.2)^2) / 0.02)
plate_shape = Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4)
function run(N, tend; report_times = ())
    h = 1 / N
    chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - Moving(plate_shape, translating((1.0, 0.0)))
    geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                            bc_map = BCD(t => MirrorBC(; degree = 1) for t in (3, 4, 5, 6, 7, 8)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eq, interp); t = 0.0
    m0 = interior_mass(pg)[1]
    nsteps = 0
    tel = @elapsed while t < tend - 1e-12
        d = min(dt, tend - t); ts(eq, pg, t, d); t += d; nsteps += 1
        for tr in report_times
            if abs(t - tr) < d/2
                inner = interior(pg)
                sh = at_time(chan, t)
                nin = count(i -> signed_distance(sh, pg.core.positions[i]) > -0.25h, inner)
                gh = ghosts(pg)
                # ghosts of the plate: distance to the plate (periodic image aware)
                gd = maximum(i -> min(signed_distance(plate_shape, pg.core.positions[i] - SVector(t, 0.0)),
                                      signed_distance(plate_shape, pg.core.positions[i] - SVector(t - 2, 0.0))), gh)
                err = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner)
                @printf("   t=%.3f  N_int=%d  interior within h/4 of/inside shape(t)=%d  ΣV=%.6f  max ghost sd to plate=%.3g  err=%.3g  minU=%.4f maxU=%.4f\n",
                        t, length(inner), nin, sumV(pg), gd, err, minimum(pg.rhos[i][1] for i in inner), maximum(pg.rhos[i][1] for i in inner))
            end
        end
    end
    inner = interior(pg)
    err = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner)
    far = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner if abs(pg.core.positions[i][1] - t) > 0.4 || abs(pg.core.positions[i][2]) > 0.25)
    @printf("N=%d t=%.2f err=%.4g far=%.4g ΣV=%.6f |Ω|=1.96 Δm=%.3g minU=%.5f maxU=%.4f steps=%d time=%.1fs\n", N, t, err, far, sumV(pg),
            interior_mass(pg)[1] - m0, minimum(pg.rhos[i][1] for i in inner), maximum(pg.rhos[i][1] for i in inner), nsteps, tel)
end
for N in (20, 40, 80); run(N, 0.5); end
println("== crossing the periodic seam (plate x-max reaches 1 at t = 0.8)")
run(40, 1.4; report_times = (0.5, 0.8, 0.9, 1.0, 1.2, 1.4))
