include("common.jl")
N = 80; h = 1/N
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
plate_shape = Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4)
function mk(moving)
    p = moving ? Moving(plate_shape, translating((1.0, 0.0))) : plate_shape
    chan = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - p
    geom = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                           bc_map = BCD(t => MirrorBC(; degree = 1) for t in 3:8))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1 + 0.5sin(π * pg.core.positions[i][1])); end
    pg
end
interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
for moving in (false, true)
    pg = mk(moving)
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eq, interp); t = 0.0
    for _ in 1:5; ts(eq, pg, t, dt); t += dt; end
    n = 40
    a = @allocated ts(eq, pg, t, dt); t += dt
    el = @elapsed for _ in 1:n; ts(eq, pg, t, dt); t += dt; end
    @printf("moving=%s N_particles=%d: %.2f ms/step, %.1f kB allocated in one step\n", moving, pg.meta.N, 1e3el/n, a/1024)
    if moving
        X = pg.core.positions
        place_ghosts!(pg, X, pg.meta.boundary_time, pg.meta.boundary_time)
        a1 = @allocated place_ghosts!(pg, X, pg.meta.boundary_time, pg.meta.boundary_time)
        b = pg.meta.boundary_time
        tp = @elapsed for _ in 1:100; place_ghosts!(pg, X, b, b); end
        @printf("  place_ghosts!: %d bytes, %.1f μs per call\n", a1, 1e6tp/100)
        pg.neighbor(pg)
        sweep_boundaries!(pg, b, b, 3)
        as = @allocated sweep_boundaries!(pg, b, b, 3)
        tsw = @elapsed for _ in 1:10; sweep_boundaries!(pg, b, b, 3); end
        @printf("  sweep_boundaries! (no motion): %.1f kB, %.2f ms per call\n", as/1024, 1e3tsw/10)
        bc = MirrorBC(; degree = 1)
        buf = pg.rhos
        bc(pg, buf, 5, ts, eq, b)
        ab = @allocated bc(pg, buf, 5, ts, eq, b)
        @printf("  MirrorBC launch on a moving tag: %d bytes\n", ab)
        # the tm tuple
        tm = H.tag_motions(pg.geometry, 0.0, 0.1)
        @printf("  tag_motions tuple: length %d, %d bytes, isbits %s\n", length(tm), sizeof(tm), isbits(tm))
        a2 = @allocated H.tag_motions(pg.geometry, 0.0, 0.1)
        @printf("  tag_motions allocates %d bytes\n", a2)
    end
end
println("== components (moving)")
pg = mk(true)
ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
dt = 0.5 * get_time_step(pg, eq, interp); t = 0.0
for _ in 1:5; global t; ts(eq, pg, t, dt); t += dt; end
tn = @elapsed for _ in 1:20; pg.neighbor(pg); end
@printf("  neighbour search: %.2f ms\n", 1e3tn/20)
tsw = 0.0; asw = 0
for _ in 1:20
    global t, tsw, asw
    place_ghosts!(pg, pg.core.positions, t, t + dt); pg.neighbor(pg)
    asw += @allocated begin
        tsw += @elapsed sweep_boundaries!(pg, t, t + dt, 3)
    end
    t += dt
end
@printf("  sweep with motion: %.2f ms, %.0f kB per call\n", 1e3tsw/20, asw/20/1024)
tr = @elapsed for _ in 1:20; set_positions!(pg, pg.core.positions, false); end
@printf("  set_positions! (frozen topology, one stage): %.2f ms\n", 1e3tr/20)
