include("common.jl")
h = 1/40
W = ExponentialWeightFunction(1.0, 2.5h)
println("== (b) step started at t0 = 1 (translating plate)")
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
mk() = begin
    plate = Moving(Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4), translating((1.0, 0.0)))
    c = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
    g = GeometricDomain(c; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => MirrorBC(; degree = 1) for t in 3:8))
    p = ParticleGrid(g, (h, h), W, 1)
    for i in 1:p.meta.N; p.rhos[i] = State{1,Float64}(1.0); end
    p, c
end
pg, c = mk()
interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
t0 = 0.3; dt = 0.005
x0 = [pg.core.positions[i] for i in ghosts(pg)]
try
    ts(eq, pg, t0, dt)
    xg = [pg.core.positions[i] for i in ghosts(pg)]
    shift = maximum(abs(xg[k][1] - x0[k][1]) for k in eachindex(xg))
    println("  ghost displacement after one step from t0=$t0: ", shift, " (plate at t0+dt is at ", t0+dt, ")")
    inner = interior(pg)
    sdp = [signed_distance(at_time(c, t0+dt), pg.core.positions[i]) for i in inner]
    println("  interior particles inside the plate at t0+dt: ", count(>(0), sdp), "; ΣV = ", sumV(pg), " |Ω| = ", 2 - 0.04)
    println("  boundary_time = ", pg.meta.boundary_time, " sweep_time = ", pg.meta.sweep_time)
catch e
    println("  ERR ", sprint(showerror, e)[1:min(end,300)])
end
