include("common.jl")
function stepper_run(pg, eq, nsteps, dt; T = Float64)
    D = length(pg.meta.dx)
    interp = MUSCL(T, D, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(T))
    t = zero(T)
    for _ in 1:nsteps; ts(eq, pg, t, T(dt)); t += T(dt); end
    t
end
println("== ball leaving the canvas through a static wall (box [0,1]², ball r = 0.1 moving +x at 1, t = 0.7)")
try
    h = 1/40
    ball = Moving(Tagged(Ball((0.5, 0.5), 0.1), 5), translating((1.0, 0.0)))
    geom = GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0)) - ball; mins = (0.0, 0.0), maxs = (1.0, 1.0), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0 + pg.core.positions[i][2]); end
    eq = LinearAdvection(((0.0,), (0.0,)), Float64)
    t = stepper_run(pg, eq, 112, 0.00625)
    inner = interior(pg)
    @printf("  ok t=%.3f: ΣV=%.5f |Ω(t)|≈%.5f; ghosts outside the canvas: %d; interior within the ball: %d; U in [%.4f, %.4f] (init [1, 2])\n", t, sumV(pg),
            1 - 0.5 * π * 0.01 + 0.0, count(i -> pg.core.is_boundary[i] && any(pg.core.positions[i] .> pg.domain.canvas_maxs), 1:pg.meta.N),
            count(i -> norm(pg.core.positions[i] - SVector(0.5 + t, 0.5)) < 0.1, inner), minimum(pg.rhos[i][1] for i in inner), maximum(pg.rhos[i][1] for i in inner))
catch e
    println("  ERR ", sprint(showerror, e)[1:min(end, 400)])
end
println("== two balls 3h apart moving towards each other (tags 5, 6), at rest fluid, until 1h gap")
try
    h = 1/40
    b1 = Moving(Tagged(Ball((0.35, 0.5), 0.1), 5), translating((0.5, 0.0)))
    b2 = Moving(Tagged(Ball((0.65 + 0.075, 0.5), 0.1), 6), translating((-0.5, 0.0)))
    geom = GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0)) - b1 - b2; mins = (0.0, 0.0), maxs = (1.0, 1.0), bc_map = BCD(t => MirrorBC(; degree = 1) for t in 1:6))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    println("  piece_of_tag = ", geom.piece_of_tag)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0 + pg.core.positions[i][2]); end
    eq = LinearAdvection(((0.0,), (0.0,)), Float64)
    # gap initially 0.65+0.075-0.1 - (0.35+0.1) = 0.175 = 7h; close to 1h: t = 0.15
    t = stepper_run(pg, eq, 48, 0.003125)
    inner = interior(pg)
    gap = (0.725 - 0.5t - 0.1) - (0.35 + 0.5t + 0.1)
    @printf("  ok t=%.3f gap=%.3fh: ΣV=%.5f exact %.5f; interior in the gap (|y-0.5|<h): %d; U in [%.4f, %.4f]\n", t, gap / h, sumV(pg), 1 - 2π * 0.01,
            count(i -> abs(pg.core.positions[i][2] - 0.5) < h && 0.45 + 0.5t < pg.core.positions[i][1] < 0.625 - 0.5t, inner),
            minimum(pg.rhos[i][1] for i in inner), maximum(pg.rhos[i][1] for i in inner))
catch e
    println("  ERR ", sprint(showerror, e)[1:min(end, 400)])
end
println("== 3-D translating ball, fluid at rest, h = 1/16, 20 steps")
try
    h = 1/16
    ball = Moving(Tagged(Ball((0.4, 0.5, 0.5), 0.2), 7), translating((0.5, 0.0, 0.0)))
    geom = GeometricDomain(Cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) - ball; mins = (0.0, 0.0, 0.0), maxs = (1.0, 1.0, 1.0), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:7))
    pg = ParticleGrid(geom, (h, h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0 + pg.core.positions[i][2]); end
    eq = LinearAdvection(((0.0,), (0.0,), (0.0,)), Float64)
    V0 = sumV(pg)
    t = stepper_run(pg, eq, 20, 0.025)
    inner = interior(pg)
    @printf("  ok t=%.3f: ΣV=%.5f (t=0: %.5f, exact %.5f); interior inside/within h/4 of the ball: %d; U in [%.4f, %.4f]\n", t, sumV(pg), V0, 1 - 4/3*π*0.008,
            count(i -> norm(pg.core.positions[i] - SVector(0.4 + 0.5t, 0.5, 0.5)) < 0.2 + h/4 - 1e-9, inner), minimum(pg.rhos[i][1] for i in inner), maximum(pg.rhos[i][1] for i in inner))
catch e
    println("  ERR ", sprint(showerror, e)[1:min(end, 400)])
end
println("== Float32 plate translating (translating with Float32 velocity), 20 steps")
try
    h = 1f0/40
    plate = Moving(Tagged(Cuboid((-0.2f0, -0.05f0), (0.2f0, 0.05f0)), t -> t + 4), translating((1f0, 0f0)))
    chan = Cuboid((-1f0, -0.5f0), (1f0, 0.5f0); open = (true, false)) - plate
    geom = GeometricDomain(chan; is_periodic = (true, false), mins = (-1f0, -0.5f0), maxs = (1f0, 0.5f0), bc_map = BCD(t => MirrorBC(; degree = 1) for t in 3:8))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1f0, 2.5f0h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float32}(1f0); end
    eq = LinearAdvection(((1f0,), (0f0,)), Float32)
    t = stepper_run(pg, eq, 40, 0.005; T = Float32)
    inner = interior(pg)
    @printf("  ok t=%.4f boundary_time=%.6f: ΣV=%.5f (exact 1.96); U in [%.6f, %.6f]\n", t, pg.meta.boundary_time, sumV(pg), minimum(pg.rhos[i][1] for i in inner), maximum(pg.rhos[i][1] for i in inner))
catch e
    println("  ERR ", sprint(showerror, e)[1:min(end, 600)])
end
