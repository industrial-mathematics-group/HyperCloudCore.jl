# Accuracy at a co-moving receding piston face: data consistent with the face condition, and ExtrapolationBC(1)
include("common.jl")
Ns2 = parse.(Int, split(get(ENV, "NS2", "20,40,80"), ","))
function run2d(N, u0, bcface; tend = 0.5)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5))
    geom = GeometricDomain(intersect(box, piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => bcface))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4h; t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq, pg, t, d); t += d; end
    face = 1 + t
    inner = [i for i in interior(pg) if pg.core.positions[i][1] > t + 0.2]
    e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner]
    xs = [pg.core.positions[i][1] for i in inner]
    ewall = maximum(e[k] for k in eachindex(e) if xs[k] > 1.0 + 3h && xs[k] < face - 3h; init = 0.0)
    @printf("   N=%3d max err %.4g; beyond 3h of the face %.4g; in the uncovered part (1+3h < x < face-3h) %.4g; ΣV %.6f (%.6f)\n",
            N, maximum(e), maximum(e[k] for k in eachindex(e) if xs[k] < face - 3h), ewall, sumV(pg), 0.5face)
end
uc(x, t) = 1 + 0.5 * cos(2π * (x[1] - t - 1)) * cos(2π * x[2])     # zero normal derivative on the face and the walls
println("== 2-D receding co-moving piston, consistent data (∂n u = 0 on face and walls), MirrorBC(1) everywhere, t = 0.5")
for N in Ns2; run2d(N, uc, MirrorBC(; degree = 1)); end
println("== same, ExtrapolationBC(1) on the face")
for N in Ns2; run2d(N, uc, ExtrapolationBC(1)); end
# 1-D
function run1d(N, u0, bcface; tend = 0.5)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((1.0,)))
    geom   = GeometricDomain(intersect(Tagged(Cuboid((0.0,), (2.0,)), 1), piston); mins = (0.0,), maxs = (2.0,), bc_map = BCD(1 => FixedDirichlet(), 2 => bcface))
    pg = ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i][1], 0.0)); end
    eq = LinearAdvection(((1.0,),), Float64)
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4h; t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq, pg, t, d); t += d; end
    inner = [i for i in interior(pg) if pg.core.positions[i][1] > t + 0.2]
    e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i][1], t)) for i in inner]
    k = argmax(e)
    @printf("   N=%4d max err %.4g at x = %.4f (face %.4f)\n", N, e[k], pg.core.positions[inner[k]][1], 1 + t)
end
u1(x, t) = 1 + 0.5 * sin(2π * (x - t))
u1c(x, t) = 1 + 0.5 * cos(2π * (x - t - 1))
for (lbl, u, bc) in (("ExtrapolationBC(1), data with ∂n u ≠ 0", u1, ExtrapolationBC(1)), ("ExtrapolationBC(1), ∂n u = 0 at the face", u1c, ExtrapolationBC(1)),
                     ("MirrorBC(1), ∂n u ≠ 0", u1, MirrorBC(; degree = 1)), ("MirrorBC(1), ∂n u = 0", u1c, MirrorBC(; degree = 1)))
    println("== 1-D receding co-moving piston, ", lbl)
    for N in (20, 40, 80, 160); run1d(N, u, bc); end
end
