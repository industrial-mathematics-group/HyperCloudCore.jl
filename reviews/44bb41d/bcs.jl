# Behaviour of the non-mirror conditions with gathered ghosts (old vs new): FixedDirichlet inflow, OutflowBC,
# ExtrapolationBC, linear advection and Burgers, 1-D and 2-D
include("common.jl")
import HyperCloudCore: HyperbolicPDE
struct Burgers{T} <: HyperbolicPDE{1, 1, T} end
HyperCloudCore.flux(::Burgers{T}, U::State) where {T} = Flux{1, 1, T}((State{1, T}(U[1]^2 / 2),))
HyperCloudCore.max_eigenvalue(::Burgers, U::State, d::Int) = abs(U[1])
HyperCloudCore.velocity(::Burgers{T}, U::State, d::Int) where {T} = SMatrix{1, 1, T}(U[1])
HyperCloudCore.prim2cons(::Burgers, U::State) = U
HyperCloudCore.cons2prim(::Burgers, U::State) = U

lim(name) = name == :none ? NoLimiter() : BarthJespersenLimiter()
function run1d(eq, order, limname, bcL, bcR, u_init, tend, N)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = BCD(1 => bcL, 2 => bcR))
    pg = ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, (order == 5 ? 3.5 : 2.5) * h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u_init(pg.core.positions[i][1])); end
    interp = MUSCL(Float64, 1, 1, order, lim(limname), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    t = 0.0
    while t < tend - 1e-12
        dt = 0.4 * get_time_step(pg, eq, interp); d = min(dt, tend - t); ts(eq, pg, t, d); t += d
    end
    I = interior(pg)
    pg, I, [pg.rhos[i][1] for i in I], [pg.core.positions[i][1] for i in I]
end
lin = LinearAdvection(((1.0,),), Float64)
g(x) = exp(-(x - 0.5)^2 / 0.005)
println("== 1-D linear advection a = 1, Gaussian at 0.5; left FixedDirichlet (0), right outflow condition")
for order in (3, 5), limn in (:none, :bj), (bn, bcR) in (("Outflow", OutflowBC()), ("Extrap2", ExtrapolationBC(2)))
    r = map((80, 160)) do N
        _, _, u4, x4 = run1d(lin, order, limn, FixedDirichlet(), bcR, g, 0.4, N)
        e4 = maximum(abs.(u4 .- g.(x4 .- 0.4)))
        _, _, u12, _ = run1d(lin, order, limn, FixedDirichlet(), bcR, g, 1.2, N)
        (e4, maximum(abs.(u12)))
    end
    @printf("MUSCL%d %-4s %-8s  err t=0.4 (pulse at 0.9) N=80/160: %.3e %.3e | residual max|u| t=1.2: %.3e %.3e\n",
            order, limn, bn, r[1][1], r[2][1], r[1][2], r[2][2])
    flush(stdout)
end
println("== 1-D inflow step: ghost state 1 at x < 0 (FixedDirichlet), interior 0, a = 1, t = 0.5: max(u) - 1, min(u), err vs step smeared (L1)")
for order in (3, 5), limn in (:none, :bj)
    _, _, u, x = run1d(lin, order, limn, FixedDirichlet(), OutflowBC(), x -> x < 0 ? 1.0 : 0.0, 0.5, 160)
    @printf("MUSCL%d %-4s  overshoot %.3e  undershoot %.3e  L1 %.3e\n", order, limn, maximum(u) - 1, minimum(u), sum(abs.(u .- (x .< 0.5))) / 160)
end
println("== 1-D Burgers: u = 1 for x < 0.6, 0 after (shock speed 0.5, leaves at t = 0.8); left Dirichlet 1, right outflow; t = 0.4 and 1.0")
for order in (3, 5), limn in (:none, :bj), (bn, bcR) in (("Outflow", OutflowBC()), ("Extrap2", ExtrapolationBC(2)))
    ok = true
    s = try
        _, _, u4, x4 = run1d(Burgers{Float64}(), order, limn, FixedDirichlet(), bcR, x -> x < 0.6 ? 1.0 : 0.0, 0.4, 160)
        _, _, u10, _ = run1d(Burgers{Float64}(), order, limn, FixedDirichlet(), bcR, x -> x < 0.6 ? 1.0 : 0.0, 1.0, 160)
        @sprintf("t=0.4 max %.4f min %.4f L1 %.3e | t=1.0 max|u-1| %.3e", maximum(u4), minimum(u4), sum(abs.(u4 .- (x4 .< 0.8))) / 160, maximum(abs.(u10 .- 1)))
    catch e
        "ERROR " * sprint(showerror, e)[1:min(end, 120)]
    end
    @printf("MUSCL%d %-4s %-8s  %s\n", order, limn, bn, s)
    flush(stdout)
end
println("== 2-D: Gaussian moving with (1, 0.5) out of the unit box, all walls OutflowBC / ExtrapolationBC(2), N = 40/80")
u2(x, t) = exp(-((x[1] - 0.5 - t)^2 + (x[2] - 0.5 - 0.5t)^2) / 0.01)
lin2 = LinearAdvection(((1.0,), (0.5,)), Float64)
function run2d(bc, N, tend; order = 3)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); bc_map = BCD(1 => bc, 2 => bc, 3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u2(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(lin2, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, lin2, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(lin2, pg, t, d); t += d; end
    maximum(abs(pg.rhos[i][1] - u2(pg.core.positions[i], t)) for i in interior(pg))
end
for (bn, bc) in (("Outflow", OutflowBC()), ("Extrap2", ExtrapolationBC(2)))
    r = [(run2d(bc, N, 0.4), run2d(bc, N, 1.0)) for N in (40, 80)]
    @printf("%-8s err t=0.4 (pulse at the wall) N=40/80: %.3e %.3e | t=1.0 (gone): %.3e %.3e\n", bn, r[1][1], r[2][1], r[1][2], r[2][2])
    flush(stdout)
end
