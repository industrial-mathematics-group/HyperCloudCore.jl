# Operator-level consistency of SlipWall: exact smooth Euler solutions next to curved walls, the mirror state taken
# from the exact solution at x_w - d n, the ghost compared with the exact solution at x_w + d n. A consistent
# second-order mirror has error/d² bounded (order 2 in d); the even mirror is order 1.
using HyperCloud, StaticArrays, LinearAlgebra, Printf
const HCC = HyperCloud.HyperCloudCore
const γ = 1.4
eq = HyperCloud.EulerEquation(Val(2), Float64, γ)
# homentropic potential vortex u_θ = Γ/r about c, plus a uniform velocity V (Galilean shift: exact unsteady solution,
# frozen at t = 0 with the vortex centre at c)
function vortex(x, c, Γ, V; H0 = 4.0)
    r = x - c; ρr = norm(r); t = SVector(-r[2], r[1]) / ρr
    u = Γ / ρr * t + V
    ρ = ((γ - 1) / γ * (H0 - Γ^2 / (2ρr^2)))^(1 / (γ - 1)); p = ρ^γ
    return State{4, Float64}(ρ, ρ * u[1], ρ * u[2], p / (γ - 1) + ρ * sum(abs2, u) / 2)
end
prim(U) = (ρ = U[1]; u = SVector(U[2], U[3]) / ρ; (ρ, u, (γ - 1) * (U[4] - ρ * sum(abs2, u) / 2)))
err(A, B) = (a = prim(A); b = prim(B); max(abs(a[1] - b[1]), norm(a[2] - b[2]), abs(a[3] - b[3])))
J = SMatrix{2, 2}(0.0, 1.0, -1.0, 0.0)
function case(name, xw, n, H, wvel, Ω, acc, c, Γ, V)
    println(name)
    @printf("  %-8s %-12s %-12s %-12s %-12s\n", "d", "err new", "err/d²", "err even", "err/d")
    for d in (0.04, 0.02, 0.01, 0.005)
        Um = vortex(xw - d * n, c, Γ, V); Ue = vortex(xw + d * n, c, Γ, V)
        w  = WallPoint{2, Float64, 4}(xw, n, d, wvel, acc, Ω, H)
        w0 = WallPoint(xw, n, d, wvel)
        e1 = err(SlipWall()(eq, Um, w), Ue); e0 = err(SlipWall()(eq, Um, w0), Ue)
        @printf("  %-8.3f %-12.3e %-12.3e %-12.3e %-12.3e\n", d, e1, e1 / d^2, e0, e0 / d)
    end
end
θ = 0.7; rh = SVector(cos(θ), sin(θ)); th = SVector(-rh[2], rh[1])
Hout = (I - rh * rh') / 1.0                         # inside the circle r = 1 (concave from the fluid)
Hin  = -(I - rh * rh') / 0.5                        # outside the disc r = 0.5 (convex)
z = zero(SVector{2}); Z = zero(SMatrix{2, 2})
case("(a) static outer circle r = 1, vortex Γ = 0.5", rh, rh, Hout, z, Z, z, z, 0.5, z)
case("(b) static inner disc r = 0.5, vortex Γ = 0.5", 0.5rh, -rh, Hin, z, Z, z, z, 0.5, z)
V = SVector(0.3, 0.2)
case("(c) outer circle translating with V = (0.3, 0.2), vortex + V (Galilean shift of (a))", rh, rh, Hout, V, Z, z, z, 0.5, V)
case("(d) inner disc translating with V, vortex + V (Galilean shift of (b))", 0.5rh, -rh, Hin, V, Z, z, z, 0.5, V)
ω = 2.0; Ω = ω * J
case("(e) inner disc spinning about its centre (ω = 2), vortex Γ = 0.5", 0.5rh, -rh, Hin, Ω * (0.5rh), Ω, Ω * Ω * (0.5rh), z, 0.5, z)
case("(f) inner disc spinning about its centre (ω = 2), gas at rest", 0.5rh, -rh, Hin, Ω * (0.5rh), Ω, Ω * Ω * (0.5rh), z, 0.0, z)
case("(g) inner disc translating with V through a gas at rest (static fluid, a moving wall: not a solution, skipped)", 0.5rh, -rh, Hin, z, Z, z, z, 0.0, z)
