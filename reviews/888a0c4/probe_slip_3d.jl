# 3-D: a vortex around the z axis plus a uniform axial flow W inside the cylinder r = 1 (exact steady, irrotational,
# homentropic Euler solution). The wall's Hessian is anisotropic (1/r around, 0 along z).
using HyperCloud, StaticArrays, LinearAlgebra, Printf
const γ = 1.4
eq = HyperCloud.EulerEquation(Val(3), Float64, γ)
function flow(x, Γ, W; H0 = 4.0)
    r = SVector(x[1], x[2], 0.0); ρr = norm(r); th = SVector(-x[2], x[1], 0.0) / ρr
    u = Γ / ρr * th + SVector(0.0, 0.0, W)
    ρ = ((γ - 1) / γ * (H0 - sum(abs2, u) / 2))^(1 / (γ - 1)); p = ρ^γ
    return State{5, Float64}(ρ, (ρ * u)..., p / (γ - 1) + ρ * sum(abs2, u) / 2)
end
prim(U) = (ρ = U[1]; u = SVector(U[2], U[3], U[4]) / ρ; (ρ, u, (γ - 1) * (U[5] - ρ * sum(abs2, u) / 2)))
θ = 0.4; rh = SVector(cos(θ), sin(θ), 0.0); xw = rh; n = rh
H = SMatrix{3, 3}(I - rh * rh' - SVector(0.0, 0.0, 1.0) * SVector(0.0, 0.0, 1.0)')
for (Γ, W) in ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    println("Γ = $Γ, W = $W")
    @printf("  %-8s %-12s %-12s %-12s %-12s %-12s\n", "d", "err", "err/d²", "|Δu_z|", "|Δu_θ|", "even err")
    for d in (0.04, 0.02, 0.01, 0.005)
        Um = flow(xw - d * n, Γ, W); Ue = flow(xw + d * n, Γ, W)
        w = WallPoint{3, Float64, 9}(xw, n, d, zero(SVector{3}), zero(SVector{3}), zero(SMatrix{3, 3}), H)
        a, b = prim(SlipWall()(eq, Um, w)), prim(Ue)
        e = max(abs(a[1] - b[1]), norm(a[2] - b[2]), abs(a[3] - b[3]))
        c = prim(SlipWall()(eq, Um, WallPoint(xw, n, d, zero(SVector{3}))))
        e0 = max(abs(c[1] - b[1]), norm(c[2] - b[2]), abs(c[3] - b[3]))
        th = SVector(-rh[2], rh[1], 0.0)
        @printf("  %-8.3f %-12.3e %-12.3e %-12.3e %-12.3e %-12.3e\n", d, e, e / d^2, abs(a[2][3] - b[2][3]), abs(dot(a[2] - b[2], th)), e0)
    end
end
