# Kinematic check of SlipWall's tangential correction at rigidly moving curved walls: incompressible potential flow
# around a disc / sphere / cylinder that translates and rotates about an OFF-CENTRE axis (so the spin term
# P_t(Ωᵀn) does not cancel against H w_t). Mirror state = exact flow at x_w - d n, compared with the exact flow at
# x_w + d n. Velocity error and pressure error (unsteady Bernoulli, ρ = 1, p0 = 50 so the log form ≈ linear).
using HyperCloud, StaticArrays, LinearAlgebra, Printf
const γ = 1.4
# doublet of a moving body: φ = -k U⋅r / |r|^m (2-D: k = a², m = 2; 3-D sphere: k = a³/2, m = 3; cylinder: 2-D in xy)
function flow(x, c, U, Udot, a, D, cyl)
    r = x - c
    P = cyl ? SMatrix{3,3}(1.0,0,0, 0,1.0,0, 0,0,0) : one(SMatrix{D, D})
    r = P * r; Ue = P * U; Ude = P * Udot
    m, k = (D == 2 || cyl) ? (2, a^2) : (3, a^3 / 2)
    R = norm(r)
    φ(Uv) = -k * dot(Uv, r) / R^m
    v = -k * (Ue / R^m - m * dot(Ue, r) * r / R^(m + 2))
    dφdt = -dot(U, v) + φ(Ude)          # ∂φ/∂t = -U⋅∇φ (moving centre) + φ with U̇
    p = 50.0 - (dφdt + sum(abs2, v) / 2)
    return v, p
end
stateof(v, p) = (D = length(v); prim2cons(HyperCloud.EulerEquation(Val(D), Float64, γ), State{D + 2, Float64}(1.0, v..., p)))
function run(name, D, c, c0, Ω, a, θ; cyl = false, Vt = zero(SVector{D}))
    println(name)
    eq = HyperCloud.EulerEquation(Val(D), Float64, γ)
    U = Ω * (c - c0) + Vt; Udot = Ω * (Ω * (c - c0))   # centre velocity (translation Vt + rotation about c0)
    rh = D == 2 ? SVector(cos(θ), sin(θ)) : SVector(cos(θ) * 0.8, sin(θ) * 0.8, 0.6)
    cyl && (rh = SVector(cos(θ), sin(θ), 0.0))
    xw = c + a * rh; n = -rh
    Ptan = cyl ? I - rh * rh' - SVector(0.0, 0, 1) * SVector(0.0, 0, 1)' : I - rh * rh'
    H = SMatrix{D, D}(-Ptan / a)
    w = U + Ω * (xw - c)            # rigid velocity at the wall point: rotation about c0 = translation of c + spin about c
    acc = Udot + Ω * (Ω * (xw - c))
    @printf("  %-7s %-11s %-11s %-11s %-11s %-11s\n", "d", "|Δu| new", "/d²", "|Δp| new", "/d²", "|Δu| even")
    for d in (0.04, 0.02, 0.01, 0.005)
        vm, pm = flow(xw - d * n, c, U, Udot, a, D, cyl); ve, pe = flow(xw + d * n, c, U, Udot, a, D, cyl)
        wp = WallPoint{D, Float64, D * D}(xw, n, d, w, acc, Ω, H)
        V = cons2prim(eq, SlipWall()(eq, stateof(vm, pm), wp))
        V0 = cons2prim(eq, SlipWall()(eq, stateof(vm, pm), WallPoint(xw, n, d, w)))
        du = norm(SVector(ntuple(i -> V[i + 1], D)) - ve); dp = abs(V[D + 2] - pe)
        du0 = norm(SVector(ntuple(i -> V0[i + 1], D)) - ve)
        @printf("  %-7.3f %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n", d, du, du / d^2, dp, dp / d^2, du0)
    end
end
J = SMatrix{2, 2}(0.0, 1.0, -1.0, 0.0)
c = SVector(0.4, 0.1); c0 = SVector(-0.3, 0.5)
run("2-D disc a = 0.5 translating, U = (0.3, 0.2)", 2, c, c, zero(J), 0.5, 0.7; Vt = SVector(0.3, 0.2))
run("2-D disc a = 0.5 rotating ω = 1.5 about an off-centre axis c0", 2, c, c0, 1.5J, 0.5, 0.7)
run("2-D disc, same, other wall point", 2, c, c0, 1.5J, 0.5, 2.9)
run("2-D disc rotating about c0 and translating", 2, c, c0, -1.0J, 0.5, -1.2; Vt = SVector(-0.2, 0.4))
Ω3 = SMatrix{3, 3}(0.0, 0.9, -0.4, -0.9, 0.0, 0.7, 0.4, -0.7, 0.0)     # skew: ω = (-0.7, -0.4, 0.9)
c3 = SVector(0.2, -0.1, 0.3); c30 = SVector(-0.4, 0.3, 0.1)
run("3-D sphere a = 0.5 translating", 3, c3, c3, zero(Ω3), 0.5, 0.4; Vt = SVector(0.3, -0.2, 0.25))
run("3-D sphere rotating about an off-centre oblique axis", 3, c3, c30, Ω3, 0.5, 0.4)
Ωz = SMatrix{3, 3}(0.0, 1.2, 0, -1.2, 0.0, 0, 0, 0, 0.0)
run("3-D cylinder (axis z) translating obliquely, U = (0.3, -0.2, 0.5)", 3, c3, c3, zero(Ωz), 0.5, 0.4; cyl = true, Vt = SVector(0.3, -0.2, 0.5))
run("3-D cylinder rotating about a parallel off-centre axis + axial translation", 3, c3, c30, Ωz, 0.5, 0.4; cyl = true, Vt = SVector(0.0, 0.0, 0.5))
