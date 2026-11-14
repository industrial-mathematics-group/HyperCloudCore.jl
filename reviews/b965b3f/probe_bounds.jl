# (1) The bounds of the new SlipWall: pressure exponent clamp vs the unbounded tangential correction, for convex and
#     concave walls and large d|κ| (incl. the FD false-positive curvatures of probe_kink: |κ| = 35 F32, 5e3 F64).
# (2) Continuity at the early return: a disc translating with the gas and spinning (X4 state), random wall points,
#     F64 / F32: ghost vs the reflected uniform state, and whether the early return is taken.
using HyperCloud, StaticArrays, LinearAlgebra, Printf, Random
const γ = 1.4
for T in (Float64,)
    eq = HyperCloud.EulerEquation(Val(2), T, T(γ))
    println("(1) static wall at x = 0, n = (-1, 0) (fluid at x < 0 ... the ghost behind), tangential flow u_t = 1, ρ = p = 1 (M_t = 0.85)")
    @printf("  %-9s %-8s %-10s %-10s %-10s %-12s %-12s\n", "κ", "d", "2dκ", "p'/p", "u_t'/u_t", "ρ'", "H0'/H0")
    for κ in (-1 / 0.15, 1 / 0.15, -35.0, 35.0, -5e3), d in (0.005, 0.01, 0.025, 0.05)
        w = WallPoint{2, T, 4}(SVector{2, T}(0, 0), SVector{2, T}(-1, 0), T(d), zero(SVector{2, T}), zero(SVector{2, T}), zero(SMatrix{2, 2, T}),
                               SMatrix{2, 2, T}(0, 0, 0, κ))
        U = prim2cons(eq, State{4, T}(1, 0, 1, 1))
        V = cons2prim(eq, SlipWall()(eq, U, w))
        H0(v) = γ / (γ - 1) * v[4] / v[1] + (v[2]^2 + v[3]^2) / 2
        @printf("  %-9.3g %-8.3g %-10.3g %-10.4f %-10.4f %-12.4g %-12.4g\n", κ, d, 2d * κ, V[4], V[3], V[1], H0(V) / H0(cons2prim(eq, U)))
    end
end
println("\n(2) X4 state: disc r = 0.15 centred at c, translating with V = (0.5, 0) and spinning ω = 2; uniform gas u = V")
J(T) = SMatrix{2, 2, T}(0, 1, -1, 0)
for T in (Float64, Float32)
    eq = HyperCloud.EulerEquation(Val(2), T, T(γ))
    rng = MersenneTwister(1); worst = zero(T); nearly = 0; early = 0
    V = SVector{2, T}(0.5, 0); Ω = 2 * J(T); r = T(0.15)
    for k in 1:10000
        c = SVector{2, T}(rand(rng) - 0.5, rand(rng) - 0.5); θ = T(2π * rand(rng)); rh = SVector(cos(θ), sin(θ))
        xw = c + r * rh; n = -rh; d = T(0.03 * rand(rng))
        H = -(one(SMatrix{2, 2, T}) - rh * rh') / r
        w = V + Ω * (xw - c); a = Ω * (Ω * (xw - c))
        wp = WallPoint{2, T, 4}(xw, n, d, w, a, Ω, H)
        U = prim2cons(eq, State{4, T}(1, V..., 1))
        G = cons2prim(eq, SlipWall()(eq, U, wp))
        Gexact = SVector(V - 2 * dot(V - w, n) * n)    # reflected relative normal velocity (V⋅n = w⋅n here: = V)
        e = max(abs(G[1] - 1), abs(G[4] - 1), norm(SVector(G[2], G[3]) - V))
        worst = max(worst, e)
        # the early-return condition, recomputed as in the operator
        ur = V - w; un = dot(ur, n); ut = ur - un * n
        dpdn = -(dot(a, n) + 2 * dot(n, Ω * ur) - dot(ut, H * ut)); ωn = Ω' * n; dut = -2d * (H * ut - (ωn - dot(ωn, n) * n))
        (iszero(dpdn) && iszero(dut)) && (early += 1)
    end
    @printf("  %-8s 10000 wall points: early return taken %d times; max ghost deviation from the uniform state %.2e (eps %.1e)\n", T, early, worst, eps(T))
end
