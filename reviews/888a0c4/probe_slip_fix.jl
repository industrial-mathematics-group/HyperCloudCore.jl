# A corrected tangential treatment (relative tangential velocity + spin term, from ∂v_t/∂n = -κ (v_t - w_t) + n·Ω t
# for an irrotational lab-frame flow at a rigidly moving wall), checked on the cases of probe_slip_op.jl plus a
# rotation about an off-centre point.
include(joinpath(@__DIR__, "probe_slip_op_defs.jl"))
function fixed(eq, U, wall)
    V = cons2prim(eq, U); ρ, p = V[1], V[4]
    u = SVector(V[2], V[3]); n = wall.normal
    ur = u - wall.velocity; un = dot(ur, n); ut = ur - un * n
    dpdn = -ρ * (dot(wall.acceleration, n) + 2 * dot(n, wall.spin * ur) - dot(ut, wall.hessian * ut))
    p2 = max(p + 2 * wall.distance * dpdn, 0.01p)
    ρ2 = ρ * (p2 / p)^(1 / eq.gamma)
    t = SVector(-n[2], n[1]); κ = dot(t, wall.hessian * t)
    q2 = sum(abs2, ut)
    s = q2 > 0 ? sqrt(max(0.0, 1 - 4 * wall.distance * dot(ut, wall.hessian * ut) / q2)) : 1.0
    wt = wall.velocity - dot(wall.velocity, n) * n
    spin = 2 * wall.distance * dot(n, wall.spin * t) * t
    u2 = wt + s * ut + spin + (2 * dot(wall.velocity, n) - dot(u, n)) * n
    return prim2cons(eq, State{4, Float64}(ρ2, u2..., p2))
end
function case2(name, xw, n, H, wvel, Ω, acc, c, Γ, V)
    println(name)
    @printf("  %-8s %-12s %-12s %-12s %-12s\n", "d", "err code", "err/d²", "err fixed", "err/d²")
    for d in (0.04, 0.02, 0.01, 0.005)
        Um = vortex(xw - d * n, c, Γ, V); Ue = vortex(xw + d * n, c, Γ, V)
        w  = WallPoint{2, Float64, 4}(xw, n, d, wvel, acc, Ω, H)
        e1 = err(SlipWall()(eq, Um, w), Ue); e2 = err(fixed(eq, Um, w), Ue)
        @printf("  %-8.3f %-12.3e %-12.3e %-12.3e %-12.3e\n", d, e1, e1 / d^2, e2, e2 / d^2)
    end
end
case2("(a) static outer circle", rh, rh, Hout, z, Z, z, z, 0.5, z)
case2("(c) outer circle translating, vortex + V", rh, rh, Hout, V, Z, z, z, 0.5, V)
case2("(d) inner disc translating, vortex + V", 0.5rh, -rh, Hin, V, Z, z, z, 0.5, V)
case2("(e) inner disc spinning about its centre, vortex", 0.5rh, -rh, Hin, Ω * (0.5rh), Ω, Ω * Ω * (0.5rh), z, 0.5, z)
# (h) the vortex flow in the frame of a disc rotating about its centre while the centre translates (rigid motion
#     w(x) = V + Ω (x - c)): exact flow = vortex + V, wall velocity V + Ω(x_w - c), acceleration Ω²(x_w - c)
case2("(h) inner disc translating with V and spinning (ω = 2), vortex + V", 0.5rh, -rh, Hin, V + Ω * (0.5rh), Ω, Ω * Ω * (0.5rh), z, 0.5, V)
