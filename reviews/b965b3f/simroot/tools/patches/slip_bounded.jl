# Review experiment: f445ff1's SlipWall with the tangential correction bounded like the pressure exponent
# (2 d ‖H‖ ≤ 1/2 by scaling the curvature term; the spin term unchanged). Everything else identical.
Core.eval(HyperCloud, quote
@inline function (::SlipWall)(eq::EulerEquation{D, M, T, <:Conservative}, U::State{M, T}, wall::WallPoint{D, T}) where {D, M, T}
    V  = cons2prim(eq, U)
    ρ, p = V[1], V[M]
    u  = Space{D, T}(ntuple(d -> V[1 + d], Val(D)))
    n  = wall.normal
    ur = u - wall.velocity
    un = sum(ur .* n)
    ut = ur - un * n
    d, H, Ω = wall.distance, wall.hessian, wall.spin
    dpdn = -ρ * (sum(wall.acceleration .* n) + 2 * sum(n .* (Ω * ur)) - sum(ut .* (H * ut)))
    ωn  = Ω' * n
    nH  = maximum(abs, H)
    sH  = 2d * nH > T(0.5) ? T(0.5) / (2d * nH) : one(T)
    dut = -2d * (sH * (H * ut) - (ωn - sum(ωn .* n) * n))
    (iszero(dpdn) && iszero(dut)) && return prim2cons(eq, State{M, T}(ρ, (u - (2 * un) * n)..., p))
    p2 = p * exp(clamp(2d * dpdn / p, -T(0.5), T(0.5)))
    ρ2 = ρ * (p2 / p)^(1 / eq.gamma)
    return prim2cons(eq, State{M, T}(ρ2, (wall.velocity + ut + dut - un * n)..., p2))
end
end)
