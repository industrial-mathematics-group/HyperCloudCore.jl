# SlipWall ghosts around a square obstacle given as a user shape (exact SDF, FD Hessian) in a uniform flow u = (0.5, 0.2)
using HyperCloud, StaticArrays, LinearAlgebra, Printf
const HCC = HyperCloud.HyperCloudCore
struct USquare{T} <: HCC.AbstractShape{2, T}; a::T; end
function HCC.signed_distance(s::USquare{T}, x::SVector{2, T}) where {T}
    q = abs.(x) .- s.a
    return norm(max.(q, zero(T))) + min(max(q[1], q[2]), zero(T))
end
HCC.bounding_box(s::USquare) = ((-s.a, -s.a), (s.a, s.a))
const γ = 1.4
eq = HyperCloud.EulerEquation(Val(2), Float64, γ)
BCD = Dict{Int, HCC.AbstractBoundaryCondition}
for (label, obst) in (("user square (FD Hessian)", Tagged(USquare(0.25), 5)), ("built-in Cuboid", Tagged(Cuboid((-0.25, -0.25), (0.25, 0.25)), 5)))
    sh = Cuboid((-1.0, -1.0), (1.0, 1.0)) - obst
    geom = GeometricDomain(sh; bc_map = BCD(t => slip_wall() for t in 1:5))
    pg = ParticleGrid(geom, (0.02, 0.02), ExponentialWeightFunction(1.0, 0.05), 4)
    U0 = State{4, Float64}(1.0, 0.5, 0.2, 1 / (γ - 1) + 0.5 * (0.25 + 0.04))
    fill!(pg.rhos, U0)
    buf = copy(pg.rhos)
    slip_wall()(pg, buf, 5, nothing, eq, 0.0)
    idx = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 5]
    P = [cons2prim(eq, buf[i]) for i in idx]
    sp = [norm(SVector(v[2], v[3])) for v in P]
    @printf("%-26s %d ghosts: ρ in [%.3f, %.3f], p in [%.3f, %.3f], |u| in [%.3f, %.3f]\n", label, length(idx),
            minimum(v -> v[1], P), maximum(v -> v[1], P), minimum(v -> v[4], P), maximum(v -> v[4], P), minimum(sp), maximum(sp))
end
