# The finite-difference Hessian of user shapes (_sdf_hessian fallback): accuracy for a smooth exact SDF (circle),
# Float32 vs Float64, and what the grid stores next to the corners of a user polygon (square obstacle, exact SDF)
using HyperCloudCore, StaticArrays, LinearAlgebra, Printf, Random
const H = HyperCloudCore
struct UCircle{T} <: AbstractShape{2, T}; c::SVector{2, T}; r::T; end
H.signed_distance(s::UCircle{T}, x::SVector{2, T}) where {T} = norm(x - s.c) - s.r
H.bounding_box(s::UCircle) = (Tuple(s.c .- s.r), Tuple(s.c .+ s.r))
struct USquare{T} <: AbstractShape{2, T}; a::T; end          # exact SDF of the square |x|,|y| ≤ a
function H.signed_distance(s::USquare{T}, x::SVector{2, T}) where {T}
    q = abs.(x) .- s.a
    return norm(max.(q, zero(T))) + min(max(q[1], q[2]), zero(T))
end
H.bounding_box(s::USquare) = ((-s.a, -s.a), (s.a, s.a))
for T in (Float64, Float32)
    # pointwise on the circle (radius 0.15 and 0.5, off-origin centre)
    for (c, r) in ((SVector{2, T}(0, 0), T(0.15)), (SVector{2, T}(0.3, -0.2), T(0.5)), (SVector{2, T}(10, 10), T(0.15)))
        errs = map(range(0, 2π; length = 50)) do θ
            n = SVector{2, T}(cos(θ), sin(θ)); x = c + r * n; t = SVector(-n[2], n[1])
            Hm = H._sdf_hessian(UCircle(c, r), x)
            abs(t' * Hm * t - 1 / r) * r
        end
        @printf("%-8s circle c=%s r=%.2f: max rel error of t·Ht over the wall %.2e\n", T, c, r, maximum(errs))
    end
end
# a grid: square obstacle (user SDF) in a box, slip-like MirrorBC with a recording operator
struct Rec end
(::Rec)(eq, U, w) = State{1, Float64}((t = SVector(-w.normal[2], w.normal[1]); t' * w.hessian * t))
for T in (Float64,)
    rec = Rec()
    BCD = Dict{Int, AbstractBoundaryCondition}
    sh = Cuboid((T(-1), T(-1)), (T(1), T(1))) - Tagged(USquare(T(0.25)), 5)
    geom = GeometricDomain(sh; bc_map = BCD(t => MirrorBC(rec) for t in 1:5))
    pg = ParticleGrid(geom, (T(0.02), T(0.02)), ExponentialWeightFunction(one(T), T(0.05)), 1)
    buf = copy(pg.rhos)
    MirrorBC(rec)(pg, buf, 5, nothing, nothing, zero(T))
    idx = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 5]
    k = [buf[i][1] for i in idx]
    @printf("square obstacle (exact user SDF), %d ghosts: t·Ht min %.3e max %.3e; |κ| > 1: %d ghosts (largest at %s)\n",
            length(idx), minimum(k), maximum(k), count(>(1), abs.(k)), string(pg.core.positions[idx[argmax(abs.(k))]]))
end
