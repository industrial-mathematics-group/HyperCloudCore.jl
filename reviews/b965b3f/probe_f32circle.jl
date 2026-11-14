# F5: Float32 FD curvature of a user circle r = 0.15 about (10, 10) and (0, 0): distribution over 360 wall points
using HyperCloudCore, StaticArrays, LinearAlgebra, Printf, Statistics
const HC = HyperCloudCore
struct UCircle{T} <: AbstractShape{2, T}; c::SVector{2, T}; r::T; end
HC.signed_distance(s::UCircle{T}, x::SVector{2, T}) where {T} = norm(x - s.c) - s.r
for T in (Float32, Float64), (c, r) in ((SVector{2, T}(10, 10), T(0.15)), (SVector{2, T}(0, 0), T(0.15)), (SVector{2, T}(0, 0), T(0.05)), (SVector{2, T}(10, 10), T(0.05)))
    e = map(0:359) do θd
        n = SVector{2, T}(cosd(θd), sind(θd)); x = c + r * n; t = SVector(-n[2], n[1])
        abs(dot(t, HC._sdf_hessian(UCircle(c, r), x) * t) * r - 1)
    end
    @printf("%-8s c = %-14s r = %.2f: rel. error of t·Ht: median %.2e, max %.2e, rejected (H = 0) %d/360\n", T, string(Tuple(c)), r, median(e), maximum(e), count(==(1), e))
end
