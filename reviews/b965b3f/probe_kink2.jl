# Distribution of the accepted FD Hessians next to polygon vertices (where the exact H is 0), as a function of the
# distance δ from the vertex in units of the FD step h
using HyperCloudCore, StaticArrays, LinearAlgebra, Printf
const HC = HyperCloudCore
include(joinpath(@__DIR__, "upoly.jl"))
for T in (Float64, Float32)
    P = UPoly(Tuple(SVector{2, T}(v) for v in ((-0.3, -0.2), (0.35, -0.15), (0.0, 0.4))))
    h = (eps(T))^(1 / 4)
    bins = Dict{Int, Vector{T}}()
    for i in 1:3, sgn in (1, -1), r in range(0.0, 6.0; length = 2401)
        a = P.v[i]; b = P.v[mod1(i + sgn, 3)]; e = (b - a) / norm(b - a); x = a + T(r * h) * e
        m = maximum(abs, HC._sdf_hessian(P, x))
        push!(get!(bins, floor(Int, 2r), T[]), m)
    end
    println(T, "  (h = ", h, ")")
    for k in sort(collect(keys(bins)))
        v = bins[k]
        @printf("  δ/h in [%.1f, %.1f): %4d pts, accepted |H| > 1e-3: %4d, > 1: %4d, max %.3e (max·h %.3f)\n", k / 2, (k + 1) / 2, length(v),
                count(>(1e-3), v), count(>(1), v), maximum(v), maximum(v) * h)
    end
end
