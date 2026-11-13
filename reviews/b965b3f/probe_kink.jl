# The kink test of the FD _sdf_hessian: scans of wall points of user shapes (exact SDFs) near corners, on smooth
# strongly curved arcs, F64/F32, offset centres. Reports the largest accepted |t·Ht| (false positives at kinks) and
# the fraction of rejected points on smooth arcs (false negatives) with the error of the accepted ones.
using HyperCloudCore, StaticArrays, LinearAlgebra, Printf
const HC = HyperCloudCore
struct UPoly{T, N} <: AbstractShape{2, T}; v::NTuple{N, SVector{2, T}}; end    # convex polygon, CCW, exact SDF
function HC.signed_distance(s::UPoly{T, N}, x::SVector{2, T}) where {T, N}
    # exact: max over edges of the signed edge distances inside, distance to nearest edge segment outside
    dmin = T(Inf); inside = true; dmaxin = -T(Inf)
    for i in 1:N
        a = s.v[i]; b = s.v[mod1(i + 1, N)]; e = b - a; nrm = SVector(e[2], -e[1]) / norm(e)
        sd = dot(x - a, nrm); dmaxin = max(dmaxin, sd); sd > 0 && (inside = false)
        t = clamp(dot(x - a, e) / dot(e, e), zero(T), one(T)); dmin = min(dmin, norm(x - a - t * e))
    end
    return inside ? dmaxin : dmin
end
HC.bounding_box(s::UPoly) = ((-2.0, -2.0), (2.0, 2.0))
struct URound{T} <: AbstractShape{2, T}; c::SVector{2, T}; a::T; rc::T; end      # rounded square, corner radius rc
function HC.signed_distance(s::URound{T}, x::SVector{2, T}) where {T}
    q = abs.(x - s.c) .- (s.a - s.rc)
    return norm(max.(q, zero(T))) + min(max(q[1], q[2]), zero(T)) - s.rc
end
HC.bounding_box(s::URound) = ((-2.0, -2.0), (2.0, 2.0))
tang(n) = SVector(-n[2], n[1])
function wallnormal(s, x::SVector{2, T}) where {T}
    h = sqrt(eps(T)) * 10
    g = SVector((HC.signed_distance(s, x + SVector(h, 0)) - HC.signed_distance(s, x - SVector(h, 0))) / 2h,
                (HC.signed_distance(s, x + SVector(0, h)) - HC.signed_distance(s, x - SVector(0, h))) / 2h)
    return g / norm(g)
end
# 1. polygons: wall points along every edge, dense near the vertices (distances from the vertex 1e-6 .. 0.1)
for T in (Float64, Float32), (name, shp) in (("square", (T(-0.25), T(-0.25)), ), ("triangle", nothing), ("hexagon", nothing))
    vs = name == "square" ? ((-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)) :
         name == "triangle" ? ((-0.3, -0.2), (0.35, -0.15), (0.0, 0.4)) :
         Tuple((0.3cos(k * π / 3), 0.3sin(k * π / 3)) for k in 0:5)
    for off in (SVector{2, T}(0, 0), SVector{2, T}(10, -7))
        P = UPoly(Tuple(SVector{2, T}(v) + off for v in vs))
        N = length(vs); worst = zero(T); worstδ = zero(T); nacc = 0; ntot = 0
        for i in 1:N, δ in exp10.(range(-6, -0.7; length = 400)), sgn in (1, -1)
            a = P.v[i]; b = P.v[mod1(i + sgn, N)]; e = (b - a) / norm(b - a)
            δ < norm(b - a) / 2 || continue
            x = a + T(δ) * e
            Hm = HC._sdf_hessian(P, x); ntot += 1
            m = maximum(abs, Hm); m > 0 && (nacc += 1)
            if m > worst; worst = m; worstδ = T(δ); end
        end
        @printf("%-8s %-9s offset %-12s: %4d wall points, %4d nonzero H, largest accepted |H| %.3e at distance %.2e from a vertex (h = %.2e)\n",
                T, name, string(Tuple(off)), ntot, nacc, worst, worstδ, (eps(T) * max(1, maximum(abs, P.v[1])))^(1 / 4))
    end
end
# 1b. the same seen from the fluid INSIDE the polygon (complement: concave corners from the fluid)
for T in (Float64, Float32)
    P = UPoly(Tuple(SVector{2, T}(v) for v in ((-0.3, -0.2), (0.35, -0.15), (0.0, 0.4))))
    C = HC.ComplementShape(P)
    worst = zero(T); nacc = 0
    for i in 1:3, δ in exp10.(range(-6, -0.7; length = 400))
        a = P.v[i]; b = P.v[mod1(i + 1, 3)]; e = (b - a) / norm(b - a); x = a + T(δ) * e
        m = maximum(abs, HC._sdf_hessian(C, x)); m > 0 && (nacc += 1); worst = max(worst, m)
    end
    @printf("%-8s triangle complement: %d nonzero H of 1200, largest accepted |H| %.3e\n", T, nacc, worst)
end
# 2. rounded squares: arcs of radius rc, points on the arcs (angles 5..85 deg) and across the arc/flat junctions
for T in (Float64, Float32), off in (SVector{2, T}(0, 0), SVector{2, T}(10, -7)), rc in T.((0.1, 0.05, 0.02, 0.01, 0.005, 1e-3, 3e-4))
    S = URound(off, T(0.25), rc)
    cc = off + SVector{2, T}(0.25 - rc, 0.25 - rc)
    errs = T[]; rej = 0; tot = 0
    for θ in range(5, 85; length = 81)
        n = SVector{2, T}(cosd(θ), sind(θ)); x = cc + rc * n
        Hm = HC._sdf_hessian(S, x); tot += 1
        k = dot(tang(n), Hm * tang(n))
        iszero(Hm) ? (rej += 1) : push!(errs, abs(k * rc - 1))
    end
    # junction: points on the flat face near the arc start, distance s from the tangent point
    jmax = zero(T)
    for s in exp10.(range(-6, -1.5; length = 100))
        x = off + SVector{2, T}(0.25, 0.25 - rc - s); Hm = HC._sdf_hessian(S, x)
        jmax = max(jmax, maximum(abs, Hm) * rc)
    end
    @printf("%-8s rounded square offset %-11s rc = %-7.4g: %2d/%d arc points rejected, accepted rel. err max %.2e; flat side near junction max |H| rc %.2f\n",
            T, string(Tuple(off)), rc, rej, tot, isempty(errs) ? NaN : maximum(errs), jmax)
end
