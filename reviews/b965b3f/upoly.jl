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
