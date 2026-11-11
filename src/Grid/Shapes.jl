export AbstractShape, Cuboid, Ball, HalfSpace, Tagged, Translated, Rotated, complement
export signed_distance, boundary_normal, boundary_tag, bounding_box, shape_volume

# =========================================================================
# SHAPES: signed-distance descriptions of the physical region
# =========================================================================
#
# A shape is a region of space described by its signed distance φ(x): negative inside, zero on the
# boundary, positive outside, |φ| the distance to the nearest boundary point. Everything the boundary
# treatment needs follows from φ: the unit outward normal n = ∇φ, the nearest boundary point x − φ n and
# the mirror image x − 2φ n. Shapes compose by `min`/`max` (union/intersection/difference), move by rigid
# transforms, and carry integer tags for their boundary pieces. The solver never sees anything but the
# functions below, so a region sampled from a CAD model (a level set on a grid) only needs a subtype that
# implements them.

"""
    AbstractShape{D, T}

A region of `D`-dimensional space with element type `T`, described by its signed distance (negative
inside, positive outside). The physical region of a [`GeometricDomain`](@ref) is a shape; ghost particles
lie outside it.

# API
- `signed_distance(shape, x) -> T` (required).
- `bounding_box(shape) -> (mins, maxs)` (required for the region of a domain; may be infinite for pieces
  that are intersected with a bounded one).
- `boundary_normal(shape, x) -> Space{D, T}`: unit gradient of the signed distance, i.e. the outward normal
  of the nearest boundary point. Defaults to central differences of `signed_distance`.
- `boundary_tag(shape, x) -> Int`: the tag of the boundary piece nearest to `x` (a positive integer, used
  for points outside the shape). Defaults to `1`.
- `shape_volume(shape) -> T`: the volume of the region. Defaults to a lattice integration over the bounding
  box.

Built-in shapes: [`Cuboid`](@ref), [`Ball`](@ref), [`HalfSpace`](@ref); combinations with `union`,
`intersect`, `setdiff` (or `-`) and [`complement`](@ref); [`Translated`](@ref) and [`Rotated`](@ref) copies;
[`Tagged`](@ref) renumbers the boundary tags of a piece.
"""
abstract type AbstractShape{D, T} end

Base.ndims(::AbstractShape{D}) where {D} = D
Base.eltype(::AbstractShape{D, T}) where {D, T} = T

"""
    signed_distance(shape, x) -> T

Signed distance of `x` to the boundary of `shape`: negative inside, zero on the boundary, positive outside.
"""
function signed_distance end

"""
    boundary_normal(shape, x) -> Space{D, T}

Unit outward normal of the boundary piece nearest to `x` (the gradient of the signed distance). For the
built-in shapes it is analytic; the fallback takes central differences of [`signed_distance`](@ref) with a
step of `∛eps(T)` relative to the size of `x`, which is accurate to a few `∛eps(T)²` away from edges.
"""
function boundary_normal(s::AbstractShape{D, T}, x::Space{D, T}) where {D, T}
    h = cbrt(eps(T)) * max(one(T), maximum(abs, x))
    g = Space{D, T}(ntuple(Val(D)) do d
        e = Space{D, T}(ntuple(k -> k == d ? h : zero(T), Val(D)))
        (signed_distance(s, x + e) - signed_distance(s, x - e)) / (2h)
    end)
    return _unit(g)
end

"""
    boundary_tag(shape, x) -> Int

Tag (a positive integer) of the boundary piece of `shape` nearest to `x`. Boxes tag their faces
(`2d - 1` lower, `2d` upper face of axis `d`), every other built-in piece has tag `1`; [`Tagged`](@ref)
renumbers. For points outside a union/intersection the nearest piece is the one whose signed distance is
active (the smallest one in a union, the largest one in an intersection: the piece the point is farthest
outside of).
"""
boundary_tag(::AbstractShape, x) = 1

"""
    bounding_box(shape) -> (mins::Space{D, T}, maxs::Space{D, T})

A box containing the shape (infinite entries for unbounded pieces).
"""
function bounding_box end

"""
    shape_volume(shape; samples = 2_000_000) -> T

Volume of the region. Analytic for [`Cuboid`](@ref), [`Ball`](@ref) and their translated, rotated and tagged
copies. Every other shape is integrated on a lattice of about `samples` cubic cells over its bounding box, each
cell counted with the volume fraction `clamp(1/2 - φ/h, 0, 1)` of its centre's signed distance (exact for a
wall parallel to a cell face, second order otherwise: relative errors below 1e-5 in 2-D and 1e-3 in 3-D at
the default for piecewise smooth boundaries). Define a method for shapes whose volume is known.
"""
function shape_volume(s::AbstractShape{D, T}; samples::Int = 2_000_000) where {D, T}
    mins, maxs = bounding_box(s)
    all(isfinite, mins) && all(isfinite, maxs) || throw(ArgumentError("shape_volume: the shape is unbounded"))
    L  = maxs .- mins
    h0 = (prod(L) / samples)^(1 / D)                      # cubic cells: the fraction below assumes one width
    ns = ntuple(d -> max(2, ceil(Int, L[d] / h0)), Val(D))
    h  = Space{D, T}(ntuple(d -> L[d] / ns[d], Val(D)))
    hm = maximum(h)
    vol_cell = prod(h)
    acc = zero(T)
    for I in CartesianIndices(ns)
        x = Space{D, T}(ntuple(d -> mins[d] + (I[d] - T(0.5)) * h[d], Val(D)))
        acc += clamp(T(0.5) - signed_distance(s, x) / hm, zero(T), one(T))
    end
    return acc * vol_cell
end

# Whether `x` belongs to the region (domains use it for the interior test and the tags). Defaults to
# `signed_distance <= 0`; a ball compares squared distances (see `_inside(::Ball, x)`).
_inside(s::AbstractShape{D, T}, x::Space{D, T}) where {D, T} = signed_distance(s, x) <= zero(T)

# Volume of the shape within the box `lo..hi` (the domain box): the shape's own volume when the box is its
# bounding box (analytic where the shape defines it), the analytic box overlap for a box, else a lattice
# integration of the shape cut by the box. Called once, by the `GeometricDomain` constructor.
function _clipped_volume(s::AbstractShape{D, T}, lo::Space{D, T}, hi::Space{D, T}) where {D, T}
    bb_lo, bb_hi = bounding_box(s)
    # the box contains the (bounded) shape: its own volume, analytic where the shape defines it
    (all(isfinite, bb_lo) && all(isfinite, bb_hi) && all(lo .<= bb_lo) && all(hi .>= bb_hi)) && return T(shape_volume(s))
    return T(shape_volume(intersect(s, Cuboid{D, T}(Tuple(lo), Tuple(hi)))))
end

# (φ, n, tag) of the active boundary piece at `x`. Leaves evaluate their three functions; composites pick
# the active child's triple, so a tag or normal is never taken from a piece other than the one whose
# distance is in force.
@inline _probe(s::AbstractShape, x) = (signed_distance(s, x), boundary_normal(s, x), boundary_tag(s, x))

@inline function _unit(g::Space{D, T}) where {D, T}
    n2 = sum(abs2, g)
    n2 > zero(T) || return Space{D, T}(ntuple(k -> k == 1 ? one(T) : zero(T), Val(D)))
    return g / sqrt(n2)
end

_space(::Type{T}, ::Val{D}, x) where {T, D} = Space{D, T}(ntuple(d -> T(x[d]), Val(D)))

# -------------------------------------------------------------------------
# Primitives
# -------------------------------------------------------------------------
"""
    Cuboid(mins, maxs; open = false)

The axis-aligned box `mins ≤ x ≤ maxs`. Its signed distance is exact (Euclidean outside the edges and
corners). The faces are tagged `2d - 1` (lower) and `2d` (upper) for axis `d`; outside several faces the one
the point is farthest outside of decides, the lower tag on ties.

`open` (one `Bool` or one per axis) drops the faces of an axis: the box is unbounded along it, which is how
a domain represents a periodic axis (its ends are no boundary).
"""
struct Cuboid{D, T} <: AbstractShape{D, T}
    mins::Space{D, T}
    maxs::Space{D, T}
    open::SVector{D, Bool}
    function Cuboid{D, T}(mins, maxs; open::Union{Bool, NTuple{D, Bool}, SVector{D, Bool}} = false) where {D, T}
        op = open isa Bool ? SVector{D, Bool}(ntuple(_ -> open, Val(D))) : SVector{D, Bool}(open)
        m, M = _space(T, Val(D), mins), _space(T, Val(D), maxs)
        all(m .<= M) || throw(ArgumentError("Cuboid: mins must not exceed maxs"))
        return new{D, T}(m, M, op)
    end
end
function Cuboid(mins::NTuple{D, Real}, maxs::NTuple{D, Real}; open::Union{Bool, NTuple{D, Bool}} = false) where {D}
    T = float(promote_type(typeof.(mins)..., typeof.(maxs)...))
    return Cuboid{D, T}(mins, maxs; open = open)
end
Cuboid(mins::Space{D, T}, maxs::Space{D, T}; open = false) where {D, T} = Cuboid{D, T}(Tuple(mins), Tuple(maxs); open = open)

# Per-axis outside distance (negative inside): max(mins - x, x - maxs); -Inf on open axes.
@inline _box_q(b::Cuboid{D, T}, x::Space{D, T}) where {D, T} =
    Space{D, T}(ntuple(d -> b.open[d] ? T(-Inf) : max(b.mins[d] - x[d], x[d] - b.maxs[d]), Val(D)))

function signed_distance(b::Cuboid{D, T}, x::Space{D, T}) where {D, T}
    q = _box_q(b, x)
    outside = sqrt(sum(abs2, max.(q, zero(T))))
    return outside + min(maximum(q), zero(T))
end

# The face with the largest outside distance (the nearest face from inside); lower faces first, ties to the
# first one found. Returns (tag, axis, sign).
@inline function _box_face(b::Cuboid{D, T}, x::Space{D, T}) where {D, T}
    best_tag, best_axis, best_sign, best = 0, 1, one(T), T(-Inf)
    for d in 1:D
        b.open[d] && continue
        o_lo = b.mins[d] - x[d]
        o_hi = x[d] - b.maxs[d]
        o_lo > best && (best_tag = 2d - 1; best_axis = d; best_sign = -one(T); best = o_lo)
        o_hi > best && (best_tag = 2d;     best_axis = d; best_sign =  one(T); best = o_hi)
    end
    return best_tag, best_axis, best_sign
end

boundary_tag(b::Cuboid{D, T}, x::Space{D, T}) where {D, T} = max(_box_face(b, x)[1], 1)

function boundary_normal(b::Cuboid{D, T}, x::Space{D, T}) where {D, T}
    q = _box_q(b, x)
    if maximum(q) > zero(T)             # outside: towards the nearest point of the box
        g = Space{D, T}(ntuple(Val(D)) do d
            q[d] > zero(T) ? (x[d] > b.maxs[d] ? q[d] : -q[d]) : zero(T)
        end)
        return _unit(g)
    end
    _, axis, sgn = _box_face(b, x)      # inside (or on the boundary): the nearest face
    return Space{D, T}(ntuple(d -> d == axis ? sgn : zero(T), Val(D)))
end

bounding_box(b::Cuboid{D, T}) where {D, T} = (Space{D, T}(ntuple(d -> b.open[d] ? T(-Inf) : b.mins[d], Val(D))),
                                           Space{D, T}(ntuple(d -> b.open[d] ? T(Inf) : b.maxs[d], Val(D))))
shape_volume(b::Cuboid{D, T}; kwargs...) where {D, T} = any(b.open) ? T(Inf) : prod(b.maxs .- b.mins)

"""
    Ball(center, radius)

The ball `|x - center| ≤ radius` (an interval in 1-D, a disc in 2-D). Boundary tag `1`.
"""
struct Ball{D, T} <: AbstractShape{D, T}
    center::Space{D, T}
    radius::T
    function Ball{D, T}(center, radius) where {D, T}
        radius > 0 || throw(ArgumentError("Ball: the radius must be positive"))
        return new{D, T}(_space(T, Val(D), center), T(radius))
    end
end
function Ball(center::NTuple{D, Real}, radius::Real) where {D}
    T = float(promote_type(typeof.(center)..., typeof(radius)))
    return Ball{D, T}(center, radius)
end
Ball(center::Space{D, T}, radius::Real) where {D, T} = Ball{D, T}(Tuple(center), radius)

signed_distance(b::Ball{D, T}, x::Space{D, T}) where {D, T} = sqrt(sum(abs2, x - b.center)) - b.radius
# Inside test on the squared distance: the `sqrt` of the signed distance rounds points one ulp outside onto the
# sphere (exactly as the spherical domain decided before shapes)
_inside(b::Ball{D, T}, x::Space{D, T}) where {D, T} = sum(abs2, x - b.center) <= b.radius^2
boundary_normal(b::Ball{D, T}, x::Space{D, T}) where {D, T} = _unit(x - b.center)
bounding_box(b::Ball) = (b.center .- b.radius, b.center .+ b.radius)
shape_volume(b::Ball{D, T}; kwargs...) where {D, T} =
    D == 1 ? 2b.radius : D == 2 ? T(π) * b.radius^2 : T(4π / 3) * b.radius^3

"""
    HalfSpace(point, normal)

The half space `(x - point) ⋅ normal ≤ 0` (`normal` is normalised; it points out of the region). Unbounded:
intersect it with a bounded shape to make a domain. Boundary tag `1`.
"""
struct HalfSpace{D, T} <: AbstractShape{D, T}
    point::Space{D, T}
    normal::Space{D, T}
    function HalfSpace{D, T}(point, normal) where {D, T}
        n = _space(T, Val(D), normal)
        sum(abs2, n) > zero(T) || throw(ArgumentError("HalfSpace: the normal must not vanish"))
        return new{D, T}(_space(T, Val(D), point), _unit(n))
    end
end
function HalfSpace(point::NTuple{D, Real}, normal::NTuple{D, Real}) where {D}
    T = float(promote_type(typeof.(point)..., typeof.(normal)...))
    return HalfSpace{D, T}(point, normal)
end
HalfSpace(point::Space{D, T}, normal::Space{D, T}) where {D, T} = HalfSpace{D, T}(Tuple(point), Tuple(normal))

signed_distance(h::HalfSpace{D, T}, x::Space{D, T}) where {D, T} = sum((x - h.point) .* h.normal)
boundary_normal(h::HalfSpace{D, T}, x::Space{D, T}) where {D, T} = h.normal
bounding_box(h::HalfSpace{D, T}) where {D, T} = (Space{D, T}(ntuple(_ -> T(-Inf), Val(D))), Space{D, T}(ntuple(_ -> T(Inf), Val(D))))

# -------------------------------------------------------------------------
# Combinations
# -------------------------------------------------------------------------
struct UnionShape{D, T, S <: Tuple} <: AbstractShape{D, T}
    shapes::S
end
struct IntersectionShape{D, T, S <: Tuple} <: AbstractShape{D, T}
    shapes::S
end
struct ComplementShape{D, T, S <: AbstractShape{D, T}} <: AbstractShape{D, T}
    shape::S
end

function _check_same(shapes::Tuple)
    D, T = ndims(shapes[1]), eltype(shapes[1])
    all(s -> ndims(s) == D && eltype(s) == T, shapes) ||
        throw(ArgumentError("shapes of different dimension or element type cannot be combined"))
    return D, T
end
"""
    union(a::AbstractShape, b::AbstractShape...)

The union of shapes: signed distance `min` over the pieces. Outside the union the nearest piece provides
the normal and the tag.
"""
function Base.union(a::AbstractShape, b::AbstractShape, rest::AbstractShape...)
    shapes = (a, b, rest...)
    D, T = _check_same(shapes)
    return UnionShape{D, T, typeof(shapes)}(shapes)
end
"""
    intersect(a::AbstractShape, b::AbstractShape...)

The intersection of shapes: signed distance `max` over the pieces. Outside the intersection the piece the
point is farthest outside of provides the normal and the tag.
"""
function Base.intersect(a::AbstractShape, b::AbstractShape, rest::AbstractShape...)
    shapes = (a, b, rest...)
    D, T = _check_same(shapes)
    return IntersectionShape{D, T, typeof(shapes)}(shapes)
end
"""
    complement(a::AbstractShape)

Everything outside `a` (the signed distance changes sign, the normals flip; tags are kept). Unbounded.
`setdiff(a, b...)` and `a - b` are `intersect(a, complement(b)...)`: `a` with the other shapes cut out.
"""
complement(a::AbstractShape{D, T}) where {D, T} = ComplementShape{D, T, typeof(a)}(a)
complement(a::ComplementShape) = a.shape
Base.setdiff(a::AbstractShape, b::AbstractShape, rest::AbstractShape...) = intersect(a, complement(b), map(complement, rest)...)
Base.:-(a::AbstractShape, b::AbstractShape) = setdiff(a, b)

# Recursive, allocation-free selection over heterogeneous tuples: the child with the smallest (union) or
# largest (intersection) signed distance, the first one on ties.
@inline _probe_select(shapes::Tuple{Any}, x, better) = _probe(shapes[1], x)
@inline function _probe_select(shapes::Tuple, x, better)
    a = _probe(shapes[1], x)
    b = _probe_select(Base.tail(shapes), x, better)
    return better(b[1], a[1]) ? b : a
end
@inline _dist_select(shapes::Tuple{Any}, x, pick) = signed_distance(shapes[1], x)
@inline _dist_select(shapes::Tuple, x, pick) = pick(signed_distance(shapes[1], x), _dist_select(Base.tail(shapes), x, pick))

@inline _probe(u::UnionShape, x) = _probe_select(u.shapes, x, <)
@inline _probe(u::IntersectionShape, x) = _probe_select(u.shapes, x, >)
@inline function _probe(c::ComplementShape, x)
    φ, n, tag = _probe(c.shape, x)
    return (-φ, -n, tag)
end
signed_distance(u::UnionShape{D, T}, x::Space{D, T}) where {D, T} = _dist_select(u.shapes, x, min)
signed_distance(u::IntersectionShape{D, T}, x::Space{D, T}) where {D, T} = _dist_select(u.shapes, x, max)
signed_distance(c::ComplementShape{D, T}, x::Space{D, T}) where {D, T} = -signed_distance(c.shape, x)
for S in (:UnionShape, :IntersectionShape, :ComplementShape)
    @eval boundary_normal(s::$S{D, T}, x::Space{D, T}) where {D, T} = _probe(s, x)[2]
    @eval boundary_tag(s::$S{D, T}, x::Space{D, T}) where {D, T} = _probe(s, x)[3]
end

function bounding_box(u::UnionShape)
    boxes = map(bounding_box, u.shapes)
    return (reduce((a, b) -> min.(a, b), first.(boxes)), reduce((a, b) -> max.(a, b), last.(boxes)))
end
function bounding_box(u::IntersectionShape)
    boxes = map(bounding_box, u.shapes)
    return (reduce((a, b) -> max.(a, b), first.(boxes)), reduce((a, b) -> min.(a, b), last.(boxes)))
end
bounding_box(c::ComplementShape{D, T}) where {D, T} = (Space{D, T}(ntuple(_ -> T(-Inf), Val(D))), Space{D, T}(ntuple(_ -> T(Inf), Val(D))))

# -------------------------------------------------------------------------
# Decorators: tags and rigid transforms
# -------------------------------------------------------------------------
"""
    Tagged(shape, tag::Int)
    Tagged(shape, f)

`shape` with its boundary tags replaced by `tag`, or renumbered by `f(tag)` (e.g. `t -> t + 4` gives the
faces of an obstacle box the tags 5–8 after the four walls of a 2-D canvas).
"""
struct Tagged{D, T, S <: AbstractShape{D, T}, F} <: AbstractShape{D, T}
    shape::S
    tag::F
    function Tagged{D, T, S, F}(shape::S, tag::F) where {D, T, S <: AbstractShape{D, T}, F}
        tag isa Integer && tag < 1 && throw(ArgumentError("Tagged: boundary tags must be ≥ 1 (0 marks interior points), got $tag"))
        return new{D, T, S, F}(shape, tag)
    end
end
Tagged(shape::AbstractShape{D, T}, tag::Integer) where {D, T} = Tagged{D, T, typeof(shape), Int}(shape, Int(tag))
Tagged(shape::AbstractShape{D, T}, f) where {D, T} = Tagged{D, T, typeof(shape), typeof(f)}(shape, f)
_inside(t::Tagged{D, T}, x::Space{D, T}) where {D, T} = _inside(t.shape, x)
@inline _retag(tag::Int, ::Int) = tag
@inline _retag(f, tag::Int) = Int(f(tag))
signed_distance(t::Tagged{D, T}, x::Space{D, T}) where {D, T} = signed_distance(t.shape, x)
boundary_normal(t::Tagged{D, T}, x::Space{D, T}) where {D, T} = boundary_normal(t.shape, x)
boundary_tag(t::Tagged{D, T}, x::Space{D, T}) where {D, T} = _retag(t.tag, boundary_tag(t.shape, x))
@inline function _probe(t::Tagged, x)
    φ, n, tag = _probe(t.shape, x)
    return (φ, n, _retag(t.tag, tag))
end
bounding_box(t::Tagged) = bounding_box(t.shape)
shape_volume(t::Tagged; kwargs...) = shape_volume(t.shape; kwargs...)

"""
    Translated(shape, offset)

`shape` moved by `offset`.
"""
struct Translated{D, T, S <: AbstractShape{D, T}} <: AbstractShape{D, T}
    shape::S
    offset::Space{D, T}
end
Translated(s::AbstractShape{D, T}, offset) where {D, T} = Translated{D, T, typeof(s)}(s, _space(T, Val(D), offset))
signed_distance(t::Translated{D, T}, x::Space{D, T}) where {D, T} = signed_distance(t.shape, x - t.offset)
boundary_normal(t::Translated{D, T}, x::Space{D, T}) where {D, T} = boundary_normal(t.shape, x - t.offset)
boundary_tag(t::Translated{D, T}, x::Space{D, T}) where {D, T} = boundary_tag(t.shape, x - t.offset)
@inline _probe(t::Translated, x) = _probe(t.shape, x - t.offset)
_inside(t::Translated{D, T}, x::Space{D, T}) where {D, T} = _inside(t.shape, x - t.offset)
bounding_box(t::Translated) = (b = bounding_box(t.shape); (b[1] + t.offset, b[2] + t.offset))
shape_volume(t::Translated; kwargs...) = shape_volume(t.shape; kwargs...)

"""
    Rotated(shape, R::SMatrix; center = 0)
    Rotated(shape, angle; center = 0)                 # 2-D
    Rotated(shape, axis, angle; center = 0)           # 3-D

`shape` rotated by the orthogonal matrix `R` about `center` (a 2-D angle in radians, counter-clockwise, or a
3-D axis and angle build `R`).
"""
struct Rotated{D, T, S <: AbstractShape{D, T}, L} <: AbstractShape{D, T}
    shape::S
    R::SMatrix{D, D, T, L}
    center::Space{D, T}
end
function Rotated(s::AbstractShape{D, T}, R::SMatrix{D, D}; center = ntuple(_ -> 0, Val(D))) where {D, T}
    Rt = SMatrix{D, D, T, D * D}(R)
    maximum(abs, Rt' * Rt - I) <= 100 * eps(T) || throw(ArgumentError("Rotated: R is not orthogonal"))
    return Rotated{D, T, typeof(s), D * D}(s, Rt, _space(T, Val(D), center))
end
function Rotated(s::AbstractShape{2, T}, angle::Real; kwargs...) where {T}
    c, sn = T(cos(angle)), T(sin(angle))
    return Rotated(s, SMatrix{2, 2, T, 4}(c, sn, -sn, c); kwargs...)
end
function Rotated(s::AbstractShape{3, T}, axis, angle::Real; kwargs...) where {T}
    k = _unit(_space(T, Val(3), axis))
    c, sn = T(cos(angle)), T(sin(angle))
    K = SMatrix{3, 3, T, 9}(0, k[3], -k[2], -k[3], 0, k[1], k[2], -k[1], 0)
    return Rotated(s, SMatrix{3, 3, T, 9}(I) + sn * K + (1 - c) * (K * K); kwargs...)
end
@inline _local(r::Rotated, x) = r.R' * (x - r.center) + r.center
_inside(r::Rotated{D, T}, x::Space{D, T}) where {D, T} = _inside(r.shape, _local(r, x))
signed_distance(r::Rotated{D, T}, x::Space{D, T}) where {D, T} = signed_distance(r.shape, _local(r, x))
boundary_normal(r::Rotated{D, T}, x::Space{D, T}) where {D, T} = r.R * boundary_normal(r.shape, _local(r, x))
boundary_tag(r::Rotated{D, T}, x::Space{D, T}) where {D, T} = boundary_tag(r.shape, _local(r, x))
@inline function _probe(r::Rotated, x)
    φ, n, tag = _probe(r.shape, _local(r, x))
    return (φ, r.R * n, tag)
end
function bounding_box(r::Rotated{D, T}) where {D, T}
    mins, maxs = bounding_box(r.shape)
    (all(isfinite, mins) && all(isfinite, maxs)) || return (Space{D, T}(ntuple(_ -> T(-Inf), Val(D))), Space{D, T}(ntuple(_ -> T(Inf), Val(D))))
    lo, hi = Space{D, T}(ntuple(_ -> T(Inf), Val(D))), Space{D, T}(ntuple(_ -> T(-Inf), Val(D)))
    for I in CartesianIndices(ntuple(_ -> 0:1, Val(D)))
        corner = Space{D, T}(ntuple(d -> I[d] == 0 ? mins[d] : maxs[d], Val(D)))
        y = r.R * (corner - r.center) + r.center
        lo, hi = min.(lo, y), max.(hi, y)
    end
    return (lo, hi)
end
shape_volume(r::Rotated; kwargs...) = shape_volume(r.shape; kwargs...)

# The box overlap of a cuboid (open axes span the domain box), see `_clipped_volume`
function _clipped_volume(c::Cuboid{D, T}, lo::Space{D, T}, hi::Space{D, T}) where {D, T}
    return prod(ntuple(d -> c.open[d] ? hi[d] - lo[d] : max(zero(T), min(c.maxs[d], hi[d]) - max(c.mins[d], lo[d])), Val(D)))
end
