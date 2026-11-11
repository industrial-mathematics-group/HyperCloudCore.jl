export GeometricDomain, get_rectangular_domain, get_spherical_domain, wall_geometry

include("Shapes.jl")

# =========================================================================
# 1. THE PURE GEOMETRY LAYER
# =========================================================================
"""
    GeometricDomain{GEO, D, T, F_Interior, F_Tag, S}
    GeometricDomain(shape::AbstractShape; bc_map, is_periodic = false, mins = nothing, maxs = nothing, label = Val(:shape))
    GeometricDomain(label, mins, maxs, is_periodic, is_interior, get_tag, bc_map)

The physical region of a problem and its boundary pieces: the bounding box, which points are inside, how
boundary points are tagged, which axes are periodic, and which boundary condition belongs to which tag.

The first constructor describes the region by a signed-distance [`AbstractShape`](@ref) (built-in boxes, balls
and half spaces, their unions, intersections and differences, translated and rotated copies). Points with
`signed_distance(shape, x) <= 0` are interior; the others get `boundary_tag(shape, x)`. Every particle of a grid
built on such a domain stores its outward wall normal and signed wall distance (`pg.core.normals`,
`pg.core.wall_distances`), which the higher-order conditions [`ExtrapolationBC`](@ref) and [`MirrorBC`](@ref)
use. The bounding box is that of the shape; along an axis where it is unbounded (open box axes, i.e. periodic
axes, or shapes cut from half spaces) `mins`/`maxs` must be given; on bounded axes they may enlarge the box (the
lattice is anchored at it), but a box that would crop the shape is an error (intersect the shape with a
[`Cuboid`](@ref) instead; the bounding boxes of rotated and combined shapes are conservative). A shape used on a
periodic axis must be open along it (its ends are no boundary; `Cuboid(...; open = ...)`). The volume of the
region (the shape within the box; [`domain_volume`](@ref)) is computed once here, analytically where the shape
allows it and by [`shape_volume`](@ref) otherwise; pass `volume` if it is known.

The second constructor takes the interior test `is_interior(x)` and tag function `get_tag(x)` directly (no
shape; `shape === nothing`). Such a domain has no wall normals (`NaN`), so conditions that need them refuse it
at grid construction.

# Fields
- `label::GEO`: The geometric identifier (e.g., `Val(:rectangular)`, `Val(:spherical)`, `Val(:shape)`), used
  to dispatch [`domain_volume`](@ref).
- `mins::Space{D, T}`: The minimal bounding box coordinates.
- `maxs::Space{D, T}`: The maximal bounding box coordinates.
- `is_periodic::SVector{D, Bool}`: Periodic boundary flags for each spatial dimension.
- `is_interior::F_Interior`: A function evaluating whether a given coordinate resides within the geometry.
- `get_tag::F_Tag`: A function generating integer boundary tags based on spatial location (`0` inside).
- `bc_map::Dict{Int, AbstractBoundaryCondition}`: A mapping of integer tags to specific boundary condition evaluators.
- `shape::S`: The [`AbstractShape`](@ref) of the region, or `nothing`.
- `volume::T`: The volume of the region within the box (`NaN` without a shape).
"""
struct GeometricDomain{GEO, D, T, F_Interior, F_Tag, S}
    label::GEO
    mins::Space{D, T}
    maxs::Space{D, T}
    is_periodic::SVector{D, Bool}
    is_interior::F_Interior
    get_tag::F_Tag
    bc_map::Dict{Int, AbstractBoundaryCondition}
    shape::S
    volume::T
end

GeometricDomain(label, mins::Space{D, T}, maxs::Space{D, T}, is_periodic::SVector{D, Bool}, is_interior, get_tag,
                bc_map::Dict{Int, AbstractBoundaryCondition}) where {D, T} =
    GeometricDomain(label, mins, maxs, is_periodic, is_interior, get_tag, bc_map, nothing, T(NaN))

_periodic_flags(is_periodic::Bool, ::Val{D}) where {D} = SVector{D, Bool}(ntuple(_ -> is_periodic, Val(D)))
_periodic_flags(is_periodic::NTuple{D, Bool}, ::Val{D}) where {D} = SVector{D, Bool}(is_periodic)

function GeometricDomain(shape::AbstractShape{D, T};
                         bc_map::Dict{Int, AbstractBoundaryCondition} = Dict{Int, AbstractBoundaryCondition}(),
                         is_periodic::Union{Bool, NTuple{D, Bool}} = false,
                         mins = nothing, maxs = nothing, label = Val(:shape), volume = nothing) where {D, T}
    is_per_svec = _periodic_flags(is_periodic, Val(D))
    bb_lo, bb_hi = bounding_box(shape)
    # mins/maxs close the unbounded axes; on bounded ones they may enlarge the box (the lattice is anchored at it),
    # but a box that would crop the shape is refused (the cut would carry no boundary condition)
    for d in 1:D, (lim, bb, below) in ((mins, bb_lo[d], true), (maxs, bb_hi[d], false))
        (lim === nothing || !isfinite(bb)) && continue
        (below ? T(lim[d]) > bb : T(lim[d]) < bb) &&
            throw(ArgumentError("GeometricDomain: mins/maxs ($(lim[d]) on axis $d) crop the shape (bounding box " *
                                "$(bb_lo[d]) .. $(bb_hi[d]); the boxes of rotated and combined shapes are conservative, " *
                                "give one containing it); intersect the shape with a Cuboid to cut it"))
    end
    lo = Space{D, T}(ntuple(d -> mins === nothing ? bb_lo[d] : T(mins[d]), Val(D)))
    hi = Space{D, T}(ntuple(d -> maxs === nothing ? bb_hi[d] : T(maxs[d]), Val(D)))
    (all(isfinite, lo) && all(isfinite, hi)) ||
        throw(ArgumentError("GeometricDomain: the shape is unbounded along some axis (bounding box $bb_lo .. $bb_hi); " *
                            "give mins/maxs, or intersect the shape with a Cuboid"))
    all(lo .<= hi) || throw(ArgumentError("GeometricDomain: mins must not exceed maxs"))
    is_interior_func = (pos) -> _inside(shape, pos)
    tag_func = (pos) -> _inside(shape, pos) ? 0 : _checked_tag(boundary_tag(shape, pos), pos)
    vol = volume === nothing ? _clipped_volume(shape, lo, hi) : T(volume)
    return GeometricDomain(label, lo, hi, is_per_svec, is_interior_func, tag_func, bc_map, shape, vol)
end

# Boundary tags are positive: `0` marks interior particles (the scheme would update such a ghost without a condition)
function _checked_tag(tag::Integer, pos)
    tag >= 1 || throw(ArgumentError("boundary_tag returned $tag at $pos: boundary tags must be ≥ 1 (0 marks interior points)"))
    return Int(tag)
end

"""
    get_rectangular_domain(::Type{T}, mins, maxs; bc_map, is_periodic = false)

Generates a rectangular `GeometricDomain` instance (the [`Cuboid`](@ref) `mins ≤ x ≤ maxs`, open along the periodic
axes).
- Tags every boundary point with the face it lies outside of: `2d - 1` for the lower and `2d` for the upper face
  of axis `d`, i.e. 1-D: 1 left, 2 right; 2-D: 1 left, 2 right, 3 bottom, 4 top; 3-D: 1/2 at `x = min/max`, 3/4 at
  `y`, 5/6 at `z`. Outside several faces (edges and corners of the ghost band) the one it is farthest outside of
  wins, the lower tag on ties. The faces of a periodic axis are ignored, so a 2-D channel periodic in x has only the
  tags 3 and 4. (Up to 0.7.9 the 2-D tag was that of the nearest face plane, also measured from inside, so ghosts
  outside one wall but near another plane got the other wall's tag.)
- Up to 0.7.9, 1-D and 3-D boxes tagged every boundary point 1 (both walls of an interval, all faces of a box);
  a map `Dict(1 => bc)` now covers only the lower wall (x-min face): give both, e.g. `Dict(1 => bc, 2 => bc)`.
- The `ParticleGrid` constructor warns when `bc_map` names a tag no boundary particle carries, or when boundary
  particles carry a tag `bc_map` does not cover (they then keep their state, like a Dirichlet condition).
"""
function get_rectangular_domain(
    ::Type{T}, 
    mins::NTuple{D, Real}, 
    maxs::NTuple{D, Real};
    bc_map::Dict{Int, AbstractBoundaryCondition} = Dict{Int, AbstractBoundaryCondition}(),
    is_periodic::Union{Bool, NTuple{D, Bool}} = false
) where {D, T}
    
    is_per = _periodic_flags(is_periodic, Val(D))
    # The box is open along the periodic axes: their ends are no boundary (the former tag function ignored them).
    # The tag rule of `Cuboid` is the former one: the face the point is farthest outside of, the lower tag on ties.
    box = Cuboid{D, T}(mins, maxs; open = Tuple(is_per))
    return GeometricDomain(box; bc_map = bc_map, is_periodic = Tuple(is_per), mins = mins, maxs = maxs,
                           label = Val(:rectangular))
end

"""
    get_spherical_domain(::Type{T}, center, radius; bc_map)

Generates a spherical (or circular) `GeometricDomain` instance (the [`Ball`](@ref) of `center` and `radius`).
- Assigns a uniform boundary tag of 1 to all exterior points.
"""
function get_spherical_domain(
    ::Type{T}, 
    center::NTuple{D, Real}, 
    radius::Real;
    bc_map::Dict{Int, AbstractBoundaryCondition} = Dict{Int, AbstractBoundaryCondition}(),
    is_periodic::Union{Bool, NTuple{D, Bool}} = false
) where {D, T}

    return GeometricDomain(Ball{D, T}(center, radius); bc_map = bc_map, is_periodic = is_periodic, label = Val(:spherical))
end

"""
    ComputationalDomain{D, T}
    ComputationalDomain(geom::GeometricDomain, nominal_dx, interp_range_factor)

A numerical wrapper surrounding the `GeometricDomain` that manages computational padding, canvas expansion, and periodic limits.

# Constructors

    ComputationalDomain(geom, nominal_dx, interp_range_factor)

Constructs the domain by extending the baseline mathematical boundaries out by a specified ghost layer width (dictated by `interp_range_factor`), unless periodicity is explicitly activated. It calculates and stores inverse lengths to assist with rapid periodic distance computations downstream.

# Fields
- `canvas_mins::Space{D, T}`: Expanded minimal coordinates including ghost layers.
- `canvas_maxs::Space{D, T}`: Expanded maximal coordinates including ghost layers.
- `is_periodic::SVector{D, Bool}`: Dimensional periodicity flags.
- `L::Space{D, T}`: The physical lengths of the domain.
- `L_inv::Space{D, T}`: Pre-computed inverse lengths.
- `L_wrap::Space{D, T}`: Pre-computed wrap lengths for periodic distance evaluation.
- `invL_wrap::Space{D, T}`: Pre-computed inverse wrap lengths.
"""
struct ComputationalDomain{D, T}
    canvas_mins::Space{D, T}
    canvas_maxs::Space{D, T}
    is_periodic::SVector{D, Bool}
    L::Space{D, T}
    L_inv::Space{D, T}
    L_wrap::Space{D, T}
    invL_wrap::Space{D, T}
end

function ComputationalDomain(
    geom::GeometricDomain{GEO, D, T},
    nominal_dx::NTuple{D, Real},
    interp_range_factor::Real
) where {D, T, GEO}
    
    is_per_svec = geom.is_periodic
    dxs_f = Space{D, T}(nominal_dx...)
    # Ghost layers are decided per axis: a periodic axis needs none, every other axis needs a band
    # of `interp_range_factor` spacings (mixed periodicity used to drop all ghost layers).
    # `interp_range_factor` is R / max(dx): an axis with a smaller spacing needs more layers to be R thick
    # (it used to get the same number, i.e. a band thinner than R on the finer axis of an anisotropic grid).
    # The product is formed in `T` and pulled down by a few ulps of `T`, so an exact ratio (e.g. 2.5 · 2 = 5)
    # does not get an extra layer from round-off.
    max_dx  = maximum(dxs_f)
    range_f = T(interp_range_factor)
    N_ghost = ntuple(Val(D)) do d
        dxs_f[d] == max_dx ? ceil(Int, interp_range_factor) :
                             ceil(Int, range_f * (max_dx / dxs_f[d]) * (1 - 4 * eps(T)))
    end

    canvas_mins = Space{D, T}(ntuple(d -> is_per_svec[d] ? geom.mins[d] : geom.mins[d] - N_ghost[d] * dxs_f[d], Val(D)))
    canvas_maxs = Space{D, T}(ntuple(d -> is_per_svec[d] ? geom.maxs[d] : geom.maxs[d] + N_ghost[d] * dxs_f[d], Val(D)))
    
    L_physical = geom.maxs - geom.mins
    L_wrap = Space{D, T}(ntuple(d -> is_per_svec[d] ? L_physical[d] : zero(T), Val(D)))
    invL_wrap = Space{D, T}(ntuple(d -> is_per_svec[d] ? one(T) / L_physical[d] : zero(T), Val(D)))
    L_inv = one(T) ./ max.(L_physical, T(1e-12))

    return ComputationalDomain{D, T}(
        canvas_mins, canvas_maxs, is_per_svec, L_physical, L_inv, L_wrap, invL_wrap
    )
end

# =========================================================================
# 3. UNIVERSAL NARROW-BAND POINT GENERATOR
# =========================================================================
# Lattice index `j` (point `canvas_min + j dx`) of the wall at `w`, or -1 if the wall falls between two lattice
# points (the extent is not a multiple of `dx`). The tolerance is the round-off of the ratio only: the rounding of
# `w` and of `canvas_min` (half an ulp each, in units of `dx`; `canvas_min` may lie in the next binade, e.g. below
# a wall at -1 in Float32), of their difference and of the division. An extent that is not a multiple of `dx`
# by more than that is not snapped, however close (as before 0.7.9). A quarter spacing caps the tolerance, so a
# wall is never assigned to a point that is not its own.
function _wall_index(w::T, canvas_min::T, dx::T) where {T}
    r   = (w - canvas_min) / dx
    k   = round(Int, r)
    tol = (2 * (eps(abs(w)) + eps(abs(canvas_min))) + eps(T) * abs(w - canvas_min)) / dx + 4 * eps(T) * abs(r)
    return abs(r - k) <= min(T(0.25), tol) ? k : -1
end

"""
    get_points(cd::ComputationalDomain, geom::GeometricDomain; nominal_dx, interp_range_factor, randomness, rng)

Generates a universal narrow-band particle grid utilizing the spatial canvas defined by the computational domain.

# Details
- Generates a baseline Cartesian point cloud separated by `nominal_dx`, applying optional random positional offsets.
- Sorts generated points into interior candidates and ghost candidates using the underlying geometry's `is_interior` functor.
- Filters exterior ghosts aggressively using a cutoff distance dictated by the `interp_range_factor`, yielding an optimized narrow-band boundary layer rather than filling the entire canvas.
- Outputs the filtered particle positions alongside their respective boundary booleans, tags, volumes, effective
  spacing, outward wall normals and signed wall distances (the last two `NaN` for a domain without a shape).
"""
function get_points(
    cd::ComputationalDomain{D, T},
    geom::GeometricDomain{GEO, D, T};
    nominal_dx::NTuple{D, Real},
    interp_range_factor::Real,
    randomness::Tuple = ntuple(i -> zero(T), D),
    rng = Random.default_rng(),
    kwargs...
) where {D, T, GEO}
    
    dxs_f = Space{D, T}(nominal_dx...)
    rand_f = Space{D, T}(randomness...)
    
    # Points per axis. On a periodic axis the end point coincides with the start point after
    # wrapping, so it is not generated (otherwise the seam particle would exist twice).
    Ns_total = ntuple(Val(D)) do d
        n = max(1, round(Int, (cd.canvas_maxs[d] - cd.canvas_mins[d]) / dxs_f[d]))
        cd.is_periodic[d] ? n : n + 1
    end
    
    inner_points = Space{D, T}[]
    ghost_candidates = Space{D, T}[]
    
    # The lattice points of a non-periodic wall are put exactly on the wall: `canvas_mins + (I - 1) dx` misses
    # `maxs` by a few ulps for many spacings (e.g. [-1, 1] with dx = 2/30), and the closed-interval classification
    # then made that wall point a ghost while the one on `mins` stayed interior. The wall point is chosen by its
    # index (`_wall_index`), so exactly one lattice point per wall moves and no neighbour of it can land on the
    # wall as well (a tolerance on the distance did that at very fine Float32 resolutions).
    wall_lo = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.mins[d], cd.canvas_mins[d], dxs_f[d]), Val(D))
    wall_hi = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.maxs[d], cd.canvas_mins[d], dxs_f[d]), Val(D))
    # A jittered point that left a periodic axis belongs to the other end: classifying it before wrapping made it
    # a (frozen) boundary particle at the seam.
    wrap(p) = Space{D, T}(ntuple(d -> cd.is_periodic[d] ? _wrap_periodic(p[d], geom.mins[d], cd.L_wrap[d]) : p[d], Val(D)))
    # Points less than h/2 outside a wall (shape-based domains) are snapped onto it, see `_snap_near_wall!`
    h_snap = minimum(dxs_f) / 2
    near_wall = Space{D, T}[]
    for I in CartesianIndices(Ns_total)
        lattice = Space{D, T}(ntuple(Val(D)) do d
            j = I[d] - 1
            j == wall_lo[d] ? geom.mins[d] : j == wall_hi[d] ? geom.maxs[d] : cd.canvas_mins[d] + j * dxs_f[d]
        end)
        jitter = Space{D, T}(ntuple(d -> rand_f[d] * (rand(rng, T) * 2 - 1), Val(D)))
        pos = wrap(lattice + jitter)
        if geom.is_interior(pos)
            push!(inner_points, pos)
        elseif !iszero(jitter) && geom.is_interior(lattice)
            # The jitter carried an interior lattice point (e.g. one on a wall) across the wall: it is reflected
            # back inside instead of becoming a ghost that sits almost on the wall. Such ghosts (about half of a
            # jittered wall row, down to 1e-3 dx outside) are slaved to the interior next to them every stage;
            # with them MirrorBC did not converge (channel along a wall, N = 20/40/80: max error 0.32/0.32/0.35)
            # and OutflowBC lost accuracy (0.23 at N = 80), without them both converge (0.28/0.062/0.023 and
            # 0.032, exact ghosts 0.026).
            push!(inner_points, _reflect_inside(geom, pos, lattice, wrap))
        elseif geom.shape !== nothing && signed_distance(geom.shape, pos) < h_snap
            # (the signed distance is a lower bound of the wall distance outside a combined shape)
            push!(near_wall, pos)
        else
            push!(ghost_candidates, pos)
        end
    end
    
    _snap_near_wall!(inner_points, ghost_candidates, near_wall, geom, h_snap, wrap)

    max_dx = maximum(dxs_f)
    cutoff_dist_sq = (T(interp_range_factor) * max_dx + max_dx)^2
    surviving_ghosts = _narrow_band(inner_points, ghost_candidates, cutoff_dist_sq)

    positions = vcat(inner_points, surviving_ghosts)
    N = length(positions)
    
    is_boundary = zeros(Bool, N)
    tags = zeros(Int, N)
    volumes = fill(prod(dxs_f), N)
    normals = Vector{Space{D, T}}(undef, N)
    wall_distances = Vector{T}(undef, N)

    for i in 1:N
        pos = positions[i]
        tag = geom.get_tag(pos)
        tags[i] = tag
        is_boundary[i] = (tag != 0)
        normals[i], wall_distances[i] = wall_geometry(geom, pos)
    end

    return positions, is_boundary, tags, volumes, Tuple(dxs_f), normals, wall_distances
end

# Points within `h_snap` = h/2 outside a wall (curved walls, obstacles and box extents off the lattice, jittered rows
# that the reflection below does not cover) would be ghosts sitting almost on the wall, with a gap of up to h between
# the wall and the first interior particle: the mirror points of the ghosts then fall into that gap, the fit of
# `MirrorBC` extrapolates, and the condition did not converge (a channel wall 0.1h off the lattice row, N = 20/40/80,
# t = 1: max error 0.41/0.67/0.63 against 0.09/0.04/0.016 with exact ghosts). Such a point is snapped onto the wall
# (its nearest wall point) and becomes interior, as the lattice row on a box face does: every ghost is then at least
# h/2 outside and the interior reaches the wall. On an unjittered lattice a snapped point stays at least h/2 from every
# other lattice point; a snapped point that would come within h/4 of an interior particle (jittered or reflected
# ones next to the wall, other snapped points behind corners) is pushed out to h/2 as a ghost instead, also if it
# does not test interior after rounding. Distances are minimum images across periodic seams.
function _snap_near_wall!(inner, ghosts, near, geom::GeometricDomain{GEO, D, T}, h_snap::T, wrap) where {GEO, D, T}
    isempty(near) && return nothing
    cell = h_snap / 2                                  # closer than h/4 is a clash
    # Cells of width >= `cell`; on a periodic axis a whole number of them spans the period and the index wraps
    L  = geom.maxs - geom.mins
    nc = ntuple(d -> geom.is_periodic[d] ? max(1, floor(Int, L[d] / cell)) : 0, Val(D))
    w  = ntuple(d -> geom.is_periodic[d] ? L[d] / nc[d] : cell, Val(D))
    key(p) = ntuple(d -> floor(Int, (p[d] - geom.mins[d]) / w[d]), Val(D))
    wrapkey(k) = ntuple(d -> geom.is_periodic[d] ? mod(k[d], nc[d]) : k[d], Val(D))
    mindist2(a, b) = sum(ntuple(Val(D)) do d
        δ = a[d] - b[d]
        geom.is_periodic[d] && (δ -= L[d] * round(δ / L[d]))
        δ^2
    end)
    taken = Dict{NTuple{D, Int}, Vector{Space{D, T}}}()
    for p in inner                                     # the interior particles next to the wall
        signed_distance(geom.shape, p) > -h_snap && push!(get!(taken, wrapkey(key(p)), Space{D, T}[]), p)
    end
    for pos in near
        n, φ = wall_geometry(geom, pos)
        if !(isfinite(φ) && φ < h_snap)
            push!(ghosts, pos)                         # farther than the lower bound suggested
            continue
        end
        s = wrap(pos - φ * n)
        geom.is_interior(s) || (s = wrap(pos - (φ + 64 * eps(T) * max(one(T), maximum(abs, pos))) * n))   # rounding
        k = key(s)
        clash = any(Iterators.product(ntuple(_ -> -1:1, Val(D))...)) do off
            any(q -> mindist2(q, s) < cell^2, get(taken, wrapkey(ntuple(d -> k[d] + off[d], Val(D))), ()))
        end
        if !clash && geom.is_interior(s)
            push!(inner, s)
            push!(get!(taken, wrapkey(k), Space{D, T}[]), s)
        else
            push!(ghosts, wrap(pos + (h_snap - φ) * n))
        end
    end
    return nothing
end

# The mirror image of `pos` behind the wall (a shape-based domain), or the lattice position it was jittered from
# if there is no shape or the mirror image is not interior (a corner)
function _reflect_inside(geom::GeometricDomain, pos, lattice, wrap)
    geom.shape === nothing && return lattice
    n, φ = wall_geometry(geom, pos)
    isfinite(φ) || return lattice
    p = wrap(pos - 2φ * n)
    return geom.is_interior(p) ? p : lattice
end

"""
    wall_geometry(geom::GeometricDomain, x) -> (normal, distance)

Outward unit wall normal and signed wall distance (negative inside the region) of the point `x`, from the
domain's shape; `NaN`s for a domain without a shape. Stored per particle in `pg.core.normals` and
`pg.core.wall_distances` when a grid is built or particles are inserted.

The signed distance of a combined shape (`max`/`min` of its pieces) is only a bound where several pieces are
active at once: behind a corner formed by two pieces (outside the floor *and* inside the step of a channel, or
outside both faces of a wedge) the nearest piece is not the nearest point of the region. The wall point is
therefore found in two stages: projecting onto the active piece and repeating from there until the projection
lies on the boundary (alternating projections, which reach *a* boundary point but slowly at acute corners and
not necessarily the nearest one at obtuse corners), then moving along the boundary towards `x` (the tangential
part of `x - p`, projected back onto the boundary, with step halving) while that gets closer, which ends at the
nearest point of a face or at the corner. The normal is the direction from that point to `x` and the distance its
length. For a point whose first projection already lies on the boundary (every point of a primitive shape, most
points of a combination) the shape's own normal and distance are returned unchanged. Inside a combination the
distance is the combined signed distance (a bound for unions; no condition reads interior wall data).
"""
wall_geometry(geom::GeometricDomain{GEO, D, T}, x::Space{D, T}) where {GEO, D, T} =
    geom.shape === nothing ? (Space{D, T}(ntuple(_ -> T(NaN), Val(D))), T(NaN)) : _wall_geometry(geom.shape, x)
function _wall_geometry(shape::AbstractShape{D, T}, x::Space{D, T}) where {D, T}
    φ0, n0, _ = _probe(shape, x)
    tol = 64 * eps(T) * max(one(T), maximum(abs, x))
    (abs(φ0) <= tol || !isfinite(φ0)) && return (n0, φ0)     # on the boundary, or no boundary at all (all axes open)
    p = x - φ0 * n0
    φp, np, _ = _probe(shape, p)
    abs(φp) <= tol && return (n0, φ0)            # the projection is on the boundary: the piece's values are exact
    p = _onto_boundary(shape, p, tol)
    # Descend along the boundary: at a point of a face the nearest point is where `x - p` is normal to it
    for _ in 1:64
        _, np, _ = _probe(shape, p)
        d  = x - p
        t  = d - sum(d .* np) * np               # tangential part of x - p
        dist2 = sum(abs2, d)
        sum(abs2, t) <= (tol * tol) * max(one(T), dist2) && break
        step, moved = one(T), false
        for _ in 1:30
            q = _onto_boundary(shape, p + step * t, tol)
            if sum(abs2, x - q) < dist2 * (1 - 16 * eps(T))
                p, moved = q, true
                break
            end
            step /= 2
        end
        moved || break                           # a corner (x - p in its normal cone): nothing closer nearby
    end
    d = x - p
    dist = sqrt(sum(abs2, d))
    dist <= tol && return (n0, φ0)
    sgn = φ0 < zero(T) ? -one(T) : one(T)
    return (sgn * d / dist, sgn * dist)
end

# A boundary point near `p` by alternating projections onto the active piece (converges like cos² of the corner
# angle: up to 512 steps for acute corners; the last iterate is used if it has not converged)
function _onto_boundary(shape::AbstractShape{D, T}, p::Space{D, T}, tol::T) where {D, T}
    for _ in 1:512
        φ, n, _ = _probe(shape, p)
        abs(φ) <= tol && return p
        p = p - φ * n
    end
    return p
end

# Boundary particles whose tag has no condition keep their state (the stage predictor copies `pg.rhos`), and a
# condition whose tag no particle carries does nothing: both are silent, so warn (host side, at grid set-up)
# when a non-empty `bc_map` and the tags of the boundary particles do not match. An empty map (no conditions,
# every boundary particle frozen) is taken as intended. Each mismatch is reported once per session.
const _WARNED_BC_TAGS = Set{Tuple{Vector{Int}, Vector{Int}}}()
const _WARNED_BC_TAGS_LOCK = ReentrantLock()          # grids may be built from several tasks
function _check_bc_tags(geom::GeometricDomain, tags::AbstractVector{Int}, is_boundary::AbstractVector{Bool})
    isempty(geom.bc_map) && return nothing
    carried = sort!(unique(tags[i] for i in eachindex(tags) if is_boundary[i]))
    mapped  = sort!(collect(keys(geom.bc_map)))
    unused, uncovered = setdiff(mapped, carried), setdiff(carried, mapped)
    (isempty(unused) && isempty(uncovered)) && return nothing
    key = (mapped, carried)
    first_time = lock(_WARNED_BC_TAGS_LOCK) do
        key in _WARNED_BC_TAGS ? false : (push!(_WARNED_BC_TAGS, key); true)
    end
    first_time || return nothing
    @warn "bc_map and boundary tags do not match: " *
          (isempty(unused) ? "" : "no boundary particle carries tag(s) $(unused), so their condition is never applied; ") *
          (isempty(uncovered) ? "" : "boundary particles with tag(s) $(uncovered) have no condition and keep their state; ") *
          "the boundary particles carry tag(s) $(carried) (boxes tag the face `2d - 1` / `2d` of axis `d`; up to 0.7.9 1-D and " *
          "3-D boxes tagged every boundary particle 1, see get_rectangular_domain)."
    return nothing
end

@inline _wrap_periodic(x::T, lo, L) where {T} = (y = T(lo) + mod(x - T(lo), T(L)); y >= T(lo) + T(L) ? T(lo) : y)

# The ghost candidates within `sqrt(cutoff_sq)` of an interior point, in their original order. Interior
# points are hashed into cells of that width, so only the 3^D surrounding cells are searched and the
# cost is linear in the number of points (the pairwise check was quadratic and dominated 3-D set-ups).
function _narrow_band(inner::Vector{Space{D, T}}, candidates::Vector{Space{D, T}}, cutoff_sq::T) where {D, T}
    survivors = Space{D, T}[]
    (isempty(inner) || isempty(candidates)) && return survivors
    h = sqrt(cutoff_sq)
    origin = reduce((a, b) -> min.(a, b), inner)
    cell_of(p) = ntuple(d -> floor(Int, (p[d] - origin[d]) / h), Val(D))
    cells = Dict{NTuple{D, Int}, Vector{Int}}()
    for (i, p) in enumerate(inner)
        push!(get!(() -> Int[], cells, cell_of(p)), i)
    end
    window = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    for g in candidates
        c = cell_of(g)
        found = false
        for off in window
            lst = get(cells, ntuple(d -> c[d] + off[d], Val(D)), nothing)
            lst === nothing && continue
            for i in lst
                if sum(abs2, g - inner[i]) <= cutoff_sq
                    found = true
                    break
                end
            end
            found && break
        end
        found && push!(survivors, g)
    end
    return survivors
end