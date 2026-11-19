export FixedDirichlet, OutflowBC

"""
    FixedDirichlet <: AbstractBoundaryCondition
    (::FixedDirichlet)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t, reference = pg.rhos)

A strict, unchanging boundary condition strategy.

# Callable / Functor

Enforces a static Dirichlet condition upon boundary particles. Iterates over the grid and resets the state of any boundary particle matching the specified `tag` to the state stored in `reference` (`pg.rhos`, or the state array the time stepper advances) at the start of the step. Nothing else writes the state of a boundary
particle (the stage predictor keeps it), so that is its initial state unless the user changes `pg.rhos`.
"""
struct FixedDirichlet <: AbstractBoundaryCondition end

"""
    OutflowBC <: AbstractBoundaryCondition
    (::OutflowBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t, reference = pg.rhos)

A zero-gradient, transmissive boundary condition strategy.

# Callable / Functor

Enforces a zero-gradient outflow utilizing a multi-pass nearest-donor algorithm. 
- Initiates all active interior particles as valid state donors.
- Executes up to 5 symmetrical passes outward, dynamically locating the nearest resolved neighbor using squared distances and copying its state to the target particle.
- Safely falls back to the state in `reference` (`pg.rhos`, or the state array the time stepper advances) at the start of the step for any completely orphaned boundary particles that fail to resolve a donor.
"""
struct OutflowBC <: AbstractBoundaryCondition end

# =========================================================================
# BOUNDARY CONDITION DISPATCHER
# =========================================================================

"""
    apply_boundary_conditions!(pg::ParticleGrid, rhos_buffer, ts, eq, t, reference = pg.rhos)

Iterates over the active boundary condition map stored in the grid's geometry configuration and dispatches the corresponding strategy for each registered domain tag.
The map itself (a `Dict`) stays on the host; the conditions shipped with the package run as kernel
launches, so they work on a grid built with `device_grid` too. `reference` is the state array of the step start
(the time stepper passes its copy), which the conditions that keep or fall back to the initial state of a boundary
particle read ([`FixedDirichlet`](@ref), [`OutflowBC`](@ref)). A user-defined condition is any
`AbstractBoundaryCondition` callable as `(bc)(pg, rhos_buffer, tag, ts, eq, t, reference)`; one with the older
signature `(bc)(pg, rhos_buffer, tag, ts, eq, t)` is called with that.
"""
function apply_boundary_conditions!(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, ts::TimeStepper, eq::HyperbolicPDE, t::Real,
                                    reference::AbstractVector{State{M, T}} = pg.rhos) where {D, M, T}
    for (tag, bc_functor) in pg.geometry.bc_map
        bc_functor(pg, rhos_buffer, tag, ts, eq, t, reference)
    end
    return nothing
end

# A condition that does not read the reference state keeps the six-argument form
(bc::AbstractBoundaryCondition)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t, reference) = bc(pg, rhos_buffer, tag, ts, eq, t)

# =========================================================================
# CONCRETE BOUNDARY CONDITIONS
# =========================================================================

# 1. Fixed Dirichlet
"""
    (::FixedDirichlet)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

Enforces a static Dirichlet condition upon boundary particles. 
- Iterates over the grid and resets the state of any particle matching the specified boundary tag to the state stored in `reference` (the state array at the start of the step; `pg.rhos` in the six-argument form). Nothing else writes the state of a boundary
particle (the stage predictor keeps it), so that is its initial state unless the user changes `pg.rhos`.
"""
function (::FixedDirichlet)(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, tag::Int, ts, eq::HyperbolicPDE, t::Real,
                            reference::AbstractVector{State{M, T}}) where {D, M, T}
    foreach_particle(_dirichlet_body!, get_backend(), pg.meta.N, (rhos_buffer, reference, pg.core.tags, pg.core.is_boundary, tag))
    return nothing
end
(bc::FixedDirichlet)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq::HyperbolicPDE, t::Real) = bc(pg, rhos_buffer, tag, ts, eq, t, pg.rhos)

# Kernel body (see Backend.jl): a tagged boundary particle keeps its initial state.
@inline function _dirichlet_body!(i, buf, rhos, tags, is_boundary, tag)
    (is_boundary[i] && tags[i] == tag) && (buf[i] = rhos[i])
    return nothing
end

# 2. Outflow (Zero-Divergence)
"""
    (::OutflowBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

Enforces a zero-gradient outflow boundary condition using a multi-pass nearest-donor algorithm.
- Initiates all active interior particles as valid state donors, and marks boundary particles matching the specific tag as requiring resolution.
- Executes up to 5 symmetrical passes outward, dynamically locating the nearest resolved neighbor using squared distances and copying its state to the target particle.
- Safely falls back to the state in `reference` (the state array at the start of the step; `pg.rhos` in the six-argument form) for any completely orphaned boundary particles that failed to resolve a donor.
"""
function (::OutflowBC)(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, tag::Int, ts, eq::HyperbolicPDE, t::Real,
                       reference::AbstractVector{State{M, T}}) where {D, M, T}
    N       = pg.meta.N
    nb      = pg.neighbor
    status  = pg.shared.int_buffer      # 1: donor (interior), 0: unresolved target, -1: other boundary;
    backend = get_backend()             # pass + 1 once resolved in `pass`

    foreach_particle(_outflow_init!, backend, N, (status, pg.core.tags, pg.core.is_boundary, tag))

    # Each pass resolves from donors of earlier passes only, so no two particles read what another writes.
    for pass in 1:5
        foreach_particle(_outflow_pass!, backend, N, (rhos_buffer, status, nb.ranges, nb.indices, nb.distances, pass, T(Inf)))
        any(==(0), view(status, 1:N)) || break
    end

    foreach_particle(_outflow_fallback!, backend, N, (rhos_buffer, reference, status))
    return nothing
end
(bc::OutflowBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq::HyperbolicPDE, t::Real) = bc(pg, rhos_buffer, tag, ts, eq, t, pg.rhos)

# Kernel bodies of the outflow donor search (see Backend.jl).
@inline function _outflow_init!(i, status, tags, is_boundary, tag)
    status[i] = !is_boundary[i] ? 1 : (tags[i] == tag ? 0 : -1)
    return nothing
end

@inline function _outflow_pass!(i, buf, status, nb_slices, nb_indices, dists, pass, inf)
    status[i] == 0 || return nothing
    closest = -1
    min_d2  = inf
    @inbounds for k in nb_slices[i]
        j = nb_indices[k]
        s = status[j]
        if 0 < s <= pass
            d2 = sum(abs2, dists[k])
            if d2 < min_d2
                min_d2  = d2
                closest = j
            end
        end
    end
    if closest != -1
        buf[i]    = buf[closest]
        status[i] = pass + 1
    end
    return nothing
end

@inline function _outflow_fallback!(i, buf, rhos, status)
    status[i] == 0 && (buf[i] = rhos[i])
    return nothing
end

# The outflow condition of the blocks of a block-major state `Y` (`reference` likewise): the donor search once, on
# the particles (the passes record the donor of every resolved target in `donors` instead of copying a state), then
# every block copies in one launch from the root of the donor chain, the interior particle whose state the passes
# would have copied along it (bit-identical to the passes on every block)
function _outflow_blocks!(pg::ParticleGrid{D, M, T}, Y, reference, tag::Int, N::Int, blocks::UnitRange{Int}, donors) where {D, M, T}
    nb      = pg.neighbor
    status  = pg.shared.int_buffer
    backend = get_backend()
    foreach_particle(_outflow_init!, backend, N, (status, pg.core.tags, pg.core.is_boundary, tag))
    for pass in 1:5
        foreach_particle(_outflow_donor_pass!, backend, N, (donors, status, nb.ranges, nb.indices, nb.distances, pass, T(Inf)))
        any(==(0), view(status, 1:N)) || break
    end
    foreach_block(_outflow_copy!, backend, N, blocks, (PerBlock(Y), PerBlock(reference), status, donors))
    return nothing
end

# Kernel bodies (see Backend.jl): `_outflow_pass!` recording the donor, and the copy of a block from the chain's root
@inline function _outflow_donor_pass!(i, donors, status, nb_slices, nb_indices, dists, pass, inf)
    status[i] == 0 || return nothing
    closest = -1
    min_d2  = inf
    @inbounds for k in nb_slices[i]
        j = nb_indices[k]
        s = status[j]
        if 0 < s <= pass
            d2 = sum(abs2, dists[k])
            if d2 < min_d2
                min_d2  = d2
                closest = j
            end
        end
    end
    if closest != -1
        donors[i] = closest
        status[i] = pass + 1
    end
    return nothing
end
@inline function _outflow_copy!(i, buf, rhos, status, donors)
    @inbounds begin
        s = status[i]
        if s == 0
            buf[i] = rhos[i]
        elseif s > 1
            j = donors[i]
            while status[j] > 1              # a pass-`p` target's donor was resolved at an earlier pass: ≤ 5 hops
                j = donors[j]
            end
            buf[i] = buf[j]
        end
    end
    return nothing
end
# =========================================================================
# Ghost states from the interior: one-sided fits at the nearest interior particle
# =========================================================================
export ExtrapolationBC, MirrorBC, WallPoint, requires_shape, uses_wall_state

"""
    WallPoint{D, T, L}

The wall as seen from a ghost particle, handed to the operator of a [`MirrorBC`](@ref): `position` (the wall
point the ghost is reflected in), the outward unit `normal` there, the `distance` of the ghost to the wall, the
wall `velocity` and `acceleration` (those of a [`Moving`](@ref) piece at the wall point, [`wall_velocity`](@ref)
and [`wall_acceleration`](@ref); zero for a static wall), the piece's angular velocity as the skew matrix `spin`
(`Ω`: the wall velocity is `w(x) = v + Ω (x - c)`; zero for a static wall) and the `hessian` of the region's signed
distance at the wall point (the wall's curvature: `t ⋅ hessian * t` for a unit tangent `t` is negative where the
wall is convex seen from the region, `-1/r` around a disc obstacle, and positive where it is concave, `+1/r`
inside a circle; zero for a flat wall).

An operator that mirrors the pressure of a fluid at the wall needs the normal momentum balance there, which the
even mirror (`∂p/∂n = 0`) misses at first order at accelerating, rotating or curved walls. Following a fluid
particle along the wall, `(u - w) ⋅ n = 0` gives
`∂p/∂n = -ρ (n ⋅ a + 2 n ⋅ Ω (u - w) - u_r ⋅ hessian * u_r)` with `u_r` the tangential velocity relative to the
wall (for a disc spinning in a gas at rest the three terms cancel). The four- and five-argument constructors leave
the acceleration, spin and curvature zero.
"""
struct WallPoint{D, T, L}
    position::Space{D, T}
    normal::Space{D, T}
    distance::T
    velocity::Space{D, T}
    acceleration::Space{D, T}
    spin::SMatrix{D, D, T, L}
    hessian::SMatrix{D, D, T, L}
end
WallPoint(position::Space{D, T}, normal::Space{D, T}, distance, velocity::Space{D, T},
          acceleration::Space{D, T} = zero(Space{D, T})) where {D, T} =
    WallPoint{D, T, D * D}(position, normal, T(distance), velocity, acceleration,
                           zero(SMatrix{D, D, T, D * D}), zero(SMatrix{D, D, T, D * D}))

"""
    requires_shape(bc::AbstractBoundaryCondition) -> Bool

`true` if the condition reads the wall normals and distances of the particles, which only a domain built from
an [`AbstractShape`](@ref) provides. The `ParticleGrid` constructor refuses such a condition on a domain without
a shape. Defaults to `false`; [`MirrorBC`](@ref) returns `true`.
"""
requires_shape(::AbstractBoundaryCondition) = false

function _check_bc_shape(geom::GeometricDomain)
    geom.shape === nothing || return nothing
    bad = sort!([tag for (tag, bc) in geom.bc_map if requires_shape(bc)])
    isempty(bad) && return nothing
    throw(ArgumentError("the boundary condition(s) of tag(s) $bad need the wall normals of a shape-based domain " *
                        "(GeometricDomain(shape; ...)); this domain was built from an interior test and a tag function"))
end

"""
    ExtrapolationBC(degree = 1)

Transmissive condition of polynomial degree `degree` (0–5; in 3-D at most 3 is used): every boundary particle with the tag takes the value
at its own position of an MLS fit through the interior. The data of the fit are the interior particle nearest
to the boundary particle (the donor, found in its neighbour list and refined by walking the neighbour graph)
and the donor's interior neighbours; the fit is centred at the boundary particle and weighted by the distance
to it with the grid's weight function. It is exact for polynomials up to `degree`, so an outflow of degree `q`
keeps the first `q` derivatives of the interior solution across the boundary, where [`OutflowBC`](@ref) (a
donor copy) kinks it; degree 0 is the weighted average of the data.

The degree is lowered where it is not supported: to the order of the donor particle minus one (MOOD lowered it),
to what the data count carries (1.5× the unknowns, counting the constant term; a donor at a corner with too few
interior neighbours is replaced by its best-supported interior neighbour nearest to the target first), and
whenever the one-sided moment matrix is not positive definite even after the conditional regularisation. A
boundary particle without any interior particle within two hops keeps its state. The condition needs no wall
normals, so it works on every domain.

In Float32, degrees ≥ 3 lose accuracy at the ghosts farthest from the data (a cubic evaluated up to `R + 4Δx` beyond
one-sided data amplifies the round-off: 2-D, degree 3, jittered: 1e-1 relative error, against 1e-3 for degree 2);
use Float64 for them. A degree `q` needs enough interior data within the cutoff: in 1-D `R ≳ (q + 0.5)Δx`
(degree 4 runs as degree 3 at `R = 3.5Δx`).

Extrapolation is an *outflow* condition: the ghost states feed back into the interior scheme through the
stencils of the near-wall particles, and a high-degree one-sided extrapolation amplifies perturbations. At a
wall with tangential flow use [`MirrorBC`](@ref) (whose target point lies inside the cloud); see its docstring
and the docs for the degrees that were found stable.
"""
struct ExtrapolationBC{Q} <: AbstractBoundaryCondition
    function ExtrapolationBC{Q}() where {Q}
        (Q isa Int && 0 <= Q <= 5) || throw(ArgumentError("ExtrapolationBC: the degree must be an integer 0–5, got $Q"))
        return new{Q}()
    end
end
ExtrapolationBC(degree::Int = 1) = ExtrapolationBC{degree}()

"""
    MirrorBC(op = nothing; degree = 1, bounded = true)

Wall condition built from the mirror image: every boundary particle with the tag builds the interior fit of
[`ExtrapolationBC`](@ref) (same donor, data, degree rule and fallbacks) centred at its mirror point
`x - 2 φ n` behind the wall (`φ` its wall distance, `n` the outward normal; both from the domain's shape, see
[`wall_geometry`](@ref)) and stores `op(eq, U, wall::WallPoint)` of the value `U` found there (or
`op(eq, U, wall, U_wall)` with the same fit's value at the wall point, see [`uses_wall_state`](@ref)). The mirror
point lies inside the cloud, so the fit interpolates. Without `op`
the mirrored value itself is used, i.e. an even reflection of every component (a symmetry plane, zero normal
derivative). Physics packages supply the operators of their walls (reflecting the normal velocity for a slip
wall, a Maxwellian for a diffuse kinetic wall, ...); `op` must be a bits-type callable, and the state it
returns is used as it is.

With `bounded` (the default) the mirrored value is clamped, component by component, to the range of the data
of its fit (the donor and its interior neighbours) before `op` is applied: a discrete maximum principle on the
ghost state. It costs accuracy only where the value at the mirror point would lie outside the range of the
interior data around it (an extremum between the data points), and it is what keeps long runs along a wall
bounded: the ghost states feed back into the interior through the stencils of the near-wall particles, and
without the bound that loop grew without limit for degrees ≥ 2 with a 5th-order scheme on jittered clouds
(and slowly for degree 1), while every bounded case stayed below the initial maximum. `bounded = false` gives
the plain fit (exact for polynomials up to `degree` at every ghost).

For a planar wall the mirror point is the exact reflection, so the condition is as accurate as the fit; on a
curved wall the mirror image is second-order accurate. Behind a convex corner or edge (outside two faces) a ghost
is reflected in the face whose tag it carries, the one it lies farther outside of: its stored normal is that face's
and its distance is measured along it, so the mirror point lies behind the other face (outside the region; the fit
extrapolates the interior there, exactly for linear data) and `op` sees that face's normal. (The point reflection
through the corner, with the diagonal normal to it, gave `op` a normal that is right for neither face: a slip wall
turned the velocity along the wall into the wall at the corner with an inflow, and a uniform flow did not survive
the corner.) Requires a shape-based domain ([`requires_shape`](@ref)).
"""
struct MirrorBC{Q, F, B} <: AbstractBoundaryCondition
    op::F
    function MirrorBC{Q, F, B}(op::F) where {Q, F, B}
        (Q isa Int && 0 <= Q <= 5) || throw(ArgumentError("MirrorBC: the degree must be an integer 0–5, got $Q"))
        B isa Bool || throw(ArgumentError("MirrorBC: bounded must be a Bool"))
        isbits(op) || throw(ArgumentError("MirrorBC: the operator must be a bits type (a plain struct or function, no captured arrays)"))
        return new{Q, F, B}(op)
    end
end
MirrorBC(op = nothing; degree::Int = 1, bounded::Bool = true) = MirrorBC{degree, typeof(op), bounded}(op)
requires_shape(::MirrorBC) = true

function (::ExtrapolationBC{Q})(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t::Real) where {Q}
    _launch_ghost_fit!(pg, rhos_buffer, tag, eq, t, Val(Q), nothing, Val(false), Val(false))
    return nothing
end
function (bc::MirrorBC{Q, F, B})(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t::Real) where {Q, F, B}
    _launch_ghost_fit!(pg, rhos_buffer, tag, eq, t, Val(Q), _WallOp(bc.op), Val(true), Val(B))
    return nothing
end

# In 3-D the degree is capped at 3 (20 unknowns, the limit of the package's other 3-D fits): degrees 4-5 (35/56
# unknowns) would take StaticArrays' allocating, host-only Cholesky inside the kernel
_ghost_degree(::Val{Q}, ::Val{D}) where {Q, D} = Val(D == 3 ? min(Q, 3) : Q)

# `out`: the array the ghost states are written to (the data of the fit are always read from `buf`; the block pipeline
# fits into a scratch array first and applies its wall operators from there)
function _launch_ghost_fit!(pg::ParticleGrid{D, M, T}, buf, tag::Int, eq, t, valQ::Val, op, mirror::Val, bounded::Val;
                            out = buf) where {D, M, T}
    valQ = _ghost_degree(valQ, Val(D))
    nb   = pg.neighbor
    invL = one(T) / minimum(pg.meta.dx)
    # A moving piece: the stored ghost positions and normals belong to `boundary_time`; the kernel moves them to
    # the time of the state being built (the interior cloud does not move), and the wall point gets the velocity
    gm = (has_moving_pieces(pg.geometry) ? tag_motion(pg.geometry, tag, pg.meta.boundary_time, t) :
                                           GhostMotion(Val(D), T))::GhostMotion{D, T, D * D}   # (inferred: the launch must not box)
    foreach_particle(_ghost_fit_body!, get_backend(), pg.meta.N,
        ((buf, out), pg.core.positions, pg.core.is_boundary, pg.core.tags, pg.core.normals, pg.core.wall_distances,
         pg.core.wall_curvatures, pg.core.particle_orders, nb.ranges, nb.indices, nb.weight_func,
         (pg.domain.L_wrap, pg.domain.invL_wrap, invL), gm, tag, valQ, op, eq, mirror, bounded, Val(D)))   # (≤ 20 arguments)
    return nothing
end

# The operator enters the kernel inside a struct: a plain `Function` argument makes the Polyester loop allocate
# (368 bytes per launch), a struct holding it does not.
struct _WallOp{F}
    op::F
end
@inline _apply_wall_op(::_WallOp{Nothing}, eq, U, wall) = U
@inline _apply_wall_op(w::_WallOp, eq, U, wall) = w.op(eq, U, wall)
@inline _apply_wall_op(::_WallOp{Nothing}, eq, U, wall, Uw) = U
@inline _apply_wall_op(w::_WallOp, eq, U, wall, Uw) = w.op(eq, U, wall, Uw)
@inline _uses_wall_state(::_WallOp{Nothing}) = false
@inline _uses_wall_state(w::_WallOp) = uses_wall_state(w.op)

"""
    uses_wall_state(op) -> Bool

Whether a wall operator of [`MirrorBC`](@ref) takes the fit's value at the wall point (the foot of the ghost's
normal) as a fourth argument: `op(eq, U, wall, U_wall)`, with `U` the fitted value at the ghost's mirror point. A
balance imposed at the wall belongs at the wall point, a diffuse kinetic wall's returning flux for one: taken at the
mirror point, it differs from ghost to ghost by the depth of the mirror point (`O(Δx ∂f)`), a first-order error of
the scheme's flux through the wall out of equilibrium. `U_wall` is clamped to the donor's data range like `U` when
the condition is `bounded`. In the block pipeline such an operator's reduction gets the wall-point states of all
blocks as a further scratch array, `block_wall_reduce(op, eq, Y, N, nb, g, wall, Yw)`. Default `false`:
`op(eq, U, wall)`.
"""
uses_wall_state(op) = false

@inline _min_image(dx::Space{D, T}, L::Space{D, T}, Li::Space{D, T}) where {D, T} =
    Space{D, T}(ntuple(d -> dx[d] - L[d] * round(dx[d] * Li[d]), Val(D)))

# The interior neighbour of `c` nearest to the point `x_t` (squared distance), `0` if it has none.
@inline function _nearest_interior(c, x_t, ranges, indices, isb, pos, L, Li)
    j, best = 0, typemax(eltype(x_t))
    @inbounds for k in ranges[c]
        m = indices[k]
        isb[m] && continue
        d2 = sum(abs2, _min_image(pos[m] - x_t, L, Li))
        d2 < best && (j = m; best = d2)
    end
    return j, best
end

# The donor `j` and its interior neighbours: the data of the fit.
@inline function _support_count(j, ranges, indices, isb)
    n = 1
    @inbounds for k in ranges[j]
        isb[indices[k]] || (n += 1)
    end
    return n
end

# Unknowns of the degree-`q` fit with its constant term, with the margin of 1.5× that keeps the coefficients of a
# one-sided fit bounded (an exactly determined fit amplifies noise, and the ghost states feed back into the interior).
@inline _fit_support(D, q) = (3 * (basis_count(D, q) + 1) + 1) ÷ 2

# The donor of ghost `i` for a fit at the target point `x_t`, and the degree it carries; `j == 0`: no interior neighbour.
# Donor: the interior particle nearest to the target point. Its neighbour list is the stencil of the fit (every
# interior particle within R - |x_t - x_j| of the target is in it). Searched in the ghost's own neighbour list, then
# in its neighbours' lists (the outermost ghosts of the R + dx band have no interior neighbour), then refined by
# walking the neighbour graph towards the target.
# Fit degree and donor: the requested degree, at most the donor's order - 1 (0 = order never set), lowered one step
# at a time until the donor or, failing that, its interior neighbour nearest to the target carries the degree with
# the margin. A donor at a wall or corner has too few interior neighbours for a high degree (3-4 for the 6 unknowns of
# a 2-D quadratic) while its neighbours further inside do; the fit stays exact for polynomials whatever the distance.
# (Trying the requested degree only and otherwise falling back to what the wall donor supports made a higher request
# give a lower degree: 1-D, R = 3.5Δx, degree 4 ran as degree 1 while degree 3 ran as 3.)
@inline function _ghost_donor(i, x_t, pos, isb, orders, ranges, indices, L, Li, ::Val{Q}, ::Val{D}) where {Q, D}
    @inbounds begin
        j, best = _nearest_interior(i, x_t, ranges, indices, isb, pos, L, Li)
        if j == 0
            for k in ranges[i]
                j2, best2 = _nearest_interior(indices[k], x_t, ranges, indices, isb, pos, L, Li)
                (j2 != 0 && best2 < best) && (j = j2; best = best2)
            end
        end
        j == 0 && return 0, 0
        for _ in 1:3
            j2, best2 = _nearest_interior(j, x_t, ranges, indices, isb, pos, L, Li)
            (j2 != 0 && best2 < best) || break
            j, best = j2, best2
        end
        q = Q
        n_sup = _support_count(j, ranges, indices, isb)
        jd = j
        while q > 0
            needed = _fit_support(D, q)
            n_sup >= needed && break
            jb, bb = 0, typemax(eltype(x_t))
            for k in ranges[j]
                m = indices[k]
                (isb[m] || _support_count(m, ranges, indices, isb) < needed) && continue
                d2 = sum(abs2, _min_image(pos[m] - x_t, L, Li))
                d2 < bb && (jb = m; bb = d2)
            end
            jb != 0 && (jd = jb; break)
            q -= 1
        end
        j = jd
        o = orders[j]                                  # the MOOD cap of the donor that is fitted (a lower degree
        o > 0 && (q = min(q, o - 1))                   # needs less support, so the donor still carries it)
        return j, q
    end
end

# Discrete maximum principle on the ghost state: the mirrored value stays within the range of the data it was built
# from. Without it the feedback ghost -> near-wall stencils -> ghost grew without bound for degrees >= 2 with a
# 5th-order scheme (long runs along a wall), with it every tested case stayed bounded.
@inline function _ghost_clamp(U, j, buf, ranges, indices, isb)
    @inbounds begin
        lo = buf[j]
        hi = buf[j]
        for k in ranges[j]
            m = indices[k]
            isb[m] && continue
            lo = min.(lo, buf[m])
            hi = max.(hi, buf[m])
        end
        return clamp.(U, lo, hi)
    end
end

# Kernel body (see Backend.jl): the ghost state of particle `i` from an MLS fit centred at its target point.
@inline function _ghost_fit_body!(i, bufs, pos, isb, tags, normals, wdist, curvs, orders, ranges, indices, wf,
                                  wrap, gm, tag, ::Val{Q}, op, eq, ::Val{MIRROR}, ::Val{BOUNDED}, ::Val{D}) where {Q, MIRROR, BOUNDED, D}
    @inbounds begin
        (isb[i] && tags[i] == tag) || return nothing
        buf, out = bufs
        L, Li, invL = wrap
        x_i = move(gm, pos[i])
        n_i = rotate(gm, normals[i])
        φ_i = wdist[i]
        x_t = MIRROR ? x_i - 2φ_i * n_i : x_i
        j, q = _ghost_donor(i, x_t, pos, isb, orders, ranges, indices, L, Li, Val(Q), Val(D))
        j == 0 && return nothing
        if MIRROR && _uses_wall_state(op)
            # the operator takes the fit at the wall point too: the same fit evaluated there
            U, Uw = _fit_eval2(Val(Q), q, x_i - φ_i * n_i, j, x_t, buf, ranges[j], indices, pos, isb, L, Li, wf, invL)
            if BOUNDED
                U  = _ghost_clamp(U, j, buf, ranges, indices, isb)
                Uw = _ghost_clamp(Uw, j, buf, ranges, indices, isb)
            end
            out[i] = _apply_wall_op(op, eq, U, _ghost_wall_point(i, pos, normals, wdist, curvs, gm), Uw)
        else
            U = _fit_eval(Val(Q), q, j, x_t, buf, ranges[j], indices, pos, isb, L, Li, wf, invL)
            BOUNDED && (U = _ghost_clamp(U, j, buf, ranges, indices, isb))
            if MIRROR
                out[i] = _apply_wall_op(op, eq, U, _ghost_wall_point(i, pos, normals, wdist, curvs, gm))
            else
                out[i] = U
            end
        end
    end
    return nothing
end

# =========================================================================
# The ghost fits of a block-major state (the block pipeline): the geometry once per ghost, the fit per (ghost, block)
# =========================================================================
"""
    GhostFitScratch

Per-ghost geometry of the ghost fits of a block-major state, computed once per boundary tag and application
(`_ghost_geometry_body!`) and reused by every block (`_ghost_apply_body!`): the donor (`0`: none), the degree whose
moment matrix factorised (`-1`: none, the donor's value is used), the basis scale, the target point and the upper
Cholesky factor of the moment matrix, padded to `B` rows (the basis length of the largest degree plus one), and the fit
as weights of the stencil's states: the fit is linear in the data, so its value at the target is `Σ_t wt[t] U_t` and at
the wall point `Σ_t wt_w[t] U_t` over the donor and its interior neighbours (`S = max_nb + 1` entries per particle, at
`(i − 1) S`). Every block's fit is then a weighted sum instead of the moment sums, the weight function and the
triangular solves anew (the same fit up to round-off). `Growable` buffers of a storage array type
(`_ghost_scratch(Val(B), T, Val(D), storage)`), on the block stepper; the weights are not part of the memory budget
(`S` entries per particle: they grow on demand).
"""
struct GhostFitScratch{B, L, T, D, GI <: AbstractVector{Int}, GT <: AbstractVector{T}, GX <: AbstractVector{Space{D, T}},
                       GF <: AbstractVector{SMatrix{B, B, T, L}}}
    donor::GI
    degree::GI
    scale::GT
    target::GX
    factor::GF
    wt::GT
    wt_w::GT
end
_ghost_scratch(::Val{B}, ::Type{T}, ::Val{D}, donor, degree, scale, target, factor, wt, wt_w) where {B, T, D} =
    GhostFitScratch{B, B * B, T, D, typeof(donor), typeof(scale), typeof(target), typeof(factor)}(donor, degree, scale, target, factor, wt, wt_w)
_ghost_scratch(::Val{B}, ::Type{T}, ::Val{D}, ::Type{AT}) where {B, T, D, AT <: AbstractArray} =
    _ghost_scratch(Val(B), T, Val(D), Growable{Int}(AT), Growable{Int}(AT), Growable{T}(AT), Growable{Space{D, T}}(AT),
                   Growable{SMatrix{B, B, T, B * B}}(AT), Growable{T}(AT), Growable{T}(AT))
Adapt.adapt_structure(to, g::GhostFitScratch{B, L, T, D}) where {B, L, T, D} =
    _ghost_scratch(Val(B), T, Val(D), adapt(to, g.donor), adapt(to, g.degree), adapt(to, g.scale), adapt(to, g.target), adapt(to, g.factor),
                   adapt(to, g.wt), adapt(to, g.wt_w))
_scratch_buffers(g::GhostFitScratch) = (g.donor, g.degree, g.scale, g.target, g.factor)
function _ensure_scratch!(g::GhostFitScratch, N::Int)
    foreach(b -> ensure_capacity!(b, N), _scratch_buffers(g))
    return nothing
end
# `B` of the scratch for the fits of degree `Q` in `D` dimensions (the degree-0 fit has the constant term only)
_ghost_scratch_rows(::Val{D}, ::Val{0}) where {D} = 1
_ghost_scratch_rows(::Val{D}, ::Val{q}) where {D, q} = _val_of(basis_length(Val(D), Val(q))) + 1
_val_of(::Val{B}) where {B} = B

# The ghost fits of the blocks `blocks` of the block-major state `buf` into `out`, in two launches: the geometry of
# every ghost of `tag` into `gs` (the donor, the degree, the factor and the fit's weights of the stencil's states), then
# the fit per (ghost, block) as the weighted sum of the block's states, with the block equation `BlockOf(eq, b)`. The fit
# of `_launch_ghost_fit!` per block up to round-off (nothing of it but the states depends on the block).
function _launch_ghost_fit_blocks!(pg::ParticleGrid{D, M, T}, buf, out, tag::Int, eq, t, valQ::Val, op, mirror::Val, bounded::Val,
                                   blocks::UnitRange{Int}, gs::GhostFitScratch; block_offset::Int = 0, out_w = nothing,
                                   list = nothing) where {D, M, T}
    valQ = _ghost_degree(valQ, Val(D))
    nb   = pg.neighbor
    N    = pg.meta.N
    invL = one(T) / minimum(pg.meta.dx)
    gm = (has_moving_pieces(pg.geometry) ? tag_motion(pg.geometry, tag, pg.meta.boundary_time, t) :
                                           GhostMotion(Val(D), T))::GhostMotion{D, T, D * D}
    S    = pg.meta.max_nb + 1                    # (the stencil of a fit: the donor and its neighbours)
    wrap = (pg.domain.L_wrap, pg.domain.invL_wrap, invL, S)
    c = pg.core
    backend = get_backend()
    _ensure_scratch!(gs, N)
    ensure_capacity!(gs.wt, N * S)
    ensure_capacity!(gs.wt_w, N * S)
    geom_args = (gs, c.positions, c.is_boundary, c.tags, c.normals, c.wall_distances, c.particle_orders, nb.ranges, nb.indices, nb.weight_func,
                 wrap, gm, tag, valQ, mirror, Val(D))
    # (`list = (indices, n)`: the ghosts of the tag only, see `_tag_list!`; the same per-ghost work either way)
    list === nothing ? foreach_particle(_ghost_geometry_body!, backend, N, geom_args) :
                       foreach_listed(_ghost_geometry_body!, backend, list..., geom_args)
    foreach_block(_ghost_apply_body!, backend, N, blocks,
        ((PerBlock(buf), PerBlock(out), out_w === nothing ? nothing : PerBlock(out_w)), gs,
         c.positions, c.is_boundary, c.tags, c.normals,
         c.wall_distances, c.wall_curvatures,
         nb.ranges, nb.indices, nb.weight_func, wrap, gm, tag, valQ, op, PerBlockEq(eq), mirror, bounded); block_offset, list)   # (19 arguments)
    return nothing
end

# Kernel bodies (see Backend.jl): the geometry of ghost `i` (its donor, the degree whose moment matrix factorises, the
# factor), and the fit of one block from it
@inline function _ghost_geometry_body!(i, gs::GhostFitScratch{B}, pos, isb, tags, normals, wdist, orders, ranges, indices, wf,
                                       wrap, gm, tag, ::Val{Q}, ::Val{MIRROR}, ::Val{D}) where {B, Q, MIRROR, D}
    @inbounds begin
        (isb[i] && tags[i] == tag) || return nothing
        L, Li, invL, S = wrap
        x_i = move(gm, pos[i])
        n_i = rotate(gm, normals[i])
        φ_i = wdist[i]
        x_t = MIRROR ? x_i - 2φ_i * n_i : x_i
        j, q = _ghost_donor(i, x_t, pos, isb, orders, ranges, indices, L, Li, Val(Q), Val(D))
        gs.donor[i] = j
        j == 0 && return nothing
        q_ok, F, s = _fit_geometry(Val(Q), q, Val(B), j, x_t, ranges[j], indices, pos, isb, L, Li, wf, invL)
        gs.degree[i] = q_ok
        gs.factor[i] = F
        gs.scale[i]  = s
        gs.target[i] = x_t
        # the fit as weights of the stencil's states, at the target and at the wall point
        q_ok >= 0 && _fit_weights!(Val(Q), q_ok, F, x_i - φ_i * n_i, j, x_t, s, ranges[j], indices, pos, isb, L, Li, wf,
                                   gs.wt, gs.wt_w, (i - 1) * S)
    end
    return nothing
end
@inline function _ghost_apply_body!(i, bufs, gs::GhostFitScratch, pos, isb, tags, normals, wdist, curvs, ranges, indices, wf,
                                    wrap, gm, tag, ::Val{Q}, op, eq, ::Val{MIRROR}, ::Val{BOUNDED}) where {Q, MIRROR, BOUNDED}
    @inbounds begin
        (isb[i] && tags[i] == tag) || return nothing
        j = gs.donor[i]
        j == 0 && return nothing
        buf, out, out_w = bufs
        L, Li, invL, S = wrap
        q_ok = gs.degree[i]
        off  = (i - 1) * S
        if out_w !== nothing || (MIRROR && _uses_wall_state(op))
            # the wall-point state as well: into its own scratch (`out_w`, the fit pass of a block wall), or for the
            # operator applied here
            U, Uw = q_ok < 0 ? (buf[j], buf[j]) : _weighted_fit2(gs.wt, gs.wt_w, off, j, buf, ranges[j], indices, isb)
            if BOUNDED
                U  = _ghost_clamp(U, j, buf, ranges, indices, isb)
                Uw = _ghost_clamp(Uw, j, buf, ranges, indices, isb)
            end
            _store_wall_state!(out_w, i, Uw)
            out[i] = MIRROR ? _apply_wall_op(op, eq, U, _ghost_wall_point(i, pos, normals, wdist, curvs, gm), Uw) : U
        else
            U = q_ok < 0 ? buf[j] : _weighted_fit(gs.wt, off, j, buf, ranges[j], indices, isb)
            BOUNDED && (U = _ghost_clamp(U, j, buf, ranges, indices, isb))
            out[i] = MIRROR ? _apply_wall_op(op, eq, U, _ghost_wall_point(i, pos, normals, wdist, curvs, gm)) : U
        end
    end
    return nothing
end
@inline _store_wall_state!(::Nothing, i, Uw) = nothing

# The stencil of a ghost fit: the donor `j`, then its interior neighbours in slot order (the order of the moment sums)
@inline function _weighted_fit(wt, off, j, buf, rg, indices, isb)
    @inbounds begin
        U = wt[off + 1] * buf[j]
        t = 1
        for k in rg
            m = indices[k]
            isb[m] && continue
            t += 1
            U += wt[off + t] * buf[m]
        end
        return U
    end
end
@inline function _weighted_fit2(wt, wt_w, off, j, buf, rg, indices, isb)
    @inbounds begin
        Uj = buf[j]
        U, Uw = wt[off + 1] * Uj, wt_w[off + 1] * Uj
        t = 1
        for k in rg
            m = indices[k]
            isb[m] && continue
            t += 1
            Um = buf[m]
            U  += wt[off + t] * Um
            Uw += wt_w[off + t] * Um
        end
        return U, Uw
    end
end
# The weights of the fit of degree `q_ok` (static dispatch over the degrees as `_fit_geometry`): the fit at the target is the constant
# coefficient `e₁ᵀ C⁻¹ b`, at the wall point `p_wᵀ C⁻¹ b`, with `b = Σ_t w_t p_t U_tᵀ`: the weight of `U_t` is
# `w_t (z ⋅ p_t)` with `z = C⁻¹ e₁` (resp. `C⁻¹ p_w`)
@inline function _fit_weights!(::Val{q}, q_ok::Int, F, x_w, j, x_t, args...) where {q}
    q_ok == q && return _weights_degree(Val(q), F, x_w, j, x_t, args...)
    return _fit_weights!(Val(q - 1), q_ok, F, x_w, j, x_t, args...)
end
@inline _fit_weights!(::Val{0}, q_ok::Int, F, x_w, j, x_t, args...) = _weights_degree(Val(0), F, x_w, j, x_t, args...)
@inline _weights_degree(::Val{q}, F, x_w, j, x_t::Space{D, T}, args...) where {q, D, T} =
    _weights_degree_b(basis_length(Val(D), Val(q)), Val(q), F, x_w, j, x_t, args...)
@inline _weights_degree(::Val{0}, F, x_w, j, x_t::Space{D, T}, args...) where {D, T} = _weights_degree_b(Val(0), Val(0), F, x_w, j, x_t, args...)
@inline function _weights_degree_b(::Val{B}, ::Val{q}, F, x_w, j, x_t::Space{D, T}, s, rg, indices, pos, isb, L, Li, wf,
                                   wt, wt_w, off) where {B, q, D, T}
    C  = Cholesky(_top_left(F, Val(B + 1)), 'U', 0)
    z  = C \ SVector{B + 1, T}(ntuple(r -> r == 1 ? one(T) : zero(T), Val(B + 1)))
    zw = C \ _target_basis(Val(q), (x_w - x_t) * s)
    @inbounds begin
        r = _min_image(pos[j] - x_t, L, Li)
        p = _target_basis(Val(q), r * s)
        w = wf(sum(abs2, r))
        wt[off + 1], wt_w[off + 1] = w * dot(z, p), w * dot(zw, p)
        t = 1
        for k in rg
            m = indices[k]
            isb[m] && continue
            t += 1
            r = _min_image(pos[m] - x_t, L, Li)
            p = _target_basis(Val(q), r * s)
            w = wf(sum(abs2, r))
            wt[off + t], wt_w[off + t] = w * dot(z, p), w * dot(zw, p)
        end
    end
    return nothing
end
@inline _store_wall_state!(out_w, i, Uw) = (@inbounds out_w[i] = Uw; nothing)

# Static dispatch over the degrees Q, Q-1, …, 0 as `_fit_eval`: the geometry pass takes the first degree `≤ q_run`
# whose moment matrix factorises (`-1` if none: the apply pass uses the donor's value), the apply pass the stored one
@inline function _fit_geometry(::Val{q}, q_run::Int, ::Val{B}, args...) where {q, B}
    if q_run >= q
        F, s, ok = _geometry_degree(Val(q), args...)
        ok && return q, _pad_factor(F, Val(B)), s
    end
    return _fit_geometry(Val(q - 1), q_run, Val(B), args...)
end
@inline function _fit_geometry(::Val{0}, q_run::Int, ::Val{B}, args...) where {B}
    F, s, ok = _geometry_degree(Val(0), args...)
    return (ok ? 0 : -1), _pad_factor(F, Val(B)), s
end
@inline _geometry_degree(::Val{q}, j, x_t::Space{D, T}, args...) where {q, D, T} =
    _geometry_degree_b(basis_length(Val(D), Val(q)), Val(q), j, x_t, args...)
@inline _geometry_degree(::Val{0}, j, x_t::Space{D, T}, args...) where {D, T} = _geometry_degree_b(Val(0), Val(0), j, x_t, args...)
@inline function _geometry_degree_b(::Val{B}, ::Val{q}, j, x_t::Space{D, T}, rg, indices, pos, isb, L, Li, wf, invL) where {B, q, D, T}
    N_s, s = _ghost_moment_matrix(Val(B), Val(q), j, x_t, rg, indices, pos, isb, L, Li, wf, invL)
    C, ok  = _cholesky_moment(N_s)
    return C.factors, s, ok
end
# The fitted polynomial (coefficients `c_s`, basis scaled by `s`) at the target (its constant term) and at the offset `d_w`
@inline function _fit_values(::Val{q}, c_s::SMatrix{B1, M, T}, d_w::Space{D, T}, s) where {q, B1, M, T, D}
    return State{M, T}(ntuple(m -> c_s[1, m], Val(M))), _fit_at(Val(q), c_s, d_w * s)
end
@inline function _fit_at(::Val{q}, c_s::SMatrix{B1, M, T}, d::SVector{D, T}) where {q, B1, M, T, D}
    p = _target_basis(Val(q), d)
    return State{M, T}(ntuple(m -> sum(ntuple(r -> p[r] * c_s[r, m], Val(B1))), Val(M)))
end
# The factor of a degree below the largest, stored in the scratch's rows: zero-padded, and read back
@inline _pad_factor(F::SMatrix{B, B, T}, ::Val{Bm}) where {B, T, Bm} =
    SMatrix{Bm, Bm, T}(ntuple(k -> (r = (k - 1) % Bm + 1; c = (k - 1) ÷ Bm + 1; (r <= B && c <= B) ? (@inbounds F[r, c]) : zero(T)), Val(Bm * Bm)))
@inline _top_left(F::SMatrix{Bm, Bm, T}, ::Val{B}) where {Bm, T, B} =
    SMatrix{B, B, T}(ntuple(k -> (@inbounds F[(k - 1) % B + 1, (k - 1) ÷ B + 1]), Val(B * B)))

# The moment matrix of the fit (`_ghost_moments` without the right-hand side) with the basis scale: the same sums in the
# same order. Kept out of line like `_ghost_moments`.
@noinline function _ghost_moment_matrix(::Val{B}, ::Val{q}, j, x_t::Space{D, T}, rg, indices, pos, isb, L, Li, wf, invL) where {B, q, D, T}
    N_s = zero(SMatrix{B + 1, B + 1, T, (B + 1) * (B + 1)})
    @inbounds begin
        r2max = sum(abs2, _min_image(pos[j] - x_t, L, Li))
        for k in rg
            m = indices[k]
            isb[m] && continue
            r2max = max(r2max, sum(abs2, _min_image(pos[m] - x_t, L, Li)))
        end
        s   = r2max > zero(T) ? inv(sqrt(r2max)) : invL
        r   = _min_image(pos[j] - x_t, L, Li)
        p   = _target_basis(Val(q), r * s)
        w   = wf(sum(abs2, r))
        N_s += w * (p * p')
        for k in rg
            m = indices[k]
            isb[m] && continue
            r    = _min_image(pos[m] - x_t, L, Li)
            p    = _target_basis(Val(q), r * s)
            w    = wf(sum(abs2, r))
            N_s += w * (p * p')
        end
        return N_s, s
    end
end
# The wall as seen from ghost `i` (the wall point: the body's velocity there, not at the ghost), at the time of `gm`
@inline function _ghost_wall_point(i, pos, normals, wdist, curvs, gm)
    @inbounds begin
        x_i = move(gm, pos[i])
        n_i = rotate(gm, normals[i])
        φ_i = wdist[i]
        x_w = x_i - φ_i * n_i
        H   = gm.R * _decode_curvature(curvs[i], normals[i]) * gm.R'   # (stored for `normals[i]`)
        return WallPoint(x_w, n_i, φ_i, velocity(gm, x_w), acceleration(gm, x_w), gm.Ω, H)
    end
end

# Static dispatch over the fit degrees Q, Q-1, ..., 0 (kernels cannot build a `Val` at run time): the first
# degree `≤ q_run` whose moment matrix factorises gives the value; degree 0 is the weighted average.
@inline function _fit_eval(::Val{q}, q_run::Int, args...) where {q}
    if q_run >= q
        U, ok = _fit_degree(Val(q), args...)
        ok && return U
    end
    return _fit_eval(Val(q - 1), q_run, args...)
end
@inline function _fit_eval(::Val{0}, q_run::Int, args...)
    U, ok = _fit_degree(Val(0), args...)
    return U
end
# The same with the fit evaluated at the wall point `x_w` as well: `(U_target, U_w)`
@inline function _fit_eval2(::Val{q}, q_run::Int, x_w, args...) where {q}
    if q_run >= q
        U, Uw, ok = _fit_degree2(Val(q), x_w, args...)
        ok && return U, Uw
    end
    return _fit_eval2(Val(q - 1), q_run, x_w, args...)
end
@inline function _fit_eval2(::Val{0}, q_run::Int, x_w, args...)
    U, Uw, ok = _fit_degree2(Val(0), x_w, args...)
    return U, Uw
end
@inline _fit_degree2(::Val{q}, x_w, j, x_t::Space{D, T}, args...) where {q, D, T} =
    _fit_degree_b2(basis_length(Val(D), Val(q)), Val(q), x_w, j, x_t, args...)
@inline _fit_degree2(::Val{0}, x_w, j, x_t::Space{D, T}, args...) where {D, T} = _fit_degree_b2(Val(0), Val(0), x_w, j, x_t, args...)
@inline function _fit_degree_b2(::Val{B}, ::Val{q}, x_w, j, x_t::Space{D, T}, buf::AbstractVector{State{M, T}}, rg, indices,
                                pos, isb, L, Li, wf, invL) where {B, q, D, T, M}
    N_s, b_s, s = _ghost_moments(Val(B), Val(q), j, x_t, buf, rg, indices, pos, isb, L, Li, wf, invL)
    C, ok = _cholesky_moment(N_s)
    ok || return (@inbounds(buf[j]), @inbounds(buf[j]), false)
    c_s = C \ b_s
    U, Uw = _fit_values(Val(q), c_s, x_w - x_t, s)
    return U, Uw, true
end

@inline _fit_degree(::Val{q}, j, x_t::Space{D, T}, args...) where {q, D, T} =
    _fit_degree_b(basis_length(Val(D), Val(q)), Val(q), j, x_t, args...)
@inline _fit_degree(::Val{0}, j, x_t::Space{D, T}, args...) where {D, T} = _fit_degree_b(Val(0), Val(0), j, x_t, args...)

@inline _target_basis(::Val{0}, d::SVector{D, T}) where {D, T} = SVector{1, T}(one(T))
@inline _target_basis(::Val{q}, d::SVector{D, T}) where {q, D, T} = _intercept_basis(Val(q), d)

# MLS fit of degree `q` centred at the target point over the donor `j` and its interior neighbours, weighted by
# their distance to the target (the weight function of the grid, no cutoff: the stencil is the donor's): the
# fit's constant term is the value at the target. Centred at the target and weighted from it, the fit
# interpolates where the target lies inside the cloud (a mirror point) and extrapolates as little as possible
# where it does not (a ghost position); a fit centred at the donor and evaluated away from it amplified noise on
# jittered clouds and destabilised the interior scheme.
@inline function _fit_degree_b(::Val{B}, ::Val{q}, j, x_t::Space{D, T}, buf::AbstractVector{State{M, T}}, rg, indices,
                               pos, isb, L, Li, wf, invL) where {B, q, D, T, M}
    N_s, b_s, _ = _ghost_moments(Val(B), Val(q), j, x_t, buf, rg, indices, pos, isb, L, Li, wf, invL)
    C, ok = _cholesky_moment(N_s)
    ok || return (@inbounds(buf[j]), false)
    c_s = C \ b_s
    return (State{M, T}(ntuple(m -> c_s[1, m], Val(M))), true)
end

# The weighted moments of the fit. Kept out of line on purpose: accumulated in the same function as the Cholesky
# solve, Julia 1.13.3 / LLVM 20 miscompiled the Float32 3-D degree-2 case on the CPU backends (an unsymmetric moment
# matrix, ghost errors of 2-7 on data of size 4; KA and Float64 were right, and the trigger depended on the thread
# count). A separate function avoids the code shape; the Float32 3-D test guards it.
@noinline function _ghost_moments(::Val{B}, ::Val{q}, j, x_t::Space{D, T}, buf::AbstractVector{State{M, T}}, rg, indices,
                                  pos, isb, L, Li, wf, invL) where {B, q, D, T, M}
    N_s = zero(SMatrix{B + 1, B + 1, T, (B + 1) * (B + 1)})
    b_s = zero(SMatrix{B + 1, M, T, (B + 1) * M})
    @inbounds begin
        # The basis is scaled by the largest data distance seen from the target, not by the grid spacing: a ghost
        # position lies up to R + 4Δx from its data, so with 1/Δx the degree-q columns were ~7^q times the constant
        # one, and the Float32 moment matrix lost the fit (2-D degree 3: 9e-2 relative error). The value at the
        # target, the constant term, does not depend on the scale.
        r2max = sum(abs2, _min_image(pos[j] - x_t, L, Li))
        for k in rg
            m = indices[k]
            isb[m] && continue
            r2max = max(r2max, sum(abs2, _min_image(pos[m] - x_t, L, Li)))
        end
        s   = r2max > zero(T) ? inv(sqrt(r2max)) : invL
        r   = _min_image(pos[j] - x_t, L, Li)
        p   = _target_basis(Val(q), r * s)
        w   = wf(sum(abs2, r))
        N_s += w * (p * p')
        b_s += w * (p * buf[j]')
        for k in rg
            m = indices[k]
            isb[m] && continue
            r    = _min_image(pos[m] - x_t, L, Li)
            p    = _target_basis(Val(q), r * s)
            w    = wf(sum(abs2, r))
            N_s += w * (p * p')
            b_s += w * (p * buf[m]')
        end
    end
    return N_s, b_s, s
end
