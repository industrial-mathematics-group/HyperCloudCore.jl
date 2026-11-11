export FixedDirichlet, OutflowBC

"""
    FixedDirichlet <: AbstractBoundaryCondition
    (::FixedDirichlet)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

A strict, unchanging boundary condition strategy.

# Callable / Functor

Enforces a static Dirichlet condition upon boundary particles. Iterates over the grid and resets the state of any boundary particle matching the specified `tag` to the state stored in `pg.rhos` at the start of the step. Nothing else writes the state of a boundary
particle (the stage predictor keeps it), so that is its initial state unless the user changes `pg.rhos`.
"""
struct FixedDirichlet <: AbstractBoundaryCondition end

"""
    OutflowBC <: AbstractBoundaryCondition
    (::OutflowBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

A zero-gradient, transmissive boundary condition strategy.

# Callable / Functor

Enforces a zero-gradient outflow utilizing a multi-pass nearest-donor algorithm. 
- Initiates all active interior particles as valid state donors.
- Executes up to 5 symmetrical passes outward, dynamically locating the nearest resolved neighbor using squared distances and copying its state to the target particle.
- Safely falls back to the state in `pg.rhos` at the start of the step for any completely orphaned boundary particles that fail to resolve a donor.
"""
struct OutflowBC <: AbstractBoundaryCondition end

# =========================================================================
# BOUNDARY CONDITION DISPATCHER
# =========================================================================

"""
    apply_boundary_conditions!(pg::ParticleGrid, rhos_buffer, ts, eq, t)

Iterates over the active boundary condition map stored in the grid's geometry configuration and dispatches the corresponding strategy for each registered domain tag.
The map itself (a `Dict`) stays on the host; the conditions shipped with the package run as kernel
launches, so they work on a grid built with `device_grid` too. A user-defined condition is any callable
`(bc)(pg, rhos_buffer, tag, ts, eq, t)`.
"""
function apply_boundary_conditions!(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, ts::TimeStepper, eq::HyperbolicPDE, t::Real) where {D, M, T}
    for (tag, bc_functor) in pg.geometry.bc_map
        bc_functor(pg, rhos_buffer, tag, ts, eq, t)
    end
    return nothing
end

# =========================================================================
# CONCRETE BOUNDARY CONDITIONS
# =========================================================================

# 1. Fixed Dirichlet
"""
    (::FixedDirichlet)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)

Enforces a static Dirichlet condition upon boundary particles. 
- Iterates over the grid and resets the state of any particle matching the specified boundary tag to the state stored in `pg.rhos` at the start of the step. Nothing else writes the state of a boundary
particle (the stage predictor keeps it), so that is its initial state unless the user changes `pg.rhos`.
"""
function (::FixedDirichlet)(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, tag::Int, ts, eq::HyperbolicPDE, t::Real) where {D, M, T}
    foreach_particle(_dirichlet_body!, get_backend(), pg.meta.N, (rhos_buffer, pg.rhos, pg.core.tags, pg.core.is_boundary, tag))
    return nothing
end

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
- Safely falls back to the state in `pg.rhos` at the start of the step for any completely orphaned boundary particles that failed to resolve a donor.
"""
function (::OutflowBC)(pg::ParticleGrid{D, M, T}, rhos_buffer::AbstractVector{State{M, T}}, tag::Int, ts, eq::HyperbolicPDE, t::Real) where {D, M, T}
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

    foreach_particle(_outflow_fallback!, backend, N, (rhos_buffer, pg.rhos, status))
    return nothing
end

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
# =========================================================================
# Ghost states from the interior: one-sided fits at the nearest interior particle
# =========================================================================
export ExtrapolationBC, MirrorBC, WallPoint, requires_shape

"""
    WallPoint{D, T}

The wall as seen from a ghost particle, handed to the operator of a [`MirrorBC`](@ref): `position` (the wall
point nearest to the ghost), the outward unit `normal` there, the `distance` of the ghost to the wall, and the
wall `velocity` (zero for the static walls of this version; moving boundaries will fill it).
"""
struct WallPoint{D, T}
    position::Space{D, T}
    normal::Space{D, T}
    distance::T
    velocity::Space{D, T}
end

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
[`wall_geometry`](@ref)) and stores `op(eq, U, wall::WallPoint)` of the value `U` found there. The mirror point
lies inside the cloud, so the fit interpolates. Without `op`
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
curved wall the mirror image is second-order accurate. Behind a convex corner (outside two faces) the nearest
wall point is the corner: the normal is the diagonal direction to it and `op` is applied once with it, so an
operator that reflects the normal velocity reflects the diagonal component only there (`(u, v) → (-v, -u)` at a
box corner, not `(-u, -v)`), and only the condition of the face whose tag the ghost carries acts. Such ghosts
are in the stencils of the corner particles only. Requires a shape-based domain ([`requires_shape`](@ref)).
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
    _launch_ghost_fit!(pg, rhos_buffer, tag, eq, Val(Q), nothing, Val(false), Val(false))
    return nothing
end
function (bc::MirrorBC{Q, F, B})(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t::Real) where {Q, F, B}
    _launch_ghost_fit!(pg, rhos_buffer, tag, eq, Val(Q), _WallOp(bc.op), Val(true), Val(B))
    return nothing
end

# In 3-D the degree is capped at 3 (20 unknowns, the limit of the package's other 3-D fits): degrees 4-5 (35/56
# unknowns) would take StaticArrays' allocating, host-only Cholesky inside the kernel
_ghost_degree(::Val{Q}, ::Val{D}) where {Q, D} = Val(D == 3 ? min(Q, 3) : Q)

function _launch_ghost_fit!(pg::ParticleGrid{D, M, T}, buf, tag::Int, eq, valQ::Val, op, mirror::Val, bounded::Val) where {D, M, T}
    valQ = _ghost_degree(valQ, Val(D))
    nb   = pg.neighbor
    invL = one(T) / minimum(pg.meta.dx)
    foreach_particle(_ghost_fit_body!, get_backend(), pg.meta.N,
        (buf, pg.core.positions, pg.core.is_boundary, pg.core.tags, pg.core.normals, pg.core.wall_distances,
         pg.core.particle_orders, nb.ranges, nb.indices, nb.weight_func,
         pg.domain.L_wrap, pg.domain.invL_wrap, invL, tag, valQ, op, eq, mirror, bounded, Val(D)))
    return nothing
end

# The operator enters the kernel inside a struct: a plain `Function` argument makes the Polyester loop allocate
# (368 bytes per launch), a struct holding it does not.
struct _WallOp{F}
    op::F
end
@inline _apply_wall_op(::_WallOp{Nothing}, eq, U, wall) = U
@inline _apply_wall_op(w::_WallOp, eq, U, wall) = w.op(eq, U, wall)

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

# Kernel body (see Backend.jl): the ghost state of particle `i` from an MLS fit centred at its target point.
@inline function _ghost_fit_body!(i, buf, pos, isb, tags, normals, wdist, orders, ranges, indices, wf,
                                  L, Li, invL, tag, ::Val{Q}, op, eq, ::Val{MIRROR}, ::Val{BOUNDED}, ::Val{D}) where {Q, MIRROR, BOUNDED, D}
    @inbounds begin
        (isb[i] && tags[i] == tag) || return nothing
        x_i = pos[i]
        n_i = normals[i]
        φ_i = wdist[i]
        x_t = MIRROR ? x_i - 2φ_i * n_i : x_i
        # Donor: the interior particle nearest to the target point. Its neighbour list is the stencil of the fit
        # (every interior particle within R - |x_t - x_j| of the target is in it). Searched in the ghost's own
        # neighbour list, then in its neighbours' lists (the outermost ghosts of the R + dx band have no interior
        # neighbour), then refined by walking the neighbour graph towards the target.
        j, best = _nearest_interior(i, x_t, ranges, indices, isb, pos, L, Li)
        if j == 0
            for k in ranges[i]
                j2, best2 = _nearest_interior(indices[k], x_t, ranges, indices, isb, pos, L, Li)
                (j2 != 0 && best2 < best) && (j = j2; best = best2)
            end
        end
        j == 0 && return nothing
        for _ in 1:3
            j2, best2 = _nearest_interior(j, x_t, ranges, indices, isb, pos, L, Li)
            (j2 != 0 && best2 < best) || break
            j, best = j2, best2
        end
        # Fit degree and donor: the requested degree, at most the donor's order - 1 (0 = order never set), lowered
        # one step at a time until the donor or, failing that, its interior neighbour nearest to the target carries
        # the degree with the margin. A donor at a wall or corner has too few interior neighbours for a high degree
        # (3-4 for the 6 unknowns of a 2-D quadratic) while its neighbours further inside do; the fit stays exact for
        # polynomials whatever the distance. (Trying the requested degree only and otherwise falling back to what the
        # wall donor supports made a higher request give a lower degree: 1-D, R = 3.5Δx, degree 4 ran as degree 1
        # while degree 3 ran as 3.)
        q = Q
        o = orders[j]
        o > 0 && (q = min(q, o - 1))
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
        U = _fit_eval(Val(Q), q, j, x_t, buf, ranges[j], indices, pos, isb, L, Li, wf, invL)
        if BOUNDED
            # Discrete maximum principle on the ghost state: the mirrored value stays within the range of the data
            # it was built from. Without it the feedback ghost -> near-wall stencils -> ghost grew without bound for
            # degrees >= 2 with a 5th-order scheme (long runs along a wall), with it every tested case stayed bounded.
            lo = buf[j]
            hi = buf[j]
            for k in ranges[j]
                m = indices[k]
                isb[m] && continue
                lo = min.(lo, buf[m])
                hi = max.(hi, buf[m])
            end
            U = clamp.(U, lo, hi)
        end
        if MIRROR
            wall   = WallPoint(x_i - φ_i * n_i, n_i, φ_i, zero(x_i))
            buf[i] = _apply_wall_op(op, eq, U, wall)
        else
            buf[i] = U
        end
    end
    return nothing
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
    N_s, b_s = _ghost_moments(Val(B), Val(q), j, x_t, buf, rg, indices, pos, isb, L, Li, wf, invL)
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
    return N_s, b_s
end
