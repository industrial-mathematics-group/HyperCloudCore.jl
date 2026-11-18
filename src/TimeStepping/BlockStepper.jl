export BlockState, BlockTimeStepper, BlockOf, num_blocks, block_flux, block_max_eigenvalue, block_admissible, block_view, set_block_states!,
       block_velocity, block_is_admissible, block_nc_jump
export AbstractBlockSource, AbstractExplicitBlockSource, AbstractImplicitBlockSource, BlockSourceTerm
export reduction_type, block_reduce, block_source, block_implicit_solve
export reduction_partial_type, block_partial, finalize_reduction
export block_fluid_velocity, wall_reduction_type, block_wall_reduce, block_wall_apply
export reorder!

# =========================================================================
# BLOCK EQUATIONS
# =========================================================================
"""
    num_blocks(eq::HyperbolicPDE) -> Int
    block_flux(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, b::Int) -> Flux{D, W, T}
    block_max_eigenvalue(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, d::Int, b::Int) -> T

The contract of a **block equation**: a system of `nb · W` components whose transport is block diagonal, so
that block `b` (components `(b − 1) W + 1 … b W`, a `State{W, T}`) has a flux and wave speeds of its own that
depend on that block only. The components couple through source terms and through per-particle reductions
only (discrete-velocity models: every velocity is transported on its own, the collision couples them through
the moments). Such an equation is a `HyperbolicPDE{D, W, T}` (its state type is the block) that implements
`num_blocks`, `block_flux` and `block_max_eigenvalue`; the [`BlockTimeStepper`](@ref) runs the schemes of the
package on every block with the same grid, stencils and weights.

Every `HyperbolicPDE` is a block equation with one block: the defaults are `num_blocks = 1`,
`block_flux = flux` and `block_max_eigenvalue = max_eigenvalue`, so the block pipeline reproduces the standard
one for any equation.
"""
num_blocks(::HyperbolicPDE) = 1

"""
    block_flux(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, b::Int) -> Flux{D, W, T}

Physical flux of block `b` of a block equation (see [`num_blocks`](@ref)); the default is `flux(eq, U)`.
"""
@inline block_flux(eq::HyperbolicPDE, U::State, b::Int) = flux(eq, U)

"""
    block_max_eigenvalue(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, d::Int, b::Int) -> T

Largest absolute wave speed of block `b` along axis `d` (see [`num_blocks`](@ref)); the default is
`max_eigenvalue(eq, U, d)`.
"""
@inline block_max_eigenvalue(eq::HyperbolicPDE, U::State, d::Int, b::Int) = max_eigenvalue(eq, U, d)

"""
    block_max_eigenvalue(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, d::Int, b::Int, shift::Real) -> T

The largest `|λ − shift|` of block `b` along axis `d`: the wave speed bound in the frame moving at `shift` (a moving
grid, `ALEFrame`). The default `block_max_eigenvalue(eq, U, d, b) + |shift|` is a safe bound; a block equation whose
spectrum is known (a discrete-velocity model: its velocities) gives the exact value, so that the numerical
dissipation of a Lagrangian frame vanishes where it should.
"""
@inline block_max_eigenvalue(eq::HyperbolicPDE, U::State, d::Int, b::Int, shift::Real) = block_max_eigenvalue(eq, U, d, b) + abs(shift)

"""
    block_velocity(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, d::Int, b::Int) -> SMatrix{W, W, T}
    block_is_admissible(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, b::Int) -> Bool
    block_nc_jump(eq::HyperbolicPDE{D, W, T}, f_L::Flux, f_R::Flux, dist::Space, b::Int) -> Flux{D, W, T}

The rest of the equation contract per block, which the schemes, numerical fluxes and MOOD call on the
[`BlockOf`](@ref) wrapper: the advection matrix (`UpwindFlux`, the upwind divergences, WENO), the physical
admissibility of the block's candidate (MOOD's PAD per block; [`block_admissible`](@ref) adds the check across the
blocks) and the non-conservative jump. The defaults are the equation's own `velocity(eq, U, d)`,
`is_admissible(eq, U)` and `evaluate_nc_jump(eq, f_L, f_R, dist)`: exact for an ordinary equation run as one block
(the block is the whole state), and `is_admissible` defaults to "every component finite" for any equation. A
block equation with several blocks implements them per block where its schemes need them.
"""
@inline block_velocity(eq::HyperbolicPDE, U::State, d::Int, b::Int) = velocity(eq, U, d)

"""
    block_is_admissible(eq::HyperbolicPDE{D, W, T}, U::State{W, T}, b::Int) -> Bool

Physical admissibility of the candidate state of block `b` (MOOD PAD per block, see [`block_velocity`](@ref) for the
family); the default is `is_admissible(eq, U)`.
"""
@inline block_is_admissible(eq::HyperbolicPDE, U::State, b::Int) = is_admissible(eq, U)

"""
    block_nc_jump(eq::HyperbolicPDE{D, W, T}, f_L::Flux, f_R::Flux, dist::Space, b::Int) -> Flux{D, W, T}

Non-conservative jump of block `b` (see [`block_velocity`](@ref)); the default is `evaluate_nc_jump(eq, f_L, f_R, dist)`.
"""
@inline block_nc_jump(eq::HyperbolicPDE, f_L, f_R, dist, b::Int) = evaluate_nc_jump(eq, f_L, f_R, dist)

"""
    block_admissible(eq::HyperbolicPDE, Y, N, nb, p) -> Bool

Physical admissibility of particle `p` of a block equation across its blocks (the PAD of MOOD in the block
pipeline; block `b` of the particle is `Y[(b - 1) N + p]`): the default `true` leaves the per-block check (every
block finite, `is_admissible` of the `BlockOf` wrapper) as the only one; a discrete-velocity model adds the
positivity of the moments here. Runs inside kernels.
"""
@inline block_admissible(eq::HyperbolicPDE, Y, N::Int, nb::Int, p::Int) = true

"""
    block_fluid_velocity(eq::HyperbolicPDE{D, W, T}, Y, N, nb, p) -> Space{D, T}

The fluid velocity of particle `p` of a block equation across its blocks (block `b` at `Y[(b - 1) N + p]`): what a
`LagrangianMover` or `ScaledLagrangianMover` moves the particle with in the block pipeline (a discrete-velocity
model: the mean velocity of the moments). Runs inside kernels. No default.
"""
block_fluid_velocity(eq::HyperbolicPDE, Y, N::Int, nb::Int, p::Int) =
    error("`block_fluid_velocity` not implemented for $(typeof(eq)) (needed by a Lagrangian mover in the block pipeline)")

"""
    wall_reduction_type(op, eq) -> Type or nothing
    block_wall_reduce(op, eq, Y, N, nb, g, wall::WallPoint) -> params
    block_wall_apply(op, eq, U::State{W}, b, params, wall::WallPoint, Y, N, nb, g) -> State{W}

The operator of a `MirrorBC` in the block pipeline, in the two passes of the pipeline: the mirrored interior state
of every block is fitted into a scratch array first (the even reflection); `block_wall_reduce` then turns the
mirrored blocks of ghost `g` (block `b` at `Y[(b - 1) N + g]`) into a bits value of parameters (a kinetic wall: the
density that returns the incident flux, summed over all velocities), and `block_wall_apply` gives the ghost state
of block `b` from its mirrored value `U`, the parameters and, if it needs other blocks of the ghost (a velocity
reflected into another block), the scratch array. `wall_reduction_type(op, eq)` is the parameter type; its default
`nothing` means that the operator acts block by block, `op(BlockOf(eq, b), U, wall)`, like a per-component
reflection. Kernel functions, as the operators themselves.
"""
wall_reduction_type(op, eq) = nothing
"The reduction of a wall operator over the mirrored blocks of ghost `g` (see [`wall_reduction_type`](@ref))."
block_wall_reduce(op, eq, Y, N, nb, g, wall) = error("`block_wall_reduce` not implemented for $(typeof(op)) on $(typeof(eq))")
"The ghost state of block `b` from its mirrored value and the wall parameters (see [`wall_reduction_type`](@ref))."
block_wall_apply(op, eq, U, b, params, wall, Y, N, nb, g) = error("`block_wall_apply` not implemented for $(typeof(op)) on $(typeof(eq))")

"""
    BlockOf(eq, b) <: HyperbolicPDE{D, W, T}

Block `b` of a block equation as an ordinary PDE of `W` components: `flux` is `block_flux(eq, ·, b)` and
`max_eigenvalue` is `block_max_eigenvalue(eq, ·, d, b)`, so the schemes and numerical fluxes run on the block
unchanged. The block index is a field (a run-time value): one compiled kernel serves every block.
"""
struct BlockOf{D, W, T, EQ <: HyperbolicPDE{D, W, T}} <: HyperbolicPDE{D, W, T}
    eq::EQ
    b::Int
end

@inline flux(w::BlockOf, U::State) = block_flux(w.eq, U, w.b)
@inline max_eigenvalue(w::BlockOf, U::State, d::Int) = block_max_eigenvalue(w.eq, U, d, w.b)
@inline max_eigenvalue(w::BlockOf, U::State, d::Int, shift::Real) = block_max_eigenvalue(w.eq, U, d, w.b, shift)
# (a missing forward silently fell back on the generic defaults: an admissibility or a non-conservative jump of the
# equation was ignored on the block form, a velocity-based scheme failed)
@inline velocity(w::BlockOf, U::State, d::Int) = block_velocity(w.eq, U, d, w.b)
@inline is_admissible(w::BlockOf, U::State) = block_is_admissible(w.eq, U, w.b)
@inline evaluate_nc_jump(w::BlockOf{D, W, T}, f_L::Flux{D, W, T}, f_R::Flux{D, W, T}, dist::Space{D, T}) where {D, W, T} =
    block_nc_jump(w.eq, f_L, f_R, dist, w.b)

# =========================================================================
# BLOCK SOURCE TERMS: a reduction over the blocks, then the update per block
# =========================================================================
"""
    AbstractBlockSource
    AbstractExplicitBlockSource <: AbstractBlockSource
    AbstractImplicitBlockSource <: AbstractBlockSource

A source term of a block equation acts in **two passes**: a reduction over all blocks of a particle into a
small bits value of per-particle parameters (for a discrete-velocity model the moments and what follows from
them, e.g. the exponential parameters of the Maxwellians and the stiffness), then the update of every block
from those parameters. The contract, all kernel functions:

- `reduction_type(st) -> Type{P}`: the (bits) type of the parameters.
- `block_reduce(st, p, Y, N, nb, pg, t, dt_coeff) -> P`: the parameters of particle `p` from the full state
  array `Y` (block `b` of particle `p` at `Y[(b - 1) N + p]`, `nb` blocks). For an implicit term `dt_coeff`
  is `Δt a_ii` of the stage (the parameters of the solved stage, e.g. the Maxwellian of the moments after the
  collision); it is `0` when the term is evaluated explicitly (the parameters of the current state).
- `block_source(st, U::State{W}, b, params::P, p, pg, t) -> State{W}`: the source of block `b` (explicit
  terms; implicit terms implement it too, for the stages of an IMEX tableau without an implicit diagonal).
- `block_implicit_solve(st, U::State{W}, b, params::P, dt_coeff, p, pg, t) -> State{W}`: the solution of the
  implicit stage equation `U_out = U + dt_coeff S(U_out)` for block `b` (implicit terms).

A term is wrapped in a [`BlockSourceTerm`](@ref), which owns the parameter array, and handed to the
[`BlockTimeStepper`](@ref). Boundary particles get no source and no parameters.
"""
abstract type AbstractBlockSource end
"An explicit block source: `block_reduce` and `block_source` (see [`AbstractBlockSource`](@ref))."
abstract type AbstractExplicitBlockSource <: AbstractBlockSource end
"An implicit block source: `block_reduce`, `block_implicit_solve` and `block_source` (see [`AbstractBlockSource`](@ref))."
abstract type AbstractImplicitBlockSource <: AbstractBlockSource end

"The parameter type of a block source's reduction (see [`AbstractBlockSource`](@ref))."
reduction_type(st::AbstractBlockSource) = error("`reduction_type` not implemented for $(typeof(st))")
"The source of block `b` from the reduced parameters (see [`AbstractBlockSource`](@ref))."
block_source(st::AbstractBlockSource, U, b, params, p, pg, t) = error("`block_source` not implemented for $(typeof(st))")
"The implicit stage solution of block `b` from the reduced parameters (see [`AbstractBlockSource`](@ref))."
block_implicit_solve(st::AbstractImplicitBlockSource, U, b, params, dt_coeff, p, pg, t) =
    error("`block_implicit_solve` not implemented for $(typeof(st))")

"""
    reduction_partial_type(st) -> Type or nothing
    block_partial(st, p, Y, N, nb, blocks::UnitRange, pg, t) -> partial
    finalize_reduction(st, p, partial, pg, t, dt_coeff) -> params

The **additive form** of a block source's reduction, for a state whose blocks are split over several devices: the
contribution of the blocks `blocks` of particle `p` is a bits value that adds up over block ranges (`+`; for a
discrete-velocity model the moments of those velocities), and `finalize_reduction` turns the sum over all blocks into
the parameters of [`block_reduce`](@ref) (the Maxwellian, the stiffness, …). A source that gives
`reduction_partial_type` (default `nothing`: the reduction reads all blocks at once) gets `block_reduce` for free,
`finalize_reduction(st, p, block_partial(st, p, Y, N, nb, 1:nb, pg, t), pg, t, dt_coeff)` (the default `block_reduce`). The decomposition of a
block state over devices exchanges these partials only (one per particle and device); the stepper of one device
still runs the whole reduction.
"""
reduction_partial_type(st::AbstractBlockSource) = nothing
"The contribution of the blocks `blocks` of particle `p` to the reduction, additive over block ranges (see [`reduction_partial_type`](@ref))."
block_partial(st::AbstractBlockSource, p, Y, N, nb, blocks::UnitRange, pg, t) = error("`block_partial` not implemented for $(typeof(st))")
"The parameters of [`block_reduce`](@ref) from the sum of the partials over all blocks (see [`reduction_partial_type`](@ref))."
finalize_reduction(st::AbstractBlockSource, p, partial, pg, t, dt_coeff) = error("`finalize_reduction` not implemented for $(typeof(st))")

"The reduction of a block source over the blocks of particle `p` (see [`AbstractBlockSource`](@ref)); by default the additive form over all blocks (`finalize_reduction` of `block_partial` over `1:nb`) when the source gives one."
@inline block_reduce(st::AbstractBlockSource, p, Y, N, nb, pg, t, dt_coeff) = _block_reduce(reduction_partial_type(st), st, p, Y, N, nb, pg, t, dt_coeff)
@inline _block_reduce(::Nothing, st, p, Y, N, nb, pg, t, dt_coeff) = error("`block_reduce` not implemented for $(typeof(st))")
@inline _block_reduce(::Type, st, p, Y, N, nb, pg, t, dt_coeff) = finalize_reduction(st, p, block_partial(st, p, Y, N, nb, 1:nb, pg, t), pg, t, dt_coeff)

"""
    BlockSourceTerm(st::AbstractBlockSource; storage = Array)

A block source with its per-particle parameter array (a `Growable` of `reduction_type(st)` with arrays of type
`storage`), what the [`BlockTimeStepper`](@ref) takes.
"""
struct BlockSourceTerm{S <: AbstractBlockSource, P, G <: Growable{P}}
    term::S
    params::G
end
function BlockSourceTerm(st::AbstractBlockSource; storage::Type{AT} = Array) where {AT <: AbstractArray}
    P = reduction_type(st)
    isbitstype(P) || throw(ArgumentError("the reduction type of $(nameof(typeof(st))) must be a bits type, got $P"))
    g = Growable{P}(AT)
    return BlockSourceTerm{typeof(st), P, typeof(g)}(st, g)
end

_is_implicit(::BlockSourceTerm{<:AbstractImplicitBlockSource}) = true
_is_implicit(::BlockSourceTerm) = false

# The reduction pass of a source over the full stage array (block-major, stride N)
function reduce_blocks!(bst::BlockSourceTerm, Y::AbstractVector, N::Int, nb::Int, pg, t, dt_coeff)
    ensure_capacity!(bst.params, N)
    foreach_particle(_block_reduce_body!, get_backend(), N, (bst.params, bst.term, Y, N, nb, pg, pg.core.is_boundary, t, dt_coeff))
    return nothing
end
@inline function _block_reduce_body!(p, params, st, Y, N, nb, pg, is_boundary, t, dt_coeff)
    is_boundary[p] && return nothing
    params[p] = block_reduce(st, p, Y, N, nb, pg, t, dt_coeff)
    return nothing
end

# Kernel bodies of the per-block passes (see Backend.jl): the explicit source added to the right-hand side, the
# implicit stage solve with its right-hand side `(U_out − U_in) / (Δt a_ii)` (the increments of several implicit
# sources, applied one after another, add up), and the explicit evaluation of an implicit source where the stage has
# no implicit diagonal
@inline function _block_explicit_source!(p, K_E, Y, params, st, b, is_boundary, pg, t, full, dirty)
    (is_boundary[p] || !(full || dirty[p])) && return nothing
    K_E[p] += block_source(st, Y[p], b, params[p], p, pg, t)
    return nothing
end

# The cross-block admissibility check of MOOD (flags into `triggered`) and the count of inadmissible candidates
@inline function _block_admissible_flag!(p, triggered, is_boundary, dirty, full, eq, Y, N, nb)
    (is_boundary[p] || !(full || dirty[p])) && return nothing
    block_admissible(eq, Y, N, nb, p) || (triggered[p] = true)
    return nothing
end
@inline function _block_inadmissible_body!(p, flags, is_boundary, eq, Y, N, nb)
    if is_boundary[p]
        flags[p] = false
        return nothing
    end
    ok = true
    for b in 1:nb                                    # (the check of the flag pass, not only finiteness)
        ok &= block_is_admissible(eq, Y[(b - 1) * N + p], b)
    end
    flags[p] = !(ok && block_admissible(eq, Y, N, nb, p))
    return nothing
end
@inline function _block_implicit_step!(p, Y, K_I, params, st, b, is_boundary, dt_ii, inv_dt_ii, pg, t, first)
    if is_boundary[p]
        first && (K_I[p] = zero(Y[p]))
        return nothing
    end
    U_in  = Y[p]
    U_out = block_implicit_solve(st, U_in, b, params[p], dt_ii, p, pg, t)
    Y[p]  = U_out
    inc   = (U_out - U_in) * inv_dt_ii
    K_I[p] = first ? inc : K_I[p] + inc
    return nothing
end
@inline function _block_implicit_rhs!(p, K_I, Y, params, st, b, is_boundary, pg, t, first)
    val = is_boundary[p] ? zero(Y[p]) : block_source(st, Y[p], b, params[p], p, pg, t)
    K_I[p] = first ? val : K_I[p] + val
    return nothing
end

# =========================================================================
# MOVING GRIDS: the frame velocity of every particle from a reduction, the flux pass in the ALE frame
# =========================================================================
# The mover the flux pass sees: the velocities reduced before the pass (one per particle), whatever the mover. The
# kernels then use the frozen-velocity ALE frame of the particle (`ALEFrame(eq, w_i)`); the conservative Lagrangian frame
# of the standard stepper evaluates the frame velocity of every interface state, which a block cannot provide.
struct _BlockMover{GM <: GridMover, V} <: GridMover
    mover::GM
    w::V
end
@inline grid_velocity(bm::_BlockMover, eq, p_idx, U, pg, t) = @inbounds bm.w[p_idx]
@inline _lagrangian_type(::_BlockMover) = false

# The frame velocity of particle `p` from the full state array: state-independent movers as in the standard stepper,
# the Lagrangian ones through `block_fluid_velocity`
@inline _block_grid_velocity(mover::GridMover, eq, Y, N, nb, p, pg, t) = grid_velocity(mover, eq, p, Y[p], pg, t)
@inline _block_grid_velocity(::LagrangianMover, eq, Y, N, nb, p, pg, t) = block_fluid_velocity(eq, Y, N, nb, p)
@inline _block_grid_velocity(m::ScaledLagrangianMover, eq, Y, N, nb, p, pg, t) = m.alpha * block_fluid_velocity(eq, Y, N, nb, p)

@inline function _block_velocity_body!(p, w, mover, eq, Y, N, nb, pg, is_boundary, t, zero_space)
    w[p] = is_boundary[p] ? zero_space : _block_grid_velocity(mover, eq, Y, N, nb, p, pg, t)
    return nothing
end

# Kernel bodies of the block wall (see Backend.jl): the reduction over the mirrored blocks of a ghost and the application
@inline function _block_wall_reduce_body!(g, params, op, eq, S, N, nb, pos, is_boundary, tags, normals, wdist, curvs, gm, tag)
    (is_boundary[g] && tags[g] == tag) || return nothing
    params[g] = block_wall_reduce(op, eq, S, N, nb, g, _ghost_wall_point(g, pos, normals, wdist, curvs, gm))
    return nothing
end
@inline function _block_wall_apply_body!(g, Y_b, S_b, params, op, eq, b, S, N, nb, pos, is_boundary, tags, normals, wdist, curvs, gm, tag)
    (is_boundary[g] && tags[g] == tag) || return nothing
    Y_b[g] = block_wall_apply(op, eq, S_b[g], b, params[g], _ghost_wall_point(g, pos, normals, wdist, curvs, gm), S, N, nb, g)
    return nothing
end

# =========================================================================
# THE BLOCK STATE
# =========================================================================
"""
    BlockState(eq, pg; storage = Array)
    block_view(bs, b) -> AbstractVector{State{W, T}}

The state of a block equation over a particle grid: `nb = num_blocks(eq)` blocks of `State{W, T}`, one array of
length `nb · N` in block-major order (block `b` of particle `i` at index `(b − 1) N + i`), so that every block is
a contiguous vector over the particles. `block_view(bs, b)` is that vector (a view); [`set_block_states!`](@ref)
fills it from a function of the position. The array is a `Growable` with arrays of type `storage`.
"""
struct BlockState{W, T, G <: Growable{State{W, T}}}
    data::G
    nb::Int
    N::Base.RefValue{Int}
end

function BlockState(eq::HyperbolicPDE{D, W, T}, pg::ParticleGrid{D, W, T}; storage::Type{AT} = Array) where {D, W, T, AT <: AbstractArray}
    nb = num_blocks(eq)
    nb >= 1 || throw(ArgumentError("num_blocks($(nameof(typeof(eq)))) = $nb"))
    data = Growable{State{W, T}}(AT)
    ensure_capacity!(data, nb * pg.meta.N)
    return BlockState{W, T, typeof(data)}(data, nb, Ref(pg.meta.N))
end

"""
    block_view(bs::BlockState, b::Int) -> AbstractVector{State{W, T}}

The states of block `b` over the particles (a view of the block state's array, see [`BlockState`](@ref)).
"""
@inline function block_view(bs::BlockState, b::Int)
    N = bs.N[]
    1 <= b <= bs.nb || throw(BoundsError(bs, b))
    return view(parent(bs.data), ((b - 1) * N + 1):(b * N))
end

# The host routines that change the particle set (adaptivity, the sweep of moving pieces) work on state arrays that
# can grow; the block-major array cannot (its layout depends on N). They get per-block copies and the blocks are put
# back for the particle count the call leaves.
function _block_vectors(bs::BlockState{W, T}) where {W, T}
    return [Vector{State{W, T}}(Array(block_view(bs, b))) for b in 1:bs.nb]  # a device array is copied to the host
end
function _set_block_vectors!(bs::BlockState{W, T}, vecs, N::Int) where {W, T}
    bs.N[] = N
    ensure_capacity!(bs.data, bs.nb * N)
    for b in 1:bs.nb
        copyto!(parent(bs.data), (b - 1) * N + 1, vecs[b], 1, N)
    end
    return nothing
end
# The copies are made on first use (`_LazyStates`): a sweep or an adaptivity call without events copies nothing (on a
# device every copy is a transfer of the whole state)
function _with_block_vectors(f, bs::BlockState, pg::ParticleGrid)
    lazy = _LazyStates(() -> _block_vectors(bs))
    out  = f(lazy)
    if lazy.vecs !== nothing
        _set_block_vectors!(bs, lazy.vecs, pg.meta.N)
    else
        bs.N[] == pg.meta.N || error("internal: the particle set changed without the block states being remapped")
    end
    return out
end

"""
    adapt_particles!(pg, ad::ParticleAdaptivity, max_order, bs::BlockState) -> (; inserted, removed, mass_residual)

The particle adaptivity of the block pipeline: every block of `bs` is remapped, compacted and extended like the
state of the standard pipeline (`adapt_particles!` with the blocks as extra state arrays), and the block state is
laid out again for the new particle count. Nothing is copied when no pair of particles violates the spacing rules.
`mass_residual` refers to the grid's own (unused) state; compare `block_moments` before and after instead.
"""
function adapt_particles!(pg::ParticleGrid{D, W, T}, ad::ParticleAdaptivity{T}, max_order::Int, bs::BlockState{W, T}) where {D, W, T}
    bs.N[] == pg.meta.N || throw(ArgumentError("adapt_particles!: the block state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    (pg.meta.N > 0 && _has_adapt_candidates(pg, ad)) || return (inserted = 0, removed = 0, mass_residual = zero(State{W, T}))
    return _with_block_vectors(lazy -> adapt_particles!(pg, ad, max_order; states = lazy), bs, pg)
end

"""
    reorder!(pg::ParticleGrid, bs::BlockState)

The spatial reordering of the grid (`pg.reorder`) with the blocks of `bs` permuted along. The neighbour lists are
stale afterwards: run `pg.neighbor(pg)` before the next step.
"""
function reorder!(pg::ParticleGrid{D, W, T}, bs::BlockState{W, T}) where {D, W, T}
    bs.N[] == pg.meta.N || throw(ArgumentError("reorder!: the block state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    pg.reorder(pg; extra = [block_view(bs, b) for b in 1:bs.nb])
    return nothing
end

"""
    set_block_states!(f, bs::BlockState, pg::ParticleGrid)

Sets the state of every particle and block to `f(x, b)::State{W, T}` with the particle's position `x` (a host loop
through a host copy of the positions).
"""
function set_block_states!(f, bs::BlockState{W, T}, pg::ParticleGrid{D, W, T}) where {D, W, T}
    N = pg.meta.N
    bs.N[] = N
    ensure_capacity!(bs.data, bs.nb * N)
    xs  = Array(view(pg.core.positions, 1:N))
    buf = Vector{State{W, T}}(undef, bs.nb * N)
    for b in 1:bs.nb, i in 1:N
        buf[(b - 1) * N + i] = f(xs[i], b)
    end
    copyto!(parent(bs.data), 1, buf, 1, bs.nb * N)
    return nothing
end

# =========================================================================
# THE BLOCK TIME STEPPER
# =========================================================================
"""
    BlockTimeStepper(eq, div_interp, mood, sources::Tuple, tableau; storage = Array, kwargs...)
    BlockTimeStepper(eq, div_interp, sources::Tuple, tableau; kwargs...)
    BlockTimeStepper(eq, div_interp, tableau; kwargs...)
    (ts::BlockTimeStepper)(eq, pg, bs::BlockState, time, dt)

The time integrator of the block pipeline: the stage kernels of [`GeneralRKTimeStepper`](@ref) run block by
block on the views of a [`BlockState`](@ref), with the equation wrapped as [`BlockOf`](@ref)`(eq, b)`, so the
grid, the neighbour search, the MLS weights, the scheme and its pre-pass are shared by all blocks and the
state length `nb · W` is a run-time number. The stepper owns a `GeneralRKTimeStepper` for `W` components
(`inner`: the interpolator, MOOD scratch, buffers and options; its stage buffers are sized to `nb · N`), and the
`kwargs` are its options (`mood_substeps`, `reset_orders_per_stage`, `storage`, …).

`sources` are [`BlockSourceTerm`](@ref)s. Explicit ones enter the right-hand side of every stage after a
reduction over the blocks at the stage state; implicit ones are solved at the stages with an implicit diagonal
(`a_ii ≠ 0`: the reduction with `dt_coeff = Δt a_ii`, then the solve per block, the right-hand side
`K_I = (U_out − U_in) / (Δt a_ii)`; several implicit terms one after another, as in the standard stepper) and
evaluated explicitly at the others; they need an IMEX tableau. The explicit part of the tableau may be
any explicit RK method.

`mood` works as in the standard stepper (the stage-wise fixed point, the Euler substeps of SSP tableaus, the
forcing halos), with the particle's order shared by all its blocks: the criterion (`MOODu1`, `MOODu2`, `OnlyMOOD`,
`MOODPAD`) checks every block's components, the per-block admissibility is [`block_is_admissible`](@ref), and
[`block_admissible`](@ref) adds the cross-block check (e.g. positive moments); a flagged particle drops one
order, and the previous stage's right-hand side is recomputed on the affected particles for all blocks.
`MOODu2`'s curvature test reads the scheme's gradient cache, so the block's gradients at the previous stage are
gathered before its flag pass (one pre-pass per block and checkpoint).

On a moving grid (`pg.mover`) the frame velocity of every particle is reduced before each flux pass
([`block_fluid_velocity`](@ref) for the Lagrangian movers, the mover's velocity otherwise) and the pass runs in the
frozen-velocity ALE frame of the particle; the positions are stage variables as in the standard stepper, the
volumes are re-estimated at the end of the step. (The conservative Lagrangian frame of the standard stepper needs
the frame velocity of every interface state, which a block cannot give.) Moving boundary pieces are placed and
swept as in the standard stepper, the sweep remapping every block (`sweep_boundaries!` with the blocks as state
arrays); [`adapt_particles!`](@ref)`(pg, ad, order, bs)` is the particle adaptivity between steps and
[`reorder!`](@ref)`(pg, bs)` the spatial reordering. The boundary conditions of the grid apply per block with the
block's state of the step start as the reference; a `MirrorBC` whose operator reduces over the blocks
([`block_wall_reduce`](@ref)) runs in the pipeline's passes. For one block (`num_blocks(eq) == 1`) the
result is bit-identical to `GeneralRKTimeStepper` without sources, and agrees to round-off with the equivalent
sources (a separately compiled flux pass).
"""
struct BlockTimeStepper{D, W, T, EQ <: HyperbolicPDE{D, W, T}, TS <: GeneralRKTimeStepper{D, W, T},
                        EST <: Tuple{Vararg{BlockSourceTerm}}, IST <: Tuple{Vararg{BlockSourceTerm}},
                        GW <: Growable{Space{D, T}}, GS <: Growable{State{W, T}}} <: TimeStepper
    pde::EQ
    inner::TS
    explicit_sources::EST
    implicit_sources::IST
    grid_w::GW                      # the frame velocities of a moving grid (per particle)
    ghost_scratch::GS               # the mirrored blocks of the wall ghosts (block-major)
    wall_params::Dict{Int, Any}     # the wall parameters per boundary tag (a Growable of `wall_reduction_type`)
end

"""
    BlockTimeStepper(eq, inner::GeneralRKTimeStepper, sources::Tuple)

The block stepper around an existing standard stepper for `W` components (built for `BlockOf(eq, 1)`, e.g. by
`build_timestepper`: all of its options, the memory reservation included); `inner` must carry no sources of its own.
"""
function BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, inner::GeneralRKTimeStepper{D, W, T}, sources::Tuple{Vararg{BlockSourceTerm}}) where {D, W, T}
    inner.pde isa BlockOf && inner.pde.eq === eq ||
        throw(ArgumentError("BlockTimeStepper: the inner stepper must be built for BlockOf(eq, 1) of this equation"))
    (isempty(inner.explicit_sources) && isempty(inner.implicit_sources)) ||
        throw(ArgumentError("BlockTimeStepper: the inner stepper must carry no sources; give block sources to the block stepper"))
    explicit_sts = filter(st -> !_is_implicit(st), sources)
    implicit_sts = filter(_is_implicit, sources)
    s = num_stages(inner.tableau)
    if !isempty(implicit_sts) && all(i -> iszero(inner.tableau.a[i, i]), 1:s)
        throw(ArgumentError("BlockTimeStepper: implicit block sources need an IMEX tableau with an implicit diagonal " *
                            "(e.g. IMEX_ARS222_Tableau); explicit sources work with any tableau"))
    end
    AT = typeof(parent(inner.rho_n))
    gw = Growable{Space{D, T}}(AT.name.wrapper)
    gs = Growable{State{W, T}}(AT.name.wrapper)
    # the parameter arrays of the sources live where the stage arrays live (a `BlockSourceTerm` built with the default
    # `storage = Array` for a device stepper failed at the first launch: a host array as a kernel argument)
    explicit_sts = map(st -> _on_storage(st, AT.name.wrapper), explicit_sts)
    implicit_sts = map(st -> _on_storage(st, AT.name.wrapper), implicit_sts)
    return BlockTimeStepper{D, W, T, typeof(eq), typeof(inner), typeof(explicit_sts), typeof(implicit_sts), typeof(gw), typeof(gs)}(
        eq, inner, explicit_sts, implicit_sts, gw, gs, Dict{Int, Any}())
end

function BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, div_interp::DivergenceInterpolator, mood::MOOD,
                          sources::Tuple{Vararg{BlockSourceTerm}}, tableau::IMEXButcherTableau{T}; kwargs...) where {D, W, T}
    inner = GeneralRKTimeStepper(BlockOf(eq, 1), div_interp, mood, tableau; kwargs...)
    return BlockTimeStepper(eq, inner, sources)
end
BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, div_interp::DivergenceInterpolator, sources::Tuple{Vararg{BlockSourceTerm}},
                 tableau::IMEXButcherTableau{T}; kwargs...) where {D, W, T} =
    BlockTimeStepper(eq, div_interp, MOOD(NoMOOD()), sources, tableau; kwargs...)
BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, div_interp::DivergenceInterpolator, tableau::IMEXButcherTableau{T}; kwargs...) where {D, W, T} =
    BlockTimeStepper(eq, div_interp, MOOD(NoMOOD()), (), tableau; kwargs...)

_on_storage(bst::BlockSourceTerm, AT) = typeof(parent(bst.params)).name.wrapper === AT ? bst : BlockSourceTerm(bst.term; storage = AT)

@inline _blk(g::Growable, b::Int, N::Int) = view(parent(g), ((b - 1) * N + 1):(b * N))

# The inner stepper's buffers for `nb` blocks of `N` particles: the state arrays block-major (`nb N`), the positions,
# volumes and MOOD masks per particle (`update_size!(inner, nb N, …)` sized those `nb` times too, and a reservation of
# the block stepper failed on them)
function _block_update_size!(inner::GeneralRKTimeStepper, N::Int, nb::Int, M_neighbors::Int, moving::Bool)
    for g in (inner.rho_n, inner.rho_cand, inner.Y_stages..., inner.K_E_stages..., inner.K_I_stages...)
        ensure_capacity!(g, nb * N)
    end
    _checks_substeps(inner) && ensure_capacity!(inner.rho_sub, nb * N)
    if moving
        for g in (inner.x_n, inner.x_cand, inner.v_n, inner.v_cand, inner.X_stages..., inner.K_X_stages..., inner.V_stages..., inner.K_V_stages...)
            ensure_capacity!(g, N)
        end
    end
    uses_slot_buffers(inner.divergence_interpolator) && update_size!(inner.int_buffer, M_neighbors)
    update_size!(inner.work, N)
    return nothing
end

# A `MirrorBC` operator without a block reduction runs on one block's state with the block equation `BlockOf(eq, b)`;
# one written for the equation itself (`op(eq::MyEq, U, wall)`) failed inside the kernel (a MethodError, under
# Polyester a crash of the process)
function _check_block_bcs(pg::ParticleGrid{D, W, T}, eq) where {D, W, T}
    for (tag, bc) in pg.geometry.bc_map
        (bc isa MirrorBC && bc.op !== nothing && wall_reduction_type(bc.op, eq) === nothing) || continue
        hasmethod(bc.op, Tuple{typeof(BlockOf(eq, 1)), State{W, T}, WallPoint{D, T, D * D}}) && continue
        throw(ArgumentError("BlockTimeStepper: the MirrorBC operator of tag $tag has no method for the block equation " *
                            "(op(::BlockOf{…}, ::State{$W}, ::WallPoint)): on blocks it gets one block's state and " *
                            "`BlockOf(eq, b)` (`w.eq`, `w.b`), or it reduces over the blocks (`wall_reduction_type`, " *
                            "`block_wall_reduce`, `block_wall_apply`)"))
    end
    return nothing
end
# The moving-grid path also serves a static cloud whose boundary pieces move (as in the standard stepper)
@inline _block_moving(pg::ParticleGrid) = !(pg.mover isa NoGridMover) || has_moving_pieces(pg.geometry)

"""
    mood_flags(ts::BlockTimeStepper), mood_inadmissible(ts::BlockTimeStepper)

The per-step MOOD diagnostics of the block pipeline (those of the inner stepper, per particle).
"""
mood_flags(ts::BlockTimeStepper) = mood_flags(ts.inner)
mood_inadmissible(ts::BlockTimeStepper) = mood_inadmissible(ts.inner)

function (ts::BlockTimeStepper{D, W, T})(eq::HyperbolicPDE{D, W, T}, pg::ParticleGrid{D, W, T}, bs::BlockState{W, T},
                                         time::Real, dt::T) where {D, W, T}
    inner  = ts.inner
    interp = inner.divergence_interpolator
    max_order = _extract_order(interp)
    nb = bs.nb
    (nb == num_blocks(eq) && nb == num_blocks(ts.pde)) ||
        throw(ArgumentError("BlockTimeStepper: the state has $nb blocks, the equation $(num_blocks(eq)), the stepper's equation $(num_blocks(ts.pde))"))
    bs.N[] == pg.meta.N || throw(ArgumentError("BlockTimeStepper: the state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    _check_block_bcs(pg, eq)
    # moving pieces: the ghosts and the region at the step's time first (the sweep may change the particle set)
    has_moving_pieces(pg.geometry) &&
        _with_block_vectors(v -> _sync_boundaries!(pg, T(time), max_order; states = v), bs, pg)
    N      = pg.meta.N
    s      = inner.num_stages
    bt     = inner.tableau
    orders = pg.core.particle_orders
    moving = _block_moving(pg)

    _block_update_size!(inner, N, nb, length(pg.neighbor.indices), moving)
    update_size!(interp, N)
    moving && ensure_capacity!(ts.grid_w, N)
    copyto!(parent(inner.rho_n), 1, parent(bs.data), 1, nb * N)
    moving && copyto!(parent(inner.x_n), 1, pg.core.positions, 1, N)
    moving && copyto!(parent(inner.v_n), 1, pg.core.volumes, 1, N)

    fill!(view(orders, 1:N), max_order)
    fill!(view(inner.work.triggered_step, 1:N), false)
    inner.work.inadmissible[] = 0

    for i in 1:s
        if i == 1
            _block_form_candidate!(ts, eq, pg, 1, dt, time)
        else
            _block_checkpoint!(ts, eq, pg, i, dt, time)      # uses the geometry of stage i-1
            moving && _apply_stage_geometry!(inner, pg, inner.X_stages[i], time, time + bt.c_t[i] * dt)
        end
        inner.reset_orders_per_stage && fill!(view(orders, 1:N), max_order)
        _block_rhs!(ts, eq, pg, i, time + bt.c_t[i] * dt, true)
    end
    _block_checkpoint!(ts, eq, pg, s + 1, dt, time)
    copyto!(parent(bs.data), 1, parent(inner.rho_cand), 1, nb * N)
    if inner.work.inadmissible[] > 0 && !inner.work.warned[]
        inner.work.warned[] = true
        @warn "MOOD accepted $(inner.work.inadmissible[]) inadmissible candidate(s) at first order in the step at t = $time " *
              "(dt = $dt): the first-order update itself is not admissible there. Reported once per stepper; " *
              "`mood_inadmissible(ts)` gives the count of each step."
    end
    if moving
        place_ghosts!(pg, parent(inner.x_cand), time, time + dt)
        copyto!(pg.core.positions, 1, parent(inner.x_cand), 1, N)
        pg.neighbor(pg)                                    # full search + periodic wrap for the next step
        if !(pg.mover isa NoGridMover)                     # (a static cloud with moving pieces keeps the sweep's volumes)
            _check_cloud_quality(pg, time + dt)
            inner.volume_estimator === nothing || update_volumes!(pg, inner.volume_estimator)
        end
        has_moving_pieces(pg.geometry) &&
            _with_block_vectors(v -> sweep_boundaries!(pg, time, time + dt, max_order; states = v), bs, pg)
    end
    return nothing
end

# The boundary conditions of a stage array for all blocks: a `MirrorBC` whose operator reduces over the blocks runs in
# the pipeline's passes (the mirrored fit of every block into the scratch, the reduction per ghost, the application per
# block); every other condition is applied block by block with the block's reference state
function _block_apply_bcs!(ts::BlockTimeStepper{D, W, T}, eq, pg, stage::Growable, t, N::Int, nb::Int) where {D, W, T}
    inner = ts.inner
    for (tag, bc) in pg.geometry.bc_map
        if _is_block_wall(bc, eq)
            _block_mirror!(ts, bc, pg, stage, tag, eq, t, N, nb)
        else
            _block_bc_each!(bc, pg, stage, tag, inner, eq, t, N, nb)
        end
    end
    return nothing
end
# (a function barrier: `bc` comes untyped out of the map, the loop over the blocks dispatched on it in every iteration)
function _block_bc_each!(bc, pg, stage, tag, inner, eq, t, N::Int, nb::Int)
    for b in 1:nb
        bc(pg, _blk(stage, b, N), tag, inner, BlockOf(eq, b), t, _blk(inner.rho_n, b, N))
    end
    return nothing
end

function _block_mirror!(ts::BlockTimeStepper{D, W, T}, bc::MirrorBC{Q, F, B}, pg, stage::Growable, tag::Int, eq, t, N::Int, nb::Int) where {D, W, T, Q, F, B}
    scratch = ts.ghost_scratch
    ensure_capacity!(scratch, nb * N)
    for b in 1:nb
        _launch_ghost_fit!(pg, _blk(stage, b, N), tag, BlockOf(eq, b), t, Val(Q), _WallOp(nothing), Val(true), Val(B); out = _blk(scratch, b, N))
    end
    _block_mirror_passes!(_wall_params!(ts, bc, tag, eq), ts, bc, pg, stage, tag, eq, t, N, nb)
    return nothing
end
# The wall parameter array of a tag (a `Growable` of `wall_reduction_type`, on the stepper's storage), made on first use
_wall_params!(ts::BlockTimeStepper, bc::MirrorBC, tag::Int, eq) = get!(ts.wall_params, tag) do
    Growable{wall_reduction_type(bc.op, eq)}(typeof(parent(ts.ghost_scratch)).name.wrapper)
end
# Whether a condition runs in the pipeline's wall passes (a `MirrorBC` whose operator reduces over the blocks)
_is_block_wall(bc, eq) = bc isa MirrorBC && bc.op !== nothing && wall_reduction_type(bc.op, eq) !== nothing
# (a function barrier: the parameter array comes untyped out of `wall_params`)
function _block_mirror_passes!(params::Growable, ts::BlockTimeStepper{D, W, T}, bc::MirrorBC, pg, stage::Growable, tag::Int, eq, t,
                               N::Int, nb::Int) where {D, W, T}
    scratch = ts.ghost_scratch
    S = parent(scratch)
    ensure_capacity!(params, N)
    # A moving piece: the stored ghost positions and normals belong to `boundary_time`; the wall point is taken at `t`
    gm = (has_moving_pieces(pg.geometry) ? tag_motion(pg.geometry, tag, pg.meta.boundary_time, t) :
                                           GhostMotion(Val(D), T))::GhostMotion{D, T, D * D}
    c  = pg.core
    backend = get_backend()
    foreach_particle(_block_wall_reduce_body!, backend, N,
        (params, bc.op, eq, S, N, nb, c.positions, c.is_boundary, c.tags, c.normals, c.wall_distances, c.wall_curvatures, gm, tag))
    for b in 1:nb
        foreach_particle(_block_wall_apply_body!, backend, N,
            (_blk(stage, b, N), _blk(scratch, b, N), params, bc.op, eq, b, S, N, nb, c.positions, c.is_boundary, c.tags, c.normals,
             c.wall_distances, c.wall_curvatures, gm, tag))
    end
    return nothing
end

# Stage `i` for all blocks (the final combination for `i == s + 1`): the predictor, the implicit sources (solve at a stage
# with an implicit diagonal, explicit evaluation otherwise), the boundary conditions. Returns the stage time.
function _block_form_candidate!(ts::BlockTimeStepper{D, W, T}, eq, pg, i::Int, dt::T, time::Real) where {D, W, T}
    inner   = ts.inner
    s       = inner.num_stages
    bt      = inner.tableau
    N       = pg.meta.N
    nb      = num_blocks(eq)
    backend = get_backend()
    isb     = pg.core.is_boundary
    is_final = i > s
    t_i   = is_final ? time + dt : time + bt.c_t[i] * dt
    stage = is_final ? inner.rho_cand : inner.Y_stages[i]
    for b in 1:nb
        _block_candidate!(ts, pg, b, N, i, dt)
    end
    if !is_final && !isempty(ts.implicit_sources)
        a_ii = bt.a[i, i]
        Y_i  = parent(stage)
        if abs(a_ii) > T(1e-14)
            for (n, src) in enumerate(ts.implicit_sources)
                reduce_blocks!(src, Y_i, N, nb, pg, t_i, dt * a_ii)
                for b in 1:nb
                    foreach_particle(_block_implicit_step!, backend, N,
                        (_blk(stage, b, N), _blk(inner.K_I_stages[i], b, N), src.params, src.term, b, isb,
                         dt * a_ii, inv(dt * a_ii), pg, t_i, n == 1))
                end
            end
        else
            for (n, src) in enumerate(ts.implicit_sources)
                reduce_blocks!(src, Y_i, N, nb, pg, t_i, zero(T))
                for b in 1:nb
                    foreach_particle(_block_implicit_rhs!, backend, N,
                        (_blk(inner.K_I_stages[i], b, N), _blk(stage, b, N), src.params, src.term, b, isb, pg, t_i, n == 1))
                end
            end
        end
    end
    _block_apply_bcs!(ts, eq, pg, stage, t_i, N, nb)
    return t_i
end

# The right-hand side of stage `i` for all blocks: the flux pass (all interior particles, or the dirty ones of a MOOD
# re-form) and the explicit sources (reduced at the stage state on a full pass; a partial pass reuses the parameters)
function _block_rhs!(ts::BlockTimeStepper{D, W, T}, eq, pg, i::Int, t_i, full::Bool) where {D, W, T}
    inner   = ts.inner
    N       = pg.meta.N
    nb      = num_blocks(eq)
    backend = get_backend()
    Y_i     = parent(inner.Y_stages[i])
    moving  = _block_moving(pg)
    if moving
        # the frame velocity of every particle at this stage state (also on a partial pass: the stage may be an earlier one)
        foreach_particle(_block_velocity_body!, backend, N,
            (ts.grid_w, pg.mover, eq, Y_i, N, nb, pg, pg.core.is_boundary, t_i, zero(Space{D, T})))
    end
    mover = _BlockMover(pg.mover, ts.grid_w)
    for b in 1:nb
        # (a partial pass gathers the gradients of all particles: the cache is shared by the blocks)
        compute_explicit_rhs!(inner, BlockOf(eq, b), pg, _blk(inner.K_E_stages[i], b, N), inner.K_X_stages[i],
                              inner.K_V_stages[i], inner.V_stages[i], _blk(inner.Y_stages[i], b, N), t_i, full;
                              gather = true, mover = mover, moving = moving)
    end
    for src in ts.explicit_sources
        full && reduce_blocks!(src, Y_i, N, nb, pg, t_i, zero(T))
        for b in 1:nb
            foreach_particle(_block_explicit_source!, backend, N,
                (_blk(inner.K_E_stages[i], b, N), _blk(inner.Y_stages[i], b, N), src.params, src.term, b, pg.core.is_boundary, pg, t_i,
                 full, parent(inner.work.dirty)))
        end
    end
    return nothing
end

# The Euler substep W_{i-1} = Y_{i-1} + (Δt/C) K_E[i-1] of every block (what MOOD checks with an SSP tableau)
function _block_form_substep!(ts::BlockTimeStepper{D, W, T}, pg, i::Int, dt::T) where {D, W, T}
    inner = ts.inner
    N     = pg.meta.N
    nb    = num_blocks(ts.pde)                 # (checked against the state's and the step's equation at the step start)
    with_KI = !isempty(ts.implicit_sources)
    for b in 1:nb
        foreach_particle(_substep_body!, get_backend(), N,
            (_blk(inner.rho_sub, b, N), _blk(inner.Y_stages[i - 1], b, N), _blk(inner.K_E_stages[i - 1], b, N), _blk(inner.K_I_stages[i - 1], b, N),
             (inner.K_V_stages[i - 1], inner.V_stages[i - 1]), pg.core.is_boundary, inner.substep_ratio * dt, with_KI, false))
    end
    return nothing
end

# The candidate MOOD checks at checkpoint `i` (the substep or the stage state / final combination), block `b`
@inline function _block_candidate_view(ts::BlockTimeStepper, i::Int, cand_sub::Bool, b::Int, N::Int)
    inner = ts.inner
    cand_sub && return _blk(inner.rho_sub, b, N)
    return i > inner.num_stages ? _blk(inner.rho_cand, b, N) : _blk(inner.Y_stages[i], b, N)
end
@inline function _block_candidate_array(ts::BlockTimeStepper, i::Int, cand_sub::Bool)
    inner = ts.inner
    cand_sub && return parent(inner.rho_sub)
    return parent(i > inner.num_stages ? inner.rho_cand : inner.Y_stages[i])
end

# `detect_and_drop!` of the block pipeline: the flag pass per block (the criterion on the block's components, the per-block
# admissibility), the cross-block admissibility, one drop pass (the particle's order is shared by its blocks)
function _block_detect_and_drop!(ts::BlockTimeStepper, eq, pg, i::Int, cand_sub::Bool, full::Bool)
    inner = ts.inner
    work  = inner.work
    inner.mood.criterion isa NoMOOD && return false
    N  = pg.meta.N
    nb = num_blocks(eq)
    _mood_reset!(work, N)
    interp = inner.divergence_interpolator
    regather = inner.mood.criterion isa MOODu2 && has_cached_gradients(interp)
    for b in 1:nb
        # MOODu2 reads the scheme's gradient cache (its curvature test), which holds the last block's gradients:
        # the block's own gradients at the previous stage are gathered first (one pre-pass per block and checkpoint)
        regather && foreach_particle(_rhs_gather!, get_backend(), N,
            (_blk(inner.Y_stages[i - 1], b, N), pg.core.is_boundary, pg.neighbor.ranges, inner.int_buffer, interp, pg, parent(work.changed), true))
        _mood_flag_pass!(inner.mood, work, interp, pg, _blk(inner.Y_stages[i - 1], b, N),
                         _block_candidate_view(ts, i, cand_sub, b, N), full, BlockOf(eq, b))
    end
    foreach_particle(_block_admissible_flag!, get_backend(), N,
        (parent(work.triggered), pg.core.is_boundary, parent(work.dirty), full, eq, _block_candidate_array(ts, i, cand_sub), N, nb))
    return _mood_drop_pass!(inner.mood, work, pg)
end

# Interior particles whose candidate is not admissible across the blocks (counted into `work.inadmissible`)
function _block_count_inadmissible(ts::BlockTimeStepper, eq, pg, i::Int, cand_sub::Bool)
    N = pg.meta.N
    N == 0 && return 0
    flags = parent(ts.inner.work.scratch)
    foreach_particle(_block_inadmissible_body!, get_backend(), N,
        (flags, pg.core.is_boundary, eq, _block_candidate_array(ts, i, cand_sub), N, num_blocks(eq)))
    return count(view(flags, 1:N))
end

# The MOOD checkpoint of the block pipeline (see `mood_checkpoint!`): the candidate of checkpoint `i` for all blocks, the
# stage-wise fixed point against Y_{i-1} with the recomputation of K_E[i-1] on the affected particles of every block
function _block_checkpoint!(ts::BlockTimeStepper{D, W, T}, eq, pg, i::Int, dt::T, time::Real) where {D, W, T}
    inner  = ts.inner
    work   = inner.work
    N      = pg.meta.N
    t_prev = time + inner.tableau.c_t[i - 1] * dt
    reach  = _order_reach(inner.divergence_interpolator, inner.mood.strategy)
    full = true
    while true
        _block_form_candidate!(ts, eq, pg, i, dt, time)
        cand_sub = _checks_substeps(inner)
        cand_sub && _block_form_substep!(ts, pg, i, dt)
        if !_block_detect_and_drop!(ts, eq, pg, i, cand_sub, full)
            inner.mood.criterion isa NoMOOD || (work.inadmissible[] += _block_count_inadmissible(ts, eq, pg, i, cand_sub))
            return nothing
        end
        copyto!(parent(work.dirty), 1, parent(work.changed), 1, N)
        expand_mask!(parent(work.dirty), parent(work.scratch), pg, reach)
        _block_rhs!(ts, eq, pg, i - 1, t_prev, false)
        full = false
    end
end

# The predictor of stage `i` (the final combination for `i == s + 1`) for block `b`: the stage kernel of
# `GeneralRKTimeStepper` on the block's views (static grid: the position and volume arguments are unused)
function _block_candidate!(ts::BlockTimeStepper{D, W, T}, pg, b::Int, N::Int, i::Int, dt::T) where {D, W, T}
    inner    = ts.inner
    s        = inner.num_stages
    is_final = i > s
    target   = is_final ? _blk(inner.rho_cand, b, N) : _blk(inner.Y_stages[i], b, N)
    U_n      = _blk(inner.rho_n, b, N)
    K_E      = map(g -> _blk(g, b, N), inner.K_E_stages)
    K_I      = map(g -> _blk(g, b, N), inner.K_I_stages)
    moving   = _block_moving(pg)
    x_target = is_final ? inner.x_cand : inner.X_stages[i]
    v_target = is_final ? inner.v_cand : inner.V_stages[i]
    foreach_particle(_stage_predictor!, get_backend(), N,
        (target, (x_target, v_target), U_n, (inner.x_n, inner.v_n), K_E, K_I, (inner.K_X_stages, inner.K_V_stages, inner.V_stages),
         U_n, pg.core.is_boundary, inner.tableau, i, is_final ? s : i - 1, is_final, dt, moving, false, zero(T), zero(Space{D, T})))
    return nothing
end

"""
    get_time_step(pg, eq, interp, bs::BlockState; cached_weights = true) -> T

The CFL step of a block state: the smallest step over the blocks (`get_time_step` of every block with its
`BlockOf` equation and its state view).
"""
function get_time_step(pg::ParticleGrid{D, W, T}, eq::HyperbolicPDE{D, W, T}, main_grad, bs::BlockState{W, T};
                       cached_weights::Bool = true) where {D, W, T}
    bs.N[] == pg.meta.N || throw(ArgumentError("get_time_step: the block state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    bs.nb == num_blocks(eq) || throw(ArgumentError("get_time_step: the state has $(bs.nb) blocks, the equation $(num_blocks(eq))"))
    dt = T(Inf)
    for b in 1:bs.nb
        dt = min(dt, get_time_step(pg, BlockOf(eq, b), main_grad; cached_weights = cached_weights, U = block_view(bs, b)))
    end
    return dt
end
