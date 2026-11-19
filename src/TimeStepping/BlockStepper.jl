export BlockState, BlockTimeStepper, BlockOf, num_blocks, block_flux, block_max_eigenvalue, block_admissible, block_view, set_block_states!,
       block_velocity, block_is_admissible, block_nc_jump
export AbstractBlockSource, AbstractExplicitBlockSource, AbstractImplicitBlockSource, BlockSourceTerm
export reduction_type, block_reduce, block_source, block_implicit_solve
export reduction_partial_type, block_partial, finalize_reduction
export block_fluid_velocity, wall_reduction_type, block_wall_reduce, block_wall_apply
export fluid_velocity_partial_type, block_fluid_velocity_partial, finalize_fluid_velocity
export admissible_partial_type, block_admissible_partial, finalize_admissible
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
@inline block_admissible(eq::HyperbolicPDE, Y, N::Int, nb::Int, p::Int) = _block_admissible(admissible_partial_type(eq), eq, Y, N, nb, p)
@inline _block_admissible(::Nothing, eq, Y, N, nb, p) = true
@inline _block_admissible(::Type, eq, Y, N, nb, p) = finalize_admissible(eq, block_admissible_partial(eq, Y, N, nb, 1:nb, p))

"""
    block_fluid_velocity(eq::HyperbolicPDE{D, W, T}, Y, N, nb, p) -> Space{D, T}

The fluid velocity of particle `p` of a block equation across its blocks (block `b` at `Y[(b - 1) N + p]`): what a
`LagrangianMover` or `ScaledLagrangianMover` moves the particle with in the block pipeline (a discrete-velocity
model: the mean velocity of the moments). Runs inside kernels. No default.
"""
@inline block_fluid_velocity(eq::HyperbolicPDE, Y, N::Int, nb::Int, p::Int) = _block_fluid_velocity(fluid_velocity_partial_type(eq), eq, Y, N, nb, p)
@inline _block_fluid_velocity(::Nothing, eq, Y, N, nb, p) =
    error("`block_fluid_velocity` not implemented for $(typeof(eq)) (needed by a Lagrangian mover in the block pipeline)")
@inline _block_fluid_velocity(::Type, eq, Y, N, nb, p) = finalize_fluid_velocity(eq, block_fluid_velocity_partial(eq, Y, N, nb, 1:nb, p))

"""
    wall_reduction_type(op, eq) -> Type or nothing
    block_wall_reduce(op, eq, Y, N, nb, g, wall::WallPoint[, Yw]) -> params
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
"The reduction of a wall operator over the mirrored blocks of ghost `g` (see [`wall_reduction_type`](@ref)); on a state split over devices it runs on every part over the gathered ghost rows of all blocks."
block_wall_reduce(op, eq, Y, N, nb, g, wall) = error("`block_wall_reduce` not implemented for $(typeof(op)) on $(typeof(eq))")
"The reduction of an operator that takes the wall-point states too (`uses_wall_state`): `Yw` holds them, block-major like `Y`."
block_wall_reduce(op, eq, Y, N, nb, g, wall, Yw) = error("`block_wall_reduce` with the wall-point states not implemented for $(typeof(op)) on $(typeof(eq))")
@inline _wall_reduce(op, eq, S, N, nb, g, wall, ::Nothing) = block_wall_reduce(op, eq, S, N, nb, g, wall)
@inline _wall_reduce(op, eq, S, N, nb, g, wall, Sw) = block_wall_reduce(op, eq, S, N, nb, g, wall, Sw)
"The ghost state of block `b` from its mirrored value and the wall parameters (see [`wall_reduction_type`](@ref))."
block_wall_apply(op, eq, U, b, params, wall, Y, N, nb, g) = error("`block_wall_apply` not implemented for $(typeof(op)) on $(typeof(eq))")

"""
    fluid_velocity_partial_type(eq) -> Type or nothing
    block_fluid_velocity_partial(eq, Y, N, nb, blocks::UnitRange, p) -> partial
    finalize_fluid_velocity(eq, partial) -> Space{D, T}

The additive form of [`block_fluid_velocity`](@ref) (a discrete-velocity model: the density and momentum of the
velocities of a block range, the velocity from the sum), which a state split over devices exchanges; an equation that
gives it gets `block_fluid_velocity` for free. Default `nothing`.
"""
fluid_velocity_partial_type(eq::HyperbolicPDE) = nothing
"The contribution of the blocks `blocks` of particle `p` to the fluid velocity, additive over block ranges (see [`fluid_velocity_partial_type`](@ref))."
block_fluid_velocity_partial(eq::HyperbolicPDE, Y, N, nb, blocks::UnitRange, p) = error("`block_fluid_velocity_partial` not implemented for $(typeof(eq))")
"The fluid velocity from the sum of the partials over all blocks (see [`fluid_velocity_partial_type`](@ref))."
finalize_fluid_velocity(eq::HyperbolicPDE, partial) = error("`finalize_fluid_velocity` not implemented for $(typeof(eq))")

"""
    admissible_partial_type(eq) -> Type or nothing
    block_admissible_partial(eq, Y, N, nb, blocks::UnitRange, p) -> partial
    finalize_admissible(eq, partial) -> Bool

The additive form of [`block_admissible`](@ref) (the moments of a block range, the positivity from the sum), which a
state split over devices exchanges; an equation that gives it gets `block_admissible` for free. Default `nothing`: the
default `block_admissible` (`true`) also holds on a split state; an equation that implements `block_admissible` without
the additive form is refused by `BlockDevices`.
"""
admissible_partial_type(eq::HyperbolicPDE) = nothing
"The contribution of the blocks `blocks` of particle `p` to the cross-block admissibility, additive over block ranges (see [`admissible_partial_type`](@ref))."
block_admissible_partial(eq::HyperbolicPDE, Y, N, nb, blocks::UnitRange, p) = error("`block_admissible_partial` not implemented for $(typeof(eq))")
"The admissibility of a particle from the sum of the partials over all blocks (see [`admissible_partial_type`](@ref))."
finalize_admissible(eq::HyperbolicPDE, partial) = error("`finalize_admissible` not implemented for $(typeof(eq))")

"""
    _OffsetBlocks(a, shift)

The block-major array of a block range seen with global block indices: `x[i] == a[i − shift]` with
`shift = (first_block − 1) N`, so that the additive reductions of a part of a split state index it as they index the
whole state. A bits struct (kernel argument), not an `AbstractVector`.
"""
struct _OffsetBlocks{A}
    a::A
    shift::Int
end
Base.@propagate_inbounds Base.getindex(x::_OffsetBlocks, i::Int) = x.a[i - x.shift]
Base.@propagate_inbounds Base.setindex!(x::_OffsetBlocks, v, i::Int) = (x.a[i - x.shift] = v)
Base.eltype(::Type{_OffsetBlocks{A}}) where {A} = eltype(A)
Base.length(x::_OffsetBlocks) = x.shift + length(x.a)
Adapt.adapt_structure(to, x::_OffsetBlocks) = _OffsetBlocks(adapt(to, x.a), x.shift)

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
@inline function _block_inadmissible_body!(p, flags, is_boundary, eq, Y, N, nb, off)
    if is_boundary[p]
        flags[p] = false
        return nothing
    end
    ok = true
    for b in 1:nb                                    # (the check of the flag pass, not only finiteness)
        ok &= block_is_admissible(eq, Y[(b - 1) * N + p], b + off)
    end
    flags[p] = !(ok && (off != 0 || block_admissible(eq, Y, N, nb, p)))   # (a part's cross-block check is exchanged, see BlockDevices.jl)
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
@inline function _block_wall_reduce_body!(g, params, op, eq, S, N, nb, pos, is_boundary, tags, normals, wdist, curvs, gm, tag, Sw)
    (is_boundary[g] && tags[g] == tag) || return nothing
    params[g] = _wall_reduce(op, eq, S, N, nb, g, _ghost_wall_point(g, pos, normals, wdist, curvs, gm), Sw)
    return nothing
end
@inline function _block_wall_apply_body!(g, Y_b, S_b, params, op, eq, b, S, N, nb, pos, is_boundary, tags, normals, wdist, curvs, gm, tag)
    (is_boundary[g] && tags[g] == tag) || return nothing
    Y_b[g] = block_wall_apply(op, eq, S_b[g], b, params[g], _ghost_wall_point(g, pos, normals, wdist, curvs, gm), S, N, nb, g)
    return nothing
end

# =========================================================================
# LAUNCHES OVER THE PAIRS (see `foreach_block`, Backend.jl): the block equation and the scratch of a launch slot
# =========================================================================
@inline _block_arg(x::PerBlockEq, b, c, N, off) = BlockOf(x.eq, b + off)
# The interaction buffer of a launch slot: the slots `off + 1 … off + len` (the neighbour slots of the grid, once per block
# of the launch; `len == 0` when the scheme does not use them)
@inline _slot_view(ib::InteractionBuffer{D, M, T}, off::Int, len::Int) where {D, M, T} =
    InteractionBuffer{D, M, T}((@inbounds view(ib.df_flux, (off + 1):(off + len))), (@inbounds view(ib.df_scratch, (off + 1):(off + len))),
                               (@inbounds view(ib.mask, (off + 1):(off + len))))
# The scheme of a launch slot, given as `(interp, scratch)`: a MUSCL scheme with the stepper's replicated gradient cache
# (per particle, the one state-dependent cache of the scheme) shifted by the slot; the geometry caches (the MLS weights,
# the slot weights) are shared by the blocks. Any other scheme has no per-particle scratch and is passed as it is.
@inline _slot_view(x::Tuple{<:DivergenceInterpolator, Nothing}, off::Int, len::Int) = x[1]
@inline function _slot_view(x::Tuple{<:MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}, <:AbstractVector}, off::Int, len::Int) where {D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}
    m, scratch = x
    return MUSCL{D, M, T, B_LEN, MAX_ORDER, DIV_ORDER}(m.interpolators, m.limiter, m.flux, (@inbounds view(scratch, (off + 1):(off + len))),
                                                      m.grad_weights, m.div_weights, m.dweights, m.mweights, m.streamed, m.moment)
end
# The replicated per-particle scratch of a scheme for the blocks of one launch (`blocks_per_launch` copies, on the
# stepper's storage): the gradient cache of a MUSCL scheme; `nothing` for a scheme without one
_launch_scratch(::DivergenceInterpolator, AT) = nothing
_launch_scratch(m::MUSCL, AT) = Growable{eltype(m.gradients)}(AT)
_launch_scratch_size!(::Nothing, n::Int) = nothing
function _launch_scratch_size!(g::Growable, n::Int)
    ensure_capacity!(g, n)
    fill!(view(parent(g), 1:n), zero(eltype(g)))      # (as `update_size!` of the scheme does for its own cache)
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
struct BlockState{W, T, G <: Growable{State{W, T}}, GT <: Growable{T}}
    data::G
    nb::Int                         # the blocks held (all blocks of the equation, or the range of a part)
    N::Base.RefValue{Int}
    cfl::GT                         # the CFL estimate per (particle, block) (`get_time_step`)
    blocks::UnitRange{Int}          # the global indices of the blocks held
end

function BlockState(eq::HyperbolicPDE{D, W, T}, pg::ParticleGrid{D, W, T}; storage::Type{AT} = Array,
                    blocks::UnitRange{Int} = 1:num_blocks(eq)) where {D, W, T, AT <: AbstractArray}
    nb = num_blocks(eq)
    nb >= 1 || throw(ArgumentError("num_blocks($(nameof(typeof(eq)))) = $nb"))
    (first(blocks) >= 1 && last(blocks) <= nb) || throw(ArgumentError("BlockState: the blocks $blocks are not within 1:$nb"))
    data = Growable{State{W, T}}(AT)
    ensure_capacity!(data, length(blocks) * pg.meta.N)
    cfl  = Growable{T}(AT)
    return BlockState{W, T, typeof(data), typeof(cfl)}(data, length(blocks), Ref(pg.meta.N), cfl, blocks)
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
    off = first(bs.blocks) - 1
    for b in 1:bs.nb, i in 1:N
        buf[(b - 1) * N + i] = f(xs[i], b + off)
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

Every pass runs in **one launch over all blocks** (`foreach_block`: `nb · N` work-items, the stage kernels of
the standard stepper on the views of the blocks), except the passes that use per-particle scratch of the scheme (the
pre-pass and the flux pass with the interaction buffer and the gradient cache, the `MOODu2` pre-pass), which run over
`blocks_per_launch` blocks at a time with the scratch replicated that many times (`:auto`, the default: `256 ÷ W` blocks,
at least one, so that a launch covers about 256 components whatever the width and the scratch stays about 256/W times
the scheme's slot and gradient buffers of one block, see `memory_estimate`; `0`: all blocks at once; a width of 60 with
32 blocks per launch ran out of memory on a 16 GB GPU where 8 with 32 did not). A launch of `N` work-items per block
never filled a GPU and paid the launch latency `nb` times per pass.

`blocks` is the range of blocks the stepper runs (all of the equation by default; a range for the part of a state
split over devices, see [`BlockDevices`](@ref): such a stepper steps through the devices object only).

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
block's state of the step start as the reference: the package's conditions (`FixedDirichlet`, `OutflowBC`,
`ExtrapolationBC`, `MirrorBC`) in one launch over the blocks (the outflow's donor search once per tag; the ghost fits
with the geometry of the fit, the donor, the degree and the Cholesky factor, once per ghost and the fit per (ghost,
block)), any other condition block by block through its functor; the CFL step of the state likewise; a `MirrorBC` whose operator reduces over the blocks
([`block_wall_reduce`](@ref)) runs in the pipeline's passes. For one block (`num_blocks(eq) == 1`) the
result is bit-identical to `GeneralRKTimeStepper` without sources, and agrees to round-off with the equivalent
sources (a separately compiled flux pass).
"""
struct BlockTimeStepper{D, W, T, EQ <: HyperbolicPDE{D, W, T}, TS <: GeneralRKTimeStepper{D, W, T},
                        EST <: Tuple{Vararg{BlockSourceTerm}}, IST <: Tuple{Vararg{BlockSourceTerm}},
                        GW <: Growable{Space{D, T}}, GS <: Growable{State{W, T}}, GI <: Growable{Int}, GG} <: TimeStepper
    pde::EQ
    inner::TS
    explicit_sources::EST
    implicit_sources::IST
    grid_w::GW                      # the frame velocities of a moving grid (per particle)
    ghost_scratch::GS               # the mirrored blocks of the wall ghosts (block-major)
    ghost_scratch_w::GS             # the wall-point blocks of the ghosts, for an operator with `uses_wall_state` (block-major)
    wall_params::Dict{Int, Any}     # the wall parameters per boundary tag (a Growable of `wall_reduction_type`)
    blocks_per_launch::Int          # blocks per launch of the passes with per-particle scratch (0: all)
    donors::GI                      # the donor of every outflow target (per particle)
    u2_ranges::GS                   # the global range per block of a launch (the plateau test of MOODu2)
    grad_scratch::GG                # the scheme's gradient cache for the blocks of a launch (`nothing`: none)
    ghost_geom::Dict{Int, Any}      # the per-ghost geometry of the ghost fits per basis size (a `GhostFitScratch`)
    blocks::UnitRange{Int}          # the blocks the stepper runs (all of the equation, or the range of a part)
    ghost_lists::Dict{Int, Any}     # per boundary tag: its ghosts as a list of particle indices (see `_tag_list!`)
end

"""
    BlockTimeStepper(eq, inner::GeneralRKTimeStepper, sources::Tuple; blocks_per_launch = :auto)

The block stepper around an existing standard stepper for `W` components (built for `BlockOf(eq, 1)`, e.g. by
`build_timestepper`: all of its options, the memory reservation included); `inner` must carry no sources of its own.
`blocks_per_launch`: how many blocks a launch of the passes with per-particle scratch covers (see the type): `:auto`
(`256 ÷ W`, at least 1), `0` (all) or a positive number.
"""
function BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, inner::GeneralRKTimeStepper{D, W, T}, sources::Tuple{Vararg{BlockSourceTerm}};
                          blocks_per_launch::Union{Int, Symbol} = :auto, blocks::UnitRange{Int} = 1:num_blocks(eq)) where {D, W, T}
    blocks_per_launch = _blocks_per_launch(blocks_per_launch, W)
    (first(blocks) >= 1 && last(blocks) <= num_blocks(eq)) ||
        throw(ArgumentError("BlockTimeStepper: the blocks $blocks are not within 1:$(num_blocks(eq))"))
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
    gs  = Growable{State{W, T}}(AT.name.wrapper)
    gsw = Growable{State{W, T}}(AT.name.wrapper)
    gi = Growable{Int}(AT.name.wrapper)
    gr = Growable{State{W, T}}(AT.name.wrapper)
    gg = _launch_scratch(inner.divergence_interpolator, AT.name.wrapper)
    # the parameter arrays of the sources live where the stage arrays live (a `BlockSourceTerm` built with the default
    # `storage = Array` for a device stepper failed at the first launch: a host array as a kernel argument)
    explicit_sts = map(st -> _on_storage(st, AT.name.wrapper), explicit_sts)
    implicit_sts = map(st -> _on_storage(st, AT.name.wrapper), implicit_sts)
    return BlockTimeStepper{D, W, T, typeof(eq), typeof(inner), typeof(explicit_sts), typeof(implicit_sts), typeof(gw), typeof(gs), typeof(gi), typeof(gg)}(
        eq, inner, explicit_sts, implicit_sts, gw, gs, gsw, Dict{Int, Any}(), blocks_per_launch, gi, gr, gg, Dict{Int, Any}(), blocks,
        Dict{Int, Any}())
end

# The blocks per launch from the keyword: `:auto` covers about 256 components per launch
function _blocks_per_launch(n::Symbol, W::Int)
    n === :auto || throw(ArgumentError("BlockTimeStepper: blocks_per_launch must be :auto, 0 (all blocks in one launch) or a positive number, got $(repr(n))"))
    return max(1, 256 ÷ W)
end
function _blocks_per_launch(n::Int, W::Int)
    n >= 0 || throw(ArgumentError("BlockTimeStepper: blocks_per_launch must be :auto, 0 (all blocks in one launch) or a positive number, got $n"))
    return n
end

function BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, div_interp::DivergenceInterpolator, mood::MOOD,
                          sources::Tuple{Vararg{BlockSourceTerm}}, tableau::IMEXButcherTableau{T}; blocks_per_launch::Union{Int, Symbol} = :auto,
                          blocks::UnitRange{Int} = 1:num_blocks(eq), kwargs...) where {D, W, T}
    inner = GeneralRKTimeStepper(BlockOf(eq, 1), div_interp, mood, tableau; kwargs...)
    return BlockTimeStepper(eq, inner, sources; blocks_per_launch, blocks)
end

# The blocks of one launch of the passes with per-particle scratch
_launch_width(ts::BlockTimeStepper, nb::Int) = ts.blocks_per_launch == 0 ? nb : min(ts.blocks_per_launch, nb)
# The chunks of blocks those passes run over
_launch_chunks(ts::BlockTimeStepper, nb::Int) = (nc = _launch_width(ts, nb); (b0:min(b0 + nc - 1, nb) for b0 in 1:nc:nb))
# The per-ghost scratch of the ghost fits of degree `Q` (one per basis size, on the stepper's storage), made on first use
function _ghost_geom!(ts::BlockTimeStepper{D, W, T}, valQ::Val) where {D, W, T}
    rows = _ghost_scratch_rows(Val(D), _ghost_degree(valQ, Val(D)))
    return get!(ts.ghost_geom, rows) do
        _ghost_scratch(Val(rows), T, Val(D), typeof(parent(ts.ghost_scratch)).name.wrapper)
    end
end
# The ghosts of boundary tag `tag` as a list of particle indices on the stepper's storage, `(indices, n)`: the passes of
# the ghost fits and the walls run over them only (`foreach_listed`, `foreach_block(…; list)`). A launch over all
# particles left most work-items idle, and a warp with one ghost of the tag cost as much as a full one: on the 2-D orifice
# (four wall tags, a quarter of the particles ghosts) the wall passes took a third of the step on a V100. The list is
# rebuilt from host copies of the flags and tags when the geometry changes (`geometry_version`: a neighbour search, the
# particle set changed); the order is the particles' order, so every ghost computes exactly what it computed before.
function _tag_list!(ts::BlockTimeStepper, pg::ParticleGrid, tag::Int)
    N = pg.meta.N
    v = pg.meta.geometry_version
    ent = get(ts.ghost_lists, tag, nothing)
    ent !== nothing && ent[1] == v && ent[4] == N && return (ent[2], ent[3])
    isb = Array(view(parent(pg.core.is_boundary), 1:N))
    tgs = Array(view(parent(pg.core.tags), 1:N))
    # (tag 0: every interior particle, tag −1: every boundary particle; the flux pass, see `_launch_flux!`)
    sel = tag == 0 ? (i -> !isb[i]) : tag == -1 ? (i -> isb[i]) : (i -> isb[i] && tgs[i] == tag)
    idx = Int[i for i in 1:N if sel(i)]
    g = ent === nothing ? Growable{Int}(typeof(parent(ts.ghost_scratch)).name.wrapper) : ent[2]
    n = length(idx)
    ensure_capacity!(g, max(n, 1))
    n > 0 && copyto!(g, 1, idx, 1, n)
    ts.ghost_lists[tag] = (v, g, n, N)
    return (g, n)
end
# The slot stride of the interaction buffer in a launch (0: the scheme does not use it)
_slot_stride(inner::GeneralRKTimeStepper, pg) = uses_slot_buffers(inner.divergence_interpolator) ? length(pg.neighbor.indices) : 0
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
function _block_update_size!(inner::GeneralRKTimeStepper, N::Int, nb::Int, M_neighbors::Int, moving::Bool, nc::Int)
    for g in (inner.rho_n, inner.rho_cand, inner.Y_stages..., inner.K_E_stages..., inner.K_I_stages...)
        ensure_capacity!(g, nb * N)
    end
    _checks_substeps(inner) && ensure_capacity!(inner.rho_sub, nb * N)
    if moving
        for g in (inner.x_n, inner.x_cand, inner.v_n, inner.v_cand, inner.X_stages..., inner.K_X_stages..., inner.V_stages..., inner.K_V_stages...)
            ensure_capacity!(g, N)
        end
    end
    uses_slot_buffers(inner.divergence_interpolator) && update_size!(inner.int_buffer, nc * M_neighbors)   # (once per block of a launch)
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
    nb = bs.nb
    (nb == num_blocks(eq) && nb == num_blocks(ts.pde) && ts.blocks == 1:nb) ||
        throw(ArgumentError("BlockTimeStepper: the state has $nb blocks, the equation $(num_blocks(eq)), the stepper's equation $(num_blocks(ts.pde)) " *
                            "(a stepper of a block range steps through `BlockDevices`)"))
    inner     = ts.inner
    bt        = inner.tableau
    s         = inner.num_stages
    max_order = _extract_order(inner.divergence_interpolator)
    # (before the sync of moving pieces, which remaps the state: a state of another particle count is refused untouched)
    bs.N[] == pg.meta.N || throw(ArgumentError("BlockTimeStepper: the state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    # moving pieces: the ghosts and the region at the step's time first (the sweep may change the particle set)
    has_moving_pieces(pg.geometry) &&
        _with_block_vectors(v -> _sync_boundaries!(pg, T(time), max_order; states = v), bs, pg)
    _block_step_begin!(ts, eq, pg, bs, time, dt)
    N      = pg.meta.N
    orders = pg.core.particle_orders
    moving = _block_moving(pg)
    for i in 1:s
        if i == 1
            _block_form_candidate!(ts, eq, pg, 1, dt, time)
        else
            _block_checkpoint!(ts, eq, pg, i, dt, time)      # uses the geometry of stage i-1
            moving && _block_stage_geometry!(ts, pg, i, time, time + bt.c_t[i] * dt)
        end
        inner.reset_orders_per_stage && fill!(view(orders, 1:N), max_order)
        _block_rhs!(ts, eq, pg, i, time + bt.c_t[i] * dt, true)
    end
    _block_checkpoint!(ts, eq, pg, s + 1, dt, time)
    _block_step_end!(ts, eq, pg, bs, time, dt)
    moving && has_moving_pieces(pg.geometry) &&
        _with_block_vectors(v -> sweep_boundaries!(pg, time, time + dt, max_order; states = v), bs, pg)
    return nothing
end

# =========================================================================
# THE PHASES OF A STEP, as launches over the stepper's block range (`ts.blocks`: all blocks of the equation for the
# stepper of a whole state, a range of them for the part of a state split over devices, see `BlockDevices.jl`). The
# block index of a launch is global (`block_offset = first(ts.blocks) − 1`), the views and the state arrays local.
# =========================================================================
@inline _block_range(ts::BlockTimeStepper) = ts.blocks
@inline _block_offset(ts::BlockTimeStepper) = first(ts.blocks) - 1
@inline _num_local(ts::BlockTimeStepper) = length(ts.blocks)

# The boundary conditions of a stage array for the blocks of the stepper: a `MirrorBC` whose operator reduces over the
# blocks runs in the pipeline's passes (the mirrored fit of every block into the scratch, the reduction per ghost, the
# application per block); the package's other conditions run in one launch over the blocks with the block's reference
# state, any other condition block by block through its functor
function _block_apply_bcs!(ts::BlockTimeStepper{D, W, T}, eq, pg, stage::Growable, t, N::Int, nb::Int) where {D, W, T}
    for (tag, bc) in pg.geometry.bc_map
        if _is_block_wall(bc, eq)
            _block_mirror!(ts, bc, pg, stage, tag, eq, t, N, nb)
        else
            _block_bc!(bc, ts, pg, stage, tag, eq, t, N, nb)
        end
    end
    return nothing
end
# (function barriers: `bc` comes untyped out of the map)
function _block_bc!(::FixedDirichlet, ts::BlockTimeStepper, pg, stage, tag::Int, eq, t, N::Int, nb::Int)
    foreach_block(_dirichlet_body!, get_backend(), N, 1:nb, (PerBlock(stage), PerBlock(ts.inner.rho_n), pg.core.tags, pg.core.is_boundary, tag))
    return nothing
end
function _block_bc!(::OutflowBC, ts::BlockTimeStepper, pg, stage, tag::Int, eq, t, N::Int, nb::Int)
    _outflow_blocks!(pg, stage, ts.inner.rho_n, tag, N, 1:nb, ts.donors)
    return nothing
end
function _block_bc!(::ExtrapolationBC{Q}, ts::BlockTimeStepper, pg, stage, tag::Int, eq, t, N::Int, nb::Int) where {Q}
    _launch_ghost_fit_blocks!(pg, stage, stage, tag, eq, t, Val(Q), nothing, Val(false), Val(false), 1:nb, _ghost_geom!(ts, Val(Q));
                              block_offset = _block_offset(ts), list = _tag_list!(ts, pg, tag))
    return nothing
end
function _block_bc!(bc::MirrorBC{Q, F, B}, ts::BlockTimeStepper, pg, stage, tag::Int, eq, t, N::Int, nb::Int) where {Q, F, B}
    # (an operator without a reduction, applied to every block's mirrored state with `BlockOf(eq, b)`)
    _launch_ghost_fit_blocks!(pg, stage, stage, tag, eq, t, Val(Q), _WallOp(bc.op), Val(true), Val(B), 1:nb, _ghost_geom!(ts, Val(Q));
                              block_offset = _block_offset(ts), list = _tag_list!(ts, pg, tag))
    return nothing
end
function _block_bc!(bc, ts::BlockTimeStepper, pg, stage, tag::Int, eq, t, N::Int, nb::Int)
    inner = ts.inner
    off   = _block_offset(ts)
    for b in 1:nb
        bc(pg, _blk(stage, b, N), tag, inner, BlockOf(eq, b + off), t, _blk(inner.rho_n, b, N))
    end
    return nothing
end

# The kinetic wall of the stepper of a whole state: the mirrored fit of every block, the reduction per ghost over all
# blocks, the application per block (a part of a split state runs the same launches with the exchanges in between,
# see `BlockDevices.jl`)
function _block_mirror!(ts::BlockTimeStepper{D, W, T}, bc::MirrorBC{Q, F, B}, pg, stage::Growable, tag::Int, eq, t, N::Int, nb::Int) where {D, W, T, Q, F, B}
    _launch_wall_fit!(ts, bc, pg, stage, tag, eq, t, nb)
    params = _wall_params!(ts, bc, tag, eq)
    ensure_capacity!(params, N)
    gm = _wall_motion(pg, tag, t)
    c  = pg.core
    foreach_listed(_block_wall_reduce_body!, get_backend(), _tag_list!(ts, pg, tag)...,
        (params, bc.op, eq, parent(ts.ghost_scratch), N, nb, c.positions, c.is_boundary, c.tags, c.normals, c.wall_distances,
         c.wall_curvatures, gm, tag, _wall_scratch_w(ts, bc)))
    _launch_wall_apply!(ts, bc, pg, stage, tag, eq, t, N, nb, params, parent(ts.ghost_scratch))
    return nothing
end
# The mirrored fit of the blocks of the stepper into its scratch (block-major, the blocks of the range); for an operator
# that takes the wall-point states, those into the second scratch
function _launch_wall_fit!(ts::BlockTimeStepper, bc::MirrorBC{Q, F, B}, pg, stage::Growable, tag::Int, eq, t, nb::Int) where {Q, F, B}
    ensure_capacity!(ts.ghost_scratch, nb * pg.meta.N)
    out_w = nothing
    if uses_wall_state(bc.op)
        ensure_capacity!(ts.ghost_scratch_w, nb * pg.meta.N)
        out_w = ts.ghost_scratch_w
    end
    _launch_ghost_fit_blocks!(pg, stage, ts.ghost_scratch, tag, eq, t, Val(Q), _WallOp(nothing), Val(true), Val(B), 1:nb,
                              _ghost_geom!(ts, Val(Q)); block_offset = _block_offset(ts), out_w, list = _tag_list!(ts, pg, tag))
    return nothing
end
# The wall-point scratch an operator's reduction reads (`nothing` for one that does not take it)
_wall_scratch_w(ts::BlockTimeStepper, bc::MirrorBC) = uses_wall_state(bc.op) ? parent(ts.ghost_scratch_w) : nothing
# The application per (ghost, block) from the reduced parameters; `S_all` is what the operator reads the other blocks
# of the ghost from (the scratch of a whole state, or the gathered ghost rows of a split one)
function _launch_wall_apply!(ts::BlockTimeStepper{D, W, T}, bc::MirrorBC, pg, stage::Growable, tag::Int, eq, t, N::Int, nb::Int,
                             params::Growable, S_all) where {D, W, T}
    gm = _wall_motion(pg, tag, t)
    c  = pg.core
    foreach_block(_block_wall_apply_body!, get_backend(), N, 1:nb,
        (PerBlock(stage), PerBlock(ts.ghost_scratch), params, bc.op, eq, PerBlockIndex(), S_all, N, num_blocks(ts.pde), c.positions,
         c.is_boundary, c.tags, c.normals, c.wall_distances, c.wall_curvatures, gm, tag); block_offset = _block_offset(ts),
        list = _tag_list!(ts, pg, tag))
    return nothing
end
# A moving piece: the stored ghost positions and normals belong to `boundary_time`; the wall point is taken at `t`
_wall_motion(pg::ParticleGrid{D, M, T}, tag::Int, t) where {D, M, T} =
    (has_moving_pieces(pg.geometry) ? tag_motion(pg.geometry, tag, pg.meta.boundary_time, t) :
                                      GhostMotion(Val(D), T))::GhostMotion{D, T, D * D}
# The wall parameter array of a tag (a `Growable` of `wall_reduction_type`, on the stepper's storage), made on first use
_wall_params!(ts::BlockTimeStepper, bc::MirrorBC, tag::Int, eq) = get!(ts.wall_params, tag) do
    Growable{wall_reduction_type(bc.op, eq)}(typeof(parent(ts.ghost_scratch)).name.wrapper)
end
# Whether a condition runs in the pipeline's wall passes (a `MirrorBC` whose operator reduces over the blocks)
_is_block_wall(bc, eq) = bc isa MirrorBC && bc.op !== nothing && wall_reduction_type(bc.op, eq) !== nothing

# Stage `i` for the blocks of the stepper (the final combination for `i == s + 1`): the predictor, the implicit sources
# (solve at a stage with an implicit diagonal, explicit evaluation otherwise), the boundary conditions. Returns the stage time.
function _block_form_candidate!(ts::BlockTimeStepper{D, W, T}, eq, pg, i::Int, dt::T, time::Real) where {D, W, T}
    inner   = ts.inner
    s       = inner.num_stages
    bt      = inner.tableau
    N       = pg.meta.N
    nb      = _num_local(ts)
    is_final = i > s
    t_i   = is_final ? time + dt : time + bt.c_t[i] * dt
    stage = is_final ? inner.rho_cand : inner.Y_stages[i]
    _launch_predictor!(ts, pg, N, nb, i, dt)
    if !is_final && !isempty(ts.implicit_sources)
        a_ii = bt.a[i, i]
        Y_i  = parent(stage)
        if abs(a_ii) > T(1e-14)
            for (n, src) in enumerate(ts.implicit_sources)
                reduce_blocks!(src, Y_i, N, nb, pg, t_i, dt * a_ii)
                _launch_implicit_step!(ts, pg, src, stage, inner.K_I_stages[i], N, nb, dt * a_ii, t_i, n == 1)
            end
        else
            for (n, src) in enumerate(ts.implicit_sources)
                reduce_blocks!(src, Y_i, N, nb, pg, t_i, zero(T))
                _launch_implicit_rhs!(ts, pg, src, stage, inner.K_I_stages[i], N, nb, t_i, n == 1)
            end
        end
    end
    _block_apply_bcs!(ts, eq, pg, stage, t_i, N, nb)
    return t_i
end
# The implicit stage solve and its right-hand side, and the explicit evaluation of an implicit source, from the
# source's reduced parameters (`src.params`), over the (particle, block) pairs
function _launch_implicit_step!(ts::BlockTimeStepper, pg, src::BlockSourceTerm, stage, K_I, N::Int, nb::Int, dt_ii, t_i, first::Bool)
    foreach_block(_block_implicit_step!, get_backend(), N, 1:nb,
        (PerBlock(stage), PerBlock(K_I), src.params, src.term, PerBlockIndex(), pg.core.is_boundary, dt_ii, inv(dt_ii), pg, t_i, first);
        block_offset = _block_offset(ts))
    return nothing
end
function _launch_implicit_rhs!(ts::BlockTimeStepper, pg, src::BlockSourceTerm, stage, K_I, N::Int, nb::Int, t_i, first::Bool)
    foreach_block(_block_implicit_rhs!, get_backend(), N, 1:nb,
        (PerBlock(K_I), PerBlock(stage), src.params, src.term, PerBlockIndex(), pg.core.is_boundary, pg, t_i, first);
        block_offset = _block_offset(ts))
    return nothing
end

# The right-hand side of stage `i` for the blocks of the stepper: the frame velocity of a moving grid (reduced over all
# blocks), the scheme's pre-pass and the flux pass over the blocks of a launch (all interior particles, or the dirty
# ones of a MOOD re-form; a partial pass gathers the gradients of all particles, as the cache holds another block's)
# and the explicit sources (reduced at the stage state on a full pass; a partial pass reuses the parameters)
function _block_rhs!(ts::BlockTimeStepper{D, W, T}, eq, pg, i::Int, t_i, full::Bool) where {D, W, T}
    inner = ts.inner
    N     = pg.meta.N
    nb    = _num_local(ts)
    Y_i   = parent(inner.Y_stages[i])
    if _block_moving(pg)
        # the frame velocity of every particle at this stage state (also on a partial pass: the stage may be an earlier one)
        foreach_particle(_block_velocity_body!, get_backend(), N,
            (ts.grid_w, pg.mover, eq, Y_i, N, nb, pg, pg.core.is_boundary, t_i, zero(Space{D, T})))
    end
    _launch_flux!(ts, eq, pg, i, t_i, full, N, nb)
    for src in ts.explicit_sources
        full && reduce_blocks!(src, Y_i, N, nb, pg, t_i, zero(T))
        _launch_explicit_source!(ts, pg, src, i, N, nb, t_i, full)
    end
    return nothing
end
# The pre-pass and the flux pass of `compute_explicit_rhs!` over the chunks of blocks (the frame velocities of a
# moving grid in `ts.grid_w`)
function _launch_flux!(ts::BlockTimeStepper{D, W, T}, eq, pg, i::Int, t_i, full::Bool, N::Int, nb::Int) where {D, W, T}
    inner   = ts.inner
    backend = get_backend()
    moving  = _block_moving(pg)
    interp  = inner.divergence_interpolator
    isb     = pg.core.is_boundary
    mover   = _BlockMover(pg.mover, ts.grid_w)
    stride  = _slot_stride(inner, pg)
    off     = _block_offset(ts)
    prepare_geometry!(interp, pg)                           # (refreshes the geometry caches only if the grid changed)
    # The flux pass over the interior particles only (a boundary particle's work-item returns at once, and a warp of
    # interior particles with ghosts among them idled those lanes: a quarter of the particles of the 2-D orifice are
    # ghosts); the boundary rows of the right-hand side are zeroed over the boundary particles (a full pass)
    interior = _tag_list!(ts, pg, 0)
    full && foreach_block(_boundary_rhs_body!, backend, N, 1:nb,
        (PerBlock(inner.K_E_stages[i]), inner.K_X_stages[i], inner.K_V_stages[i], moving, zero(State{W, T}), zero(Space{D, T}));
        list = _tag_list!(ts, pg, -1))
    for blocks in _launch_chunks(ts, nb)
        foreach_block(_rhs_gather!, backend, N, blocks,
            (PerBlock(inner.Y_stages[i]), isb, pg.neighbor.ranges, PerSlot(inner.int_buffer, stride), PerSlot((interp, ts.grad_scratch), N), pg,
             inner.work.changed, true); block_offset = off)
        foreach_block(_rhs_flux!, backend, N, blocks,
            (PerBlockEq(eq), PerBlock(inner.Y_stages[i]), PerBlock(inner.K_E_stages[i]), inner.K_X_stages[i], inner.K_V_stages[i],
             inner.V_stages[i], isb, pg.neighbor.ranges, PerSlot(inner.int_buffer, stride), PerSlot((interp, ts.grad_scratch), N), inner.mood.strategy,
             mover, pg, inner.explicit_sources, inner.work.dirty, full, moving, t_i, zero(State{W, T}), zero(Space{D, T})); block_offset = off,
            list = interior)
    end
    return nothing
end
# Kernel body: the right-hand side of a boundary particle, zero (what `_rhs_flux!` writes for it on a full pass)
@inline function _boundary_rhs_body!(p, K_E, K_X, K_V, moving, zero_state, zero_space)
    @inbounds begin
        K_E[p] = zero_state
        moving && (K_X[p] = zero_space; K_V[p] = zero(eltype(K_V)))
    end
    return nothing
end
# An explicit source added to the right-hand side of stage `i` from its reduced parameters
function _launch_explicit_source!(ts::BlockTimeStepper, pg, src::BlockSourceTerm, i::Int, N::Int, nb::Int, t_i, full::Bool)
    inner = ts.inner
    foreach_block(_block_explicit_source!, get_backend(), N, 1:nb,
        (PerBlock(inner.K_E_stages[i]), PerBlock(inner.Y_stages[i]), src.params, src.term, PerBlockIndex(), pg.core.is_boundary, pg, t_i,
         full, parent(inner.work.dirty)); block_offset = _block_offset(ts))
    return nothing
end

# The Euler substep W_{i-1} = Y_{i-1} + (Δt/C) K_E[i-1] of every block (what MOOD checks with an SSP tableau)
function _block_form_substep!(ts::BlockTimeStepper{D, W, T}, pg, i::Int, dt::T) where {D, W, T}
    inner = ts.inner
    N     = pg.meta.N
    nb    = _num_local(ts)
    with_KI = !isempty(ts.implicit_sources)
    foreach_block(_substep_body!, get_backend(), N, 1:nb,
        (PerBlock(inner.rho_sub), PerBlock(inner.Y_stages[i - 1]), PerBlock(inner.K_E_stages[i - 1]), PerBlock(inner.K_I_stages[i - 1]),
         (inner.K_V_stages[i - 1], inner.V_stages[i - 1]), pg.core.is_boundary, inner.substep_ratio * dt, with_KI, false))
    return nothing
end

# The candidate MOOD checks at checkpoint `i` (the substep or the stage state / final combination), all blocks
@inline function _block_candidate(ts::BlockTimeStepper, i::Int, cand_sub::Bool)
    inner = ts.inner
    cand_sub && return inner.rho_sub
    return i > inner.num_stages ? inner.rho_cand : inner.Y_stages[i]
end
@inline _block_candidate_array(ts::BlockTimeStepper, i::Int, cand_sub::Bool) = parent(_block_candidate(ts, i, cand_sub))

# `detect_and_drop!` of the block pipeline: the flag pass over the blocks (the criterion on the block's components, the
# per-block admissibility; a particle is flagged if any block flags it), the cross-block admissibility, one drop pass
# (the particle's order is shared by its blocks)
function _block_detect_and_drop!(ts::BlockTimeStepper, eq, pg, i::Int, cand_sub::Bool, full::Bool)
    inner = ts.inner
    work  = inner.work
    inner.mood.criterion isa NoMOOD && return false
    N  = pg.meta.N
    nb = _num_local(ts)
    _mood_reset!(work, N)
    _launch_flag_pass!(ts, eq, pg, i, cand_sub, full, N, nb)
    foreach_particle(_block_admissible_flag!, get_backend(), N,
        (parent(work.triggered), pg.core.is_boundary, parent(work.dirty), full, eq, _block_candidate_array(ts, i, cand_sub), N, nb))
    return _mood_drop_pass!(inner.mood, work, pg)
end
# The flag pass of the criterion over the blocks of the stepper (into `work.triggered`, which is only ever set)
function _launch_flag_pass!(ts::BlockTimeStepper, eq, pg, i::Int, cand_sub::Bool, full::Bool, N::Int, nb::Int)
    inner   = ts.inner
    work    = inner.work
    crit    = inner.mood.criterion
    interp  = inner.divergence_interpolator
    backend = get_backend()
    isb     = pg.core.is_boundary
    Y_prev  = inner.Y_stages[i - 1]
    cand    = _block_candidate(ts, i, cand_sub)
    off     = _block_offset(ts)
    # MOODu2 reads the scheme's gradient cache (its curvature test), which holds another block's gradients: the block's
    # own gradients at the previous stage are gathered first, into the cache of its launch slot (the chunks of the flux pass)
    regather = crit isa MOODu2 && has_cached_gradients(interp)
    stride   = _slot_stride(inner, pg)
    for blocks in (regather ? _launch_chunks(ts, nb) : (1:nb,))
        regather && foreach_block(_rhs_gather!, backend, N, blocks,
            (PerBlock(Y_prev), isb, pg.neighbor.ranges, PerSlot(inner.int_buffer, stride), PerSlot((interp, ts.grad_scratch), N), pg,
             parent(work.changed), true); block_offset = off)
        # (the scheme with the slot's gradient cache only when the pass reads it: the launch over all blocks at once would
        # otherwise view the scratch beyond its `blocks_per_launch` slots, which a bounds-checked run refuses)
        foreach_block(_mood_flag!, backend, N, blocks,
            (inner.mood, regather ? PerSlot((interp, ts.grad_scratch), N) : interp, pg, PerBlock(Y_prev), PerBlock(cand), pg.neighbor.ranges, isb,
             parent(work.dirty), parent(work.triggered), full, PerBlockEq(eq), _block_global_ranges!(ts, crit, Y_prev, blocks, N)); block_offset = off)
    end
    return nothing
end

# The global range per block of a launch for the plateau test of MOODu2 (`_global_range` of every block, into the
# stepper's per-slot array); `nothing` when the test is off or the criterion another
_block_global_ranges!(ts::BlockTimeStepper, crit, Y_prev, blocks, N::Int) = nothing
function _block_global_ranges!(ts::BlockTimeStepper{D, W, T}, crit::MOODu2, Y_prev, blocks::UnitRange{Int}, N::Int) where {D, W, T}
    crit.plateau > zero(crit.plateau) || return nothing
    # one reduction over the blocks of the launch (an N × nc view of the block-major array, reduced over the particles)
    A  = reshape(view(parent(Y_prev), ((first(blocks) - 1) * N + 1):(last(blocks) * N)), N, length(blocks))
    lo = mapreduce(identity, (a, b) -> min.(a, b), A; dims = 1, init = State{W, T}(ntuple(_ -> typemax(T), Val(W))))
    hi = mapreduce(identity, (a, b) -> max.(a, b), A; dims = 1, init = State{W, T}(ntuple(_ -> typemin(T), Val(W))))
    copyto!(parent(ts.u2_ranges), 1, vec(hi .- lo), 1, length(blocks))
    return PerSlotEntry(parent(ts.u2_ranges))
end

# Interior particles whose candidate is not admissible across the blocks (counted into `work.inadmissible`)
function _block_count_inadmissible(ts::BlockTimeStepper, eq, pg, i::Int, cand_sub::Bool)
    N = pg.meta.N
    N == 0 && return 0
    flags = parent(ts.inner.work.scratch)
    foreach_particle(_block_inadmissible_body!, get_backend(), N,
        (flags, pg.core.is_boundary, eq, _block_candidate_array(ts, i, cand_sub), N, _num_local(ts), _block_offset(ts)))
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

# The predictor of stage `i` (the final combination for `i == s + 1`) for the blocks of the stepper: the stage kernel of
# `GeneralRKTimeStepper` over the (particle, block) pairs on the views of the blocks (the positions and volumes of a
# moving grid are per particle: every block's work-item writes the same value)
function _launch_predictor!(ts::BlockTimeStepper{D, W, T}, pg, N::Int, nb::Int, i::Int, dt::T) where {D, W, T}
    inner    = ts.inner
    s        = inner.num_stages
    is_final = i > s
    target   = is_final ? inner.rho_cand : inner.Y_stages[i]
    U_n      = PerBlock(inner.rho_n)
    moving   = _block_moving(pg)
    x_target = is_final ? inner.x_cand : inner.X_stages[i]
    v_target = is_final ? inner.v_cand : inner.V_stages[i]
    foreach_block(_stage_predictor!, get_backend(), N, 1:nb,
        (PerBlock(target), (x_target, v_target), U_n, (inner.x_n, inner.v_n), map(PerBlock, inner.K_E_stages), map(PerBlock, inner.K_I_stages),
         (inner.K_X_stages, inner.K_V_stages, inner.V_stages), U_n, pg.core.is_boundary, inner.tableau, i, is_final ? s : i - 1, is_final,
         dt, moving, false, zero(T), zero(Space{D, T})))
    return nothing
end

# The stage geometry of a moving grid (`_apply_stage_geometry!`): a neighbour search per stage resizes the interaction
# buffer for one block's slots, the launches over the blocks need them once per block of a launch (sized only at the
# step's start, a cloud whose neighbourhoods grew within the step made the last slots of a launch run past the buffer)
function _block_stage_geometry!(ts::BlockTimeStepper, pg, i::Int, t_n, t_i)
    inner = ts.inner
    _apply_stage_geometry!(inner, pg, inner.X_stages[i], t_n, t_i)
    inner.full_search_per_stage && uses_slot_buffers(inner.divergence_interpolator) &&
        update_size!(inner.int_buffer, _launch_width(ts, _num_local(ts)) * length(pg.neighbor.indices))
    return nothing
end

# The beginning of a step for the blocks of the stepper (the stepper of a whole state, or a part of a split one): the
# sizes, the copy of the state into `rho_n`, the orders; and its end: the copy out, the positions of a moving grid
function _block_step_begin!(ts::BlockTimeStepper{D, W, T}, eq, pg, bs::BlockState{W, T}, time::Real, dt::T) where {D, W, T}
    inner  = ts.inner
    interp = inner.divergence_interpolator
    bs.N[] == pg.meta.N || throw(ArgumentError("BlockTimeStepper: the state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    bs.blocks == ts.blocks || throw(ArgumentError("BlockTimeStepper: the state holds the blocks $(bs.blocks), the stepper runs $(ts.blocks)"))
    _check_block_bcs(pg, eq)
    N      = pg.meta.N
    nb     = _num_local(ts)
    moving = _block_moving(pg)
    nc = _launch_width(ts, nb)
    _block_update_size!(inner, N, nb, length(pg.neighbor.indices), moving, nc)
    update_size!(interp, N)
    _launch_scratch_size!(ts.grad_scratch, nc * N)         # (the gradient cache once per block of a launch)
    moving && ensure_capacity!(ts.grid_w, N)
    ensure_capacity!(ts.donors, N)
    ensure_capacity!(ts.u2_ranges, nb)                     # (one launch over all blocks when the scheme has no gradient cache)
    copyto!(parent(inner.rho_n), 1, parent(bs.data), 1, nb * N)
    moving && copyto!(parent(inner.x_n), 1, pg.core.positions, 1, N)
    moving && copyto!(parent(inner.v_n), 1, pg.core.volumes, 1, N)
    fill!(view(pg.core.particle_orders, 1:N), _extract_order(interp))
    fill!(view(inner.work.triggered_step, 1:N), false)
    inner.work.inadmissible[] = 0
    return nothing
end
function _block_step_end!(ts::BlockTimeStepper{D, W, T}, eq, pg, bs::BlockState{W, T}, time::Real, dt::T) where {D, W, T}
    inner = ts.inner
    N     = pg.meta.N
    copyto!(parent(bs.data), 1, parent(inner.rho_cand), 1, _num_local(ts) * N)
    if inner.work.inadmissible[] > 0 && !inner.work.warned[]
        inner.work.warned[] = true
        @warn "MOOD accepted $(inner.work.inadmissible[]) inadmissible candidate(s) at first order in the step at t = $time " *
              "(dt = $dt): the first-order update itself is not admissible there. Reported once per stepper; " *
              "`mood_inadmissible(ts)` gives the count of each step."
    end
    if _block_moving(pg)
        place_ghosts!(pg, parent(inner.x_cand), time, time + dt)
        copyto!(pg.core.positions, 1, parent(inner.x_cand), 1, N)
        pg.neighbor(pg)                                    # full search + periodic wrap for the next step
        if !(pg.mover isa NoGridMover)                     # (a static cloud with moving pieces keeps the sweep's volumes)
            _check_cloud_quality(pg, time + dt)
            inner.volume_estimator === nothing || update_volumes!(pg, inner.volume_estimator)
        end
    end
    return nothing
end

"""
    get_time_step(pg, eq, interp, bs::BlockState; cached_weights = true) -> T

The CFL step of a block state: the smallest step over the blocks it holds (`get_time_step` of every block with its
`BlockOf` equation and its state), in one launch over the (particle, block) pairs into the state's `cfl` buffer.
"""
function get_time_step(pg::ParticleGrid{D, W, T}, eq::HyperbolicPDE{D, W, T}, main_grad, bs::BlockState{W, T};
                       cached_weights::Bool = true) where {D, W, T}
    bs.N[] == pg.meta.N || throw(ArgumentError("get_time_step: the block state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    last(bs.blocks) <= num_blocks(eq) || throw(ArgumentError("get_time_step: the state holds the blocks $(bs.blocks), the equation has $(num_blocks(eq))"))
    N   = pg.meta.N
    nb  = bs.nb
    off = first(bs.blocks) - 1
    N == 0 && return T(Inf)
    ensure_capacity!(bs.cfl, nb * N)
    buf = parent(bs.cfl)
    fill!(view(buf, 1:(nb * N)), T(Inf))
    nbr = pg.neighbor
    cached_weights && prepare_geometry!(main_grad, pg)
    sw = cached_weights ? _cfl_slot_weights(main_grad, pg) : nothing
    Y  = parent(bs.data)
    if sw === nothing
        degree = _divergence_degree(main_grad)
        invL   = one(T) / minimum(pg.meta.dx)
        foreach_block(_cfl_body!, get_backend(), N, 1:nb,
            (PerBlock(buf), pg.core.is_boundary, nbr.ranges, nbr.indices, PerBlock(Y), PerBlockEq(eq), degree, _cfl_degree_of(main_grad),
             Val(D), get_distances(pg), get_weights(pg), invL, floatmin(T), one(T) / D); block_offset = off)
    else
        foreach_block(_cfl_weights_body!, get_backend(), N, 1:nb,
            (PerBlock(buf), pg.core.is_boundary, nbr.ranges, nbr.indices, PerBlock(Y), PerBlockEq(eq), sw, Val(D), floatmin(T), one(T) / D);
            block_offset = off)
    end
    return minimum(view(buf, 1:(nb * N)); init = T(Inf))
end
