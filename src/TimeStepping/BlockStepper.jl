export BlockState, BlockTimeStepper, BlockOf, num_blocks, block_flux, block_max_eigenvalue, block_admissible, block_view, set_block_states!
export AbstractBlockSource, AbstractExplicitBlockSource, AbstractImplicitBlockSource, BlockSourceTerm
export reduction_type, block_reduce, block_source, block_implicit_solve
export block_fluid_velocity, wall_reduction_type, block_wall_reduce, block_wall_apply

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
"The reduction of a block source over the blocks of particle `p` (see [`AbstractBlockSource`](@ref))."
block_reduce(st::AbstractBlockSource, p, Y, N, nb, pg, t, dt_coeff) = error("`block_reduce` not implemented for $(typeof(st))")
"The source of block `b` from the reduced parameters (see [`AbstractBlockSource`](@ref))."
block_source(st::AbstractBlockSource, U, b, params, p, pg, t) = error("`block_source` not implemented for $(typeof(st))")
"The implicit stage solution of block `b` from the reduced parameters (see [`AbstractBlockSource`](@ref))."
block_implicit_solve(st::AbstractImplicitBlockSource, U, b, params, dt_coeff, p, pg, t) =
    error("`block_implicit_solve` not implemented for $(typeof(st))")

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
    finite = true
    for b in 1:nb
        finite &= all(isfinite, Y[(b - 1) * N + p])
    end
    flags[p] = !(finite && block_admissible(eq, Y, N, nb, p))
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
    return view(parent(bs.data), ((b - 1) * N + 1):(b * N))
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
forcing halos), with the particle's order shared by all its blocks: the criterion (`MOODu1`, `OnlyMOOD`,
`MOODPAD`) checks every block's components, the per-block admissibility is that every block is finite, and
[`block_admissible`](@ref) adds the cross-block check (e.g. positive moments); a flagged particle drops one
order, and the previous stage's right-hand side is recomputed on the affected particles for all blocks.
`MOODu2` is not available yet (its curvature test reads the gradient cache, which holds the last block's).

On a moving grid (`pg.mover`) the frame velocity of every particle is reduced before each flux pass
([`block_fluid_velocity`](@ref) for the Lagrangian movers, the mover's velocity otherwise) and the pass runs in the
frozen-velocity ALE frame of the particle; the positions are stage variables as in the standard stepper, the
volumes are re-estimated at the end of the step. (The conservative Lagrangian frame of the standard stepper needs
the frame velocity of every interface state, which a block cannot give.) Moving boundary pieces are not available
yet. The boundary conditions of the grid apply per block with the block's state of the step start as the
reference; a `MirrorBC` whose operator reduces over the blocks ([`block_wall_reduce`](@ref)) runs in the
pipeline's passes. For one block (`num_blocks(eq) == 1`) the
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
    inner.mood.criterion isa MOODu2 && throw(ArgumentError("BlockTimeStepper: MOODu2 is not available in the block pipeline yet " *
                                                           "(its curvature test needs the gradient cache of the checked block); use MOODu1 or MOODPAD"))
    AT = typeof(parent(inner.rho_n))
    gw = Growable{Space{D, T}}(AT.name.wrapper)
    gs = Growable{State{W, T}}(AT.name.wrapper)
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

@inline _blk(g::Growable, b::Int, N::Int) = view(parent(g), ((b - 1) * N + 1):(b * N))
@inline _block_moving(pg::ParticleGrid) = !(pg.mover isa NoGridMover)

"""
    mood_flags(ts::BlockTimeStepper), mood_inadmissible(ts::BlockTimeStepper)

The per-step MOOD diagnostics of the block pipeline (those of the inner stepper, per particle).
"""
mood_flags(ts::BlockTimeStepper) = mood_flags(ts.inner)
mood_inadmissible(ts::BlockTimeStepper) = mood_inadmissible(ts.inner)

function (ts::BlockTimeStepper{D, W, T})(eq::HyperbolicPDE{D, W, T}, pg::ParticleGrid{D, W, T}, bs::BlockState{W, T},
                                         time::Real, dt::T) where {D, W, T}
    has_moving_pieces(pg.geometry) && throw(ArgumentError("BlockTimeStepper: moving boundary pieces are not available in the block pipeline yet"))
    N  = pg.meta.N
    nb = bs.nb
    nb == num_blocks(eq) || throw(ArgumentError("BlockTimeStepper: the state has $nb blocks, the equation $(num_blocks(eq))"))
    bs.N[] == N || throw(ArgumentError("BlockTimeStepper: the state was set for $(bs.N[]) particles, the grid has $N"))
    inner  = ts.inner
    s      = inner.num_stages
    bt     = inner.tableau
    interp = inner.divergence_interpolator
    orders = pg.core.particle_orders
    moving = _block_moving(pg)

    update_size!(inner, nb * N, length(pg.neighbor.indices), moving)
    update_size!(inner.work, N)                   # the MOOD masks are per particle
    update_size!(interp, N)
    moving && ensure_capacity!(ts.grid_w, N)
    copyto!(parent(inner.rho_n), 1, parent(bs.data), 1, nb * N)
    moving && copyto!(parent(inner.x_n), 1, pg.core.positions, 1, N)
    moving && copyto!(parent(inner.v_n), 1, pg.core.volumes, 1, N)

    max_order = _extract_order(interp)
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
        copyto!(pg.core.positions, 1, parent(inner.x_cand), 1, N)
        pg.neighbor(pg)                                    # full search + periodic wrap for the next step
        _check_cloud_quality(pg, time + dt)
        inner.volume_estimator === nothing || update_volumes!(pg, inner.volume_estimator)
    end
    return nothing
end

# The boundary conditions of a stage array for all blocks: a `MirrorBC` whose operator reduces over the blocks runs in
# the pipeline's passes (the mirrored fit of every block into the scratch, the reduction per ghost, the application per
# block); every other condition is applied block by block with the block's reference state
function _block_apply_bcs!(ts::BlockTimeStepper{D, W, T}, eq, pg, stage::Growable, t, N::Int, nb::Int) where {D, W, T}
    inner = ts.inner
    for (tag, bc) in pg.geometry.bc_map
        if bc isa MirrorBC && bc.op !== nothing && wall_reduction_type(bc.op, eq) !== nothing
            _block_mirror!(ts, bc, pg, stage, tag, eq, t, N, nb)
        else
            for b in 1:nb
                bc(pg, _blk(stage, b, N), tag, inner, BlockOf(eq, b), t, _blk(inner.rho_n, b, N))
            end
        end
    end
    return nothing
end

function _block_mirror!(ts::BlockTimeStepper{D, W, T}, bc::MirrorBC{Q, F, B}, pg, stage::Growable, tag::Int, eq, t, N::Int, nb::Int) where {D, W, T, Q, F, B}
    scratch = ts.ghost_scratch
    ensure_capacity!(scratch, nb * N)
    Y, S = parent(stage), parent(scratch)
    for b in 1:nb
        _launch_ghost_fit!(pg, _blk(stage, b, N), tag, BlockOf(eq, b), t, Val(Q), _WallOp(nothing), Val(true), Val(B); out = _blk(scratch, b, N))
    end
    P = wall_reduction_type(bc.op, eq)
    params = get!(ts.wall_params, tag) do
        Growable{P}(typeof(parent(scratch)).name.wrapper)
    end
    ensure_capacity!(params, N)
    gm = GhostMotion(Val(D), T)                        # static pieces only (moving pieces are refused)
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
    nb    = num_blocks(ts.pde)
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
    for b in 1:nb
        _mood_flag_pass!(inner.mood, work, inner.divergence_interpolator, pg, _blk(inner.Y_stages[i - 1], b, N),
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
    dt = T(Inf)
    for b in 1:bs.nb
        dt = min(dt, get_time_step(pg, BlockOf(eq, b), main_grad; cached_weights = cached_weights, U = block_view(bs, b)))
    end
    return dt
end
