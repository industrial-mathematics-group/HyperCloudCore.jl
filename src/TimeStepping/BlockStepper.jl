export BlockState, BlockTimeStepper, BlockOf, num_blocks, block_flux, block_max_eigenvalue, block_view, set_block_states!
export AbstractBlockSource, AbstractExplicitBlockSource, AbstractImplicitBlockSource, BlockSourceTerm
export reduction_type, block_reduce, block_source, block_implicit_solve

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
@inline function _block_explicit_source!(p, K_E, Y, params, st, b, is_boundary, pg, t)
    is_boundary[p] && return nothing
    K_E[p] += block_source(st, Y[p], b, params[p], p, pg, t)
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
    BlockTimeStepper(eq, div_interp, sources::Tuple, tableau; storage = Array, kwargs...)
    BlockTimeStepper(eq, div_interp, tableau; kwargs...)
    (ts::BlockTimeStepper)(eq, pg, bs::BlockState, time, dt)

The time integrator of the block pipeline: the stage kernels of [`GeneralRKTimeStepper`](@ref) run block by
block on the views of a [`BlockState`](@ref), with the equation wrapped as [`BlockOf`](@ref)`(eq, b)`, so the
grid, the neighbour search, the MLS weights, the scheme and its pre-pass are shared by all blocks and the
state length `nb · W` is a run-time number. The stepper owns a `GeneralRKTimeStepper` for `W` components
(`inner`: the interpolator, buffers and options; its stage buffers are sized to `nb · N`), and the `kwargs` are
its options.

`sources` are [`BlockSourceTerm`](@ref)s. Explicit ones enter the right-hand side of every stage after a
reduction over the blocks at the stage state; implicit ones are solved at the stages with an implicit diagonal
(`a_ii ≠ 0`: the reduction with `dt_coeff = Δt a_ii`, then the solve per block, the right-hand side
`K_I = (U_out − U_in) / (Δt a_ii)`; several implicit terms one after another, as in the standard stepper) and
evaluated explicitly at the others; they need an IMEX tableau. The explicit part of the tableau may be
any explicit RK method.

Scope (for now): static grids (no mover, no moving pieces) and no MOOD; the boundary conditions of the grid
apply per block, with the block's state of the step start as the reference. For one block
(`num_blocks(eq) == 1`) the result is bit-identical to `GeneralRKTimeStepper` without sources, and agrees to
round-off with the equivalent sources (a separately compiled flux pass).
"""
struct BlockTimeStepper{D, W, T, EQ <: HyperbolicPDE{D, W, T}, TS <: GeneralRKTimeStepper{D, W, T},
                        EST <: Tuple{Vararg{BlockSourceTerm}}, IST <: Tuple{Vararg{BlockSourceTerm}}} <: TimeStepper
    pde::EQ
    inner::TS
    explicit_sources::EST
    implicit_sources::IST
end

function BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, div_interp::DivergenceInterpolator, sources::Tuple{Vararg{BlockSourceTerm}},
                          tableau::IMEXButcherTableau{T}; kwargs...) where {D, W, T}
    explicit_sts = filter(st -> !_is_implicit(st), sources)
    implicit_sts = filter(_is_implicit, sources)
    s = num_stages(tableau)
    if !isempty(implicit_sts) && all(i -> iszero(tableau.a[i, i]), 1:s)
        throw(ArgumentError("BlockTimeStepper: implicit block sources need an IMEX tableau with an implicit diagonal " *
                            "(e.g. IMEX_ARS222_Tableau); explicit sources work with any tableau"))
    end
    inner = GeneralRKTimeStepper(BlockOf(eq, 1), div_interp, MOOD(NoMOOD()), tableau; kwargs...)
    return BlockTimeStepper{D, W, T, typeof(eq), typeof(inner), typeof(explicit_sts), typeof(implicit_sts)}(eq, inner, explicit_sts, implicit_sts)
end
BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, div_interp::DivergenceInterpolator, tableau::IMEXButcherTableau{T}; kwargs...) where {D, W, T} =
    BlockTimeStepper(eq, div_interp, (), tableau; kwargs...)

@inline _blk(g::Growable, b::Int, N::Int) = view(parent(g), ((b - 1) * N + 1):(b * N))

function (ts::BlockTimeStepper{D, W, T})(eq::HyperbolicPDE{D, W, T}, pg::ParticleGrid{D, W, T}, bs::BlockState{W, T},
                                         time::Real, dt::T) where {D, W, T}
    _is_moving(pg) && throw(ArgumentError("BlockTimeStepper: static grids only (no grid mover, no moving pieces)"))
    N  = pg.meta.N
    nb = bs.nb
    nb == num_blocks(eq) || throw(ArgumentError("BlockTimeStepper: the state has $nb blocks, the equation $(num_blocks(eq))"))
    bs.N[] == N || throw(ArgumentError("BlockTimeStepper: the state was set for $(bs.N[]) particles, the grid has $N"))
    inner   = ts.inner
    s       = inner.num_stages
    bt      = inner.tableau
    interp  = inner.divergence_interpolator
    backend = get_backend()
    isb     = pg.core.is_boundary

    update_size!(inner, nb * N, length(pg.neighbor.indices), false)
    update_size!(interp, N)
    copyto!(parent(inner.rho_n), 1, parent(bs.data), 1, nb * N)
    fill!(view(pg.core.particle_orders, 1:N), _extract_order(interp))

    for i in 1:s
        t_i   = time + bt.c_t[i] * dt
        Y_i   = parent(inner.Y_stages[i])
        for b in 1:nb
            _block_candidate!(ts, pg, b, N, i, dt)
        end
        if !isempty(ts.implicit_sources)
            a_ii = bt.a[i, i]
            if abs(a_ii) > T(1e-14)
                for (n, src) in enumerate(ts.implicit_sources)
                    reduce_blocks!(src, Y_i, N, nb, pg, t_i, dt * a_ii)
                    for b in 1:nb
                        foreach_particle(_block_implicit_step!, backend, N,
                            (_blk(inner.Y_stages[i], b, N), _blk(inner.K_I_stages[i], b, N), src.params, src.term, b, isb,
                             dt * a_ii, inv(dt * a_ii), pg, t_i, n == 1))
                    end
                end
            else
                for (n, src) in enumerate(ts.implicit_sources)
                    reduce_blocks!(src, Y_i, N, nb, pg, t_i, zero(T))
                    for b in 1:nb
                        foreach_particle(_block_implicit_rhs!, backend, N,
                            (_blk(inner.K_I_stages[i], b, N), _blk(inner.Y_stages[i], b, N), src.params, src.term, b, isb, pg, t_i, n == 1))
                    end
                end
            end
        end
        for b in 1:nb
            apply_boundary_conditions!(pg, _blk(inner.Y_stages[i], b, N), inner, BlockOf(eq, b), t_i, _blk(inner.rho_n, b, N))
        end
        for b in 1:nb
            compute_explicit_rhs!(inner, BlockOf(eq, b), pg, _blk(inner.K_E_stages[i], b, N), inner.K_X_stages[i],
                                  inner.K_V_stages[i], inner.V_stages[i], _blk(inner.Y_stages[i], b, N), t_i, true)
        end
        for src in ts.explicit_sources
            reduce_blocks!(src, Y_i, N, nb, pg, t_i, zero(T))
            for b in 1:nb
                foreach_particle(_block_explicit_source!, backend, N,
                    (_blk(inner.K_E_stages[i], b, N), _blk(inner.Y_stages[i], b, N), src.params, src.term, b, isb, pg, t_i))
            end
        end
    end
    for b in 1:nb
        _block_candidate!(ts, pg, b, N, s + 1, dt)
        apply_boundary_conditions!(pg, _blk(inner.rho_cand, b, N), inner, BlockOf(eq, b), time + dt, _blk(inner.rho_n, b, N))
    end
    copyto!(parent(bs.data), 1, parent(inner.rho_cand), 1, nb * N)
    return nothing
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
    foreach_particle(_stage_predictor!, get_backend(), N,
        (target, (inner.x_cand, inner.v_cand), U_n, (inner.x_n, inner.v_n), K_E, K_I, (inner.K_X_stages, inner.K_V_stages, inner.V_stages),
         U_n, pg.core.is_boundary, inner.tableau, i, is_final ? s : i - 1, is_final, dt, false, false, zero(T), zero(Space{D, T})))
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
