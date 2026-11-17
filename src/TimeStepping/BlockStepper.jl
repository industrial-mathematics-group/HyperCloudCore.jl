export BlockState, BlockTimeStepper, BlockOf, num_blocks, block_flux, block_max_eigenvalue, block_view, set_block_states!

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
    BlockTimeStepper(eq, div_interp, tableau; storage = Array, kwargs...)
    (ts::BlockTimeStepper)(eq, pg, bs::BlockState, time, dt)

The time integrator of the block pipeline: the stage kernels of [`GeneralRKTimeStepper`](@ref) run block by
block on the views of a [`BlockState`](@ref), with the equation wrapped as [`BlockOf`](@ref)`(eq, b)`, so the
grid, the neighbour search, the MLS weights, the scheme and its pre-pass are shared by all blocks and the
state length `nb · W` is a run-time number. The stepper owns a `GeneralRKTimeStepper` for `W` components
(`inner`: the interpolator, buffers and options; its stage buffers are sized to `nb · N`), and the `kwargs` are
its options.

Scope (for now): explicit tableaus, static grids (no mover, no moving pieces), no MOOD, no source terms; the
boundary conditions of the grid apply per block, with the block's state of the step start as the reference.
For one block (`num_blocks(eq) == 1`) the result is bit-identical to `GeneralRKTimeStepper`.
"""
struct BlockTimeStepper{D, W, T, EQ <: HyperbolicPDE{D, W, T}, TS <: GeneralRKTimeStepper{D, W, T}} <: TimeStepper
    pde::EQ
    inner::TS
end

function BlockTimeStepper(eq::HyperbolicPDE{D, W, T}, div_interp::DivergenceInterpolator, tableau::IMEXButcherTableau{T};
                          kwargs...) where {D, W, T}
    (all(iszero, tableau.a) && all(iszero, tableau.b)) ||
        throw(ArgumentError("BlockTimeStepper: explicit tableaus only (the implicit part of the tableau is not zero)"))
    inner = GeneralRKTimeStepper(BlockOf(eq, 1), div_interp, MOOD(NoMOOD()), tableau; kwargs...)
    return BlockTimeStepper{D, W, T, typeof(eq), typeof(inner)}(eq, inner)
end

@inline _blk(g::Growable, b::Int, N::Int) = view(parent(g), ((b - 1) * N + 1):(b * N))

function (ts::BlockTimeStepper{D, W, T})(eq::HyperbolicPDE{D, W, T}, pg::ParticleGrid{D, W, T}, bs::BlockState{W, T},
                                         time::Real, dt::T) where {D, W, T}
    _is_moving(pg) && throw(ArgumentError("BlockTimeStepper: static grids only (no grid mover, no moving pieces)"))
    N  = pg.meta.N
    nb = bs.nb
    nb == num_blocks(eq) || throw(ArgumentError("BlockTimeStepper: the state has $nb blocks, the equation $(num_blocks(eq))"))
    bs.N[] == N || throw(ArgumentError("BlockTimeStepper: the state was set for $(bs.N[]) particles, the grid has $N"))
    inner  = ts.inner
    s      = inner.num_stages
    bt     = inner.tableau
    interp = inner.divergence_interpolator

    update_size!(inner, nb * N, length(pg.neighbor.indices), false)
    update_size!(interp, N)
    copyto!(parent(inner.rho_n), 1, parent(bs.data), 1, nb * N)
    fill!(view(pg.core.particle_orders, 1:N), _extract_order(interp))

    for i in 1:s
        t_i = time + bt.c_t[i] * dt
        for b in 1:nb
            _block_candidate!(ts, pg, b, N, i, dt)
            apply_boundary_conditions!(pg, _blk(inner.Y_stages[i], b, N), inner, BlockOf(eq, b), t_i, _blk(inner.rho_n, b, N))
        end
        for b in 1:nb
            compute_explicit_rhs!(inner, BlockOf(eq, b), pg, _blk(inner.K_E_stages[i], b, N), inner.K_X_stages[i],
                                  inner.K_V_stages[i], inner.V_stages[i], _blk(inner.Y_stages[i], b, N), t_i, true)
        end
    end
    for b in 1:nb
        _block_candidate!(ts, pg, b, N, s + 1, dt)
        apply_boundary_conditions!(pg, _blk(inner.rho_cand, b, N), inner, BlockOf(eq, b), time + dt, _blk(inner.rho_n, b, N))
    end
    copyto!(parent(bs.data), 1, parent(inner.rho_cand), 1, nb * N)
    return nothing
end

# The explicit predictor of stage `i` (the final combination for `i == s + 1`) for block `b`: the stage kernel of
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
