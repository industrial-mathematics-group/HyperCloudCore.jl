export GeneralRKTimeStepper, GeneralIMEXTimeStepper, mood_flags, mood_inadmissible

"""
    MOODWork(storage = Array)

Scratch owned by the time stepper: `triggered` (this pass), `changed` (orders that moved, i.e. C),
`dirty` (fluxes to recompute, C ⊕ reach), `scratch`, an `Int` order snapshot for forcing, and
`triggered_step`, a per-step diagnostic of every particle that dropped an order (see `mood_flags`).
The masks are `Growable` buffers with arrays of type `storage`. `inadmissible` counts the interior candidates
accepted while not admissible in the last step (see `mood_inadmissible`), `warned` whether that was reported.
"""
struct MOODWork{GB <: Growable{Bool}, GI <: Growable{Int}}
    triggered::GB
    changed::GB
    dirty::GB
    scratch::GB
    snapshot::GI
    triggered_step::GB
    N::Base.RefValue{Int}           # particles of the last `update_size!`: the active part of the masks
    inadmissible::Base.RefValue{Int}
    warned::Base.RefValue{Bool}
end
MOODWork(::Type{AT} = Array) where {AT <: AbstractArray} =
    MOODWork(Growable{Bool}(AT), Growable{Bool}(AT), Growable{Bool}(AT), Growable{Bool}(AT), Growable{Int}(AT), Growable{Bool}(AT), Ref(0),
             Ref(0), Ref(false))

function update_size!(w::MOODWork, N::Int)
    ensure_capacity!(w.triggered, N); ensure_capacity!(w.changed, N)
    ensure_capacity!(w.dirty, N);     ensure_capacity!(w.scratch, N)
    ensure_capacity!(w.snapshot, N);  ensure_capacity!(w.triggered_step, N)
    w.N[] = N
    return nothing
end

"""
    InteractionBuffer{D, M, T}(storage = Array)

Per-slot scratch of the schemes, indexed by global neighbour slot (so per-particle slices are
disjoint and the passes are thread-safe). The host object holds `Growable` buffers with arrays of
type `storage`; kernels receive it with the raw arrays (see `DeviceAdapt.jl`). Neighbour states are
not buffered: schemes read them from the state vector through `neighbour_states` /
`neighbour_differences`.

# Fields
- `df_flux`: numerical flux differences / non-conservative jumps.
- `df_scratch`: MLS scratch.
- `mask`: directional stencil filter.
"""
struct InteractionBuffer{D, M, T, VS <: AbstractVector{State{M, T}}, VF <: AbstractVector{Flux{D, M, T}}, VB <: AbstractVector{Bool}}
    df_flux::VF
    df_scratch::VS
    mask::VB
end
InteractionBuffer{D, M, T}(df_flux::VF, df_scratch::VS, mask::VB) where {D, M, T, VS, VF, VB} =
    InteractionBuffer{D, M, T, VS, VF, VB}(df_flux, df_scratch, mask)
InteractionBuffer{D, M, T}(::Type{AT} = Array) where {D, M, T, AT <: AbstractArray} =
    InteractionBuffer{D, M, T}(Growable{Flux{D, M, T}}(AT), Growable{State{M, T}}(AT), Growable{Bool}(AT))

function update_size!(ib::InteractionBuffer, num_interactions::Int)
    ensure_capacity!(ib.df_flux, num_interactions)
    ensure_capacity!(ib.df_scratch, num_interactions)
    ensure_capacity!(ib.mask, num_interactions)
    return nothing
end

# =========================================================================
# SOURCE TUPLE UNROLLERS
# =========================================================================

@inline @generated function evaluate_sources(sts::Tuple{Vararg{AbstractSourceTerm}}, U, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    N == 0 && return :(zero(U))
    expr = :(evaluate_source(sts[1], U, p_idx, pg, t))
    for i in 2:N
        expr = :($expr + evaluate_source(sts[$i], U, p_idx, pg, t))
    end
    return expr
end

@inline function pre_solve_updates!(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, Y_stage, pg, t::Real)
    for st in sts
        pre_solve_update!(st, Y_stage, pg, t)
    end
end

@inline @generated function implicit_solve(sts::Tuple{Vararg{AbstractImplicitSourceTerm}}, U_in, dt_coeff::Real, p_idx::Int, pg, t::Real)
    N = length(sts.parameters)
    N == 0 && return :(U_in)
    quote
        U_out = U_in
        Base.Cartesian.@nexprs $N i -> U_out = implicit_solve(sts[i], U_out, dt_coeff, p_idx, pg, t)
        return U_out
    end
end

include("ButcherTableaus.jl")
include("MOOD.jl")
include("GridMovers.jl")

# =========================================================================
# THE TIME STEPPER
# =========================================================================
"""
    GeneralRKTimeStepper{D, M, T, PDE, G, MO, EST, IST} <: TimeStepper
    GeneralRKTimeStepper(pde, div_interp, mood, sources::Tuple, tableau; kwargs...)
    GeneralRKTimeStepper(pde, div_interp, mood, tableau; kwargs...)
    (ts::GeneralRKTimeStepper)(eq, pg, time, dt::T, U = pg.rhos)

The one time integrator: explicit RK and IMEX share this struct, since an explicit tableau is an
`IMEXButcherTableau` with a zero implicit part (see `RKButcherTableau`). `sources` may mix explicit
and implicit terms. With an explicit tableau (`a = 0` and `b = 0`), implicit terms are applied explicitly (their `evaluate_source`
at every stage, weighted like the flux divergence; the constructor warns), since there is no implicit part
to solve them with; a tableau whose implicit part has a zero diagonal (e.g. `a = 0`, `b ≠ 0`) is kept as given and
also evaluates them explicitly, with a warning. Several implicit sources are solved one after another per stage
(`U → implicit_solve(S₁, U) → implicit_solve(S₂, ·) …`), a first-order splitting inside the stage: the stage
relation `Y = U + Δt a_ii ΣS(Y)` holds only for a single implicit source (or commuting ones). If `pg.mover` is
not `NoGridMover`, particle positions are integrated as additional stage variables of the same tableau (see
`GridMovers.jl`).

# Algorithm (per time step)
Orders start at the interpolator's maximum consistency order. For each stage `i`:
1. Form the stage state `Y_i` (explicit predictor, implicit diagonal solve, `K_I[i]`, BCs) and,
   when moving, the stage positions `X_i`.
2. For `i > 1`, run the **stage-wise MOOD fixed point** against `Y_{i-1}` at geometry `X_{i-1}`:
   flagged particles drop an order, only the affected part of `K_E[i-1]` (set `C ⊕ _order_reach`)
   is recomputed, `Y_i` is re-formed, until nothing is flagged.
3. Switch the grid geometry to `X_i`, then compute `K_E[i]` (and `K_X[i] = w`) for all interior
   particles.
The final combination is checked the same way against `Y_s`; the final positions are written back
and a full neighbour search is run.

What MOOD checks at checkpoint `i`: for an explicit SSP tableau (SSP coefficient `C > 0`, see
`ssp_coefficient`) the forward Euler substep `W_{i-1} = Y_{i-1} + (Δt/C) K_E[i-1]` of the canonical
Shu–Osher form, in which every stage and the final update are convex combinations of `Uⁿ` and these
substeps. A checked (DMP, PAD) substep of every stage therefore makes every stage and the result a
convex combination of checked states, for any SSP method; checking `Y_i` instead only does so when
every substep enters a single stage (Euler, SSP-RK2, SSP-RK3), not for SSPRK(5,4), whose final update
reuses the substeps of stages 2 and 4. For other tableaus (`C = 0`, IMEX) the candidate `Y_i` itself
is checked (no theoretical backing). `mood_substeps = false` restores that everywhere.

# Keyword options
- `reset_orders_per_stage = false`: restore the maximum order before each stage's full RHS pass.
- `full_search_per_stage = false`: rebuild the neighbour topology at every stage position
  (default: freeze topology, refresh distances/weights only).
- `volume_estimator = ShepardVolumes()`: how `pg.core.volumes` is recomputed after each step of a
  moving grid (`nothing` leaves the volumes untouched).
- `storage = Array`: array type of the stage buffers, e.g. `ROCArray` for a stepper that advances a
  grid built with `device_grid` (the interpolator needs the same `storage`).
- `mood_substeps = true`: check the forward Euler substeps of an SSP tableau (see above) instead of
  the stage candidates.

# Fields
- `rho_n`, `rho_cand`, `Y_stages`, `K_E_stages`, `K_I_stages`: states and RHS (`K_E = -div(F − w_i ρ) + S_expl`).
- `rho_sub`, `substep_ratio`: the Euler substep MOOD checks and its step `Δt/C` per `Δt` (`0`: the
  stage candidates are checked; `rho_sub` is then never sized).
- `x_n`, `x_cand`, `X_stages`, `K_X_stages`: positions and grid velocities (sized only when moving).
- `v_n`, `v_cand`, `V_stages`, `K_V_stages`: the particle volumes the scheme's own volume rate implies (`V div w`,
  written by the Lagrangian frame; sized only when moving). At the end of a step the content `V U` is remapped onto
  the re-estimated volumes (`_content_remap!`), which makes the Lagrangian mass conservation exact up to the time
  discretisation and the flux pair defect.
- `int_buffer`, `work`: neighbour gather buffer and MOOD scratch.
All of them are `Growable` buffers, sized by `update_size!` at the start of every step.

The step advances the state array `U` (default: the grid's own `pg.rhos`) in place; the boundary conditions see the
state of the step start through the stepper's copy, never through `pg.rhos`. A separate array needs a static grid
(the moving-grid path remaps and sweeps the grid's own state).
"""
struct GeneralRKTimeStepper{D, M, T, PDE <: HyperbolicPDE, G <: DivergenceInterpolator, MO <: MOOD,
                            EST <: Tuple{Vararg{AbstractExplicitSourceTerm}},
                            IST <: Tuple{Vararg{AbstractImplicitSourceTerm}},
                            VE <: Union{Nothing, VolumeEstimator}, S,
                            GS <: Growable{State{M, T}}, GP <: Growable{Space{D, T}}, GT <: Growable{T},
                            IB <: InteractionBuffer{D, M, T}, WORK <: MOODWork,
                            BT <: IMEXButcherTableau{T}} <: TimeStepper
    pde::PDE
    divergence_interpolator::G
    mood::MO
    explicit_sources::EST
    implicit_sources::IST
    tableau::BT          # static form (`static_tableau`): a bits kernel argument of concrete type
    reset_orders_per_stage::Bool
    full_search_per_stage::Bool
    volume_estimator::VE
    substep_ratio::T     # Δt/C per Δt of the Euler substeps MOOD checks; 0: check the stage candidates

    rho_n::GS
    rho_cand::GS
    rho_sub::GS
    # Stage arrays as tuples (one entry per stage): indexable inside kernels, unlike a Vector of Vectors.
    Y_stages::NTuple{S, GS}
    K_E_stages::NTuple{S, GS}
    K_I_stages::NTuple{S, GS}

    x_n::GP
    x_cand::GP
    X_stages::NTuple{S, GP}
    K_X_stages::NTuple{S, GP}

    v_n::GT
    v_cand::GT
    V_stages::NTuple{S, GT}
    K_V_stages::NTuple{S, GT}

    int_buffer::IB
    work::WORK
    num_stages::Int

    function GeneralRKTimeStepper(
        pde::HyperbolicPDE{D, M, T}, div_interp::G, mood::MO,
        explicit_sources::EST, implicit_sources::IST, tableau::IMEXButcherTableau{T};
        reset_orders_per_stage::Bool = false, full_search_per_stage::Bool = false,
        volume_estimator::VE = ShepardVolumes(), storage::Type{AT} = Array, mood_substeps::Bool = true
    ) where {D, M, T, G, MO <: MOOD, EST <: Tuple{Vararg{AbstractExplicitSourceTerm}}, IST <: Tuple{Vararg{AbstractImplicitSourceTerm}},
             VE <: Union{Nothing, VolumeEstimator}, AT <: AbstractArray}
        s  = num_stages(tableau)
        C  = ssp_coefficient(tableau)
        if !isempty(implicit_sources) && all(iszero, tableau.a) && all(iszero, tableau.b)   # no implicit part
            @warn "Implicit source terms given with a purely explicit tableau: they are applied explicitly, with the " *
                  "explicit weights of the tableau (no implicit solve)."
            tableau = _explicit_implicit_part(tableau)
        elseif !isempty(implicit_sources) && all(i -> iszero(tableau.a[i, i]), 1:s)
            # e.g. a hand-made `a = 0`, `b ≠ 0`: kept as given, but no stage solves anything either
            @warn "Implicit source terms given with a tableau whose implicit part has a zero diagonal: they are " *
                  "evaluated explicitly, with the implicit weights of the tableau (no implicit solve)."
        end
        bt = static_tableau(tableau, Val(s))
        states    = () -> Growable{State{M, T}}(AT)
        positions = () -> Growable{Space{D, T}}(AT)
        volumes   = () -> Growable{T}(AT)
        ib   = InteractionBuffer{D, M, T}(AT)
        work = MOODWork(AT)
        ratio = (mood_substeps && C > 0) ? T(1 / C) : zero(T)
        new{D, M, T, typeof(pde), G, MO, EST, IST, VE, s, typeof(states()), typeof(positions()), typeof(volumes()), typeof(ib), typeof(work), typeof(bt)}(
            pde, div_interp, mood, explicit_sources, implicit_sources, bt,
            reset_orders_per_stage, full_search_per_stage, volume_estimator, ratio,
            states(), states(), states(),
            ntuple(_ -> states(), s), ntuple(_ -> states(), s), ntuple(_ -> states(), s),
            positions(), positions(),
            ntuple(_ -> positions(), s), ntuple(_ -> positions(), s),
            volumes(), volumes(),
            ntuple(_ -> volumes(), s), ntuple(_ -> volumes(), s),
            ib, work, s
        )
    end
end

const GeneralIMEXTimeStepper = GeneralRKTimeStepper

# An explicit tableau (`a = 0`, `b = 0`) has no implicit part to carry implicit sources. With the explicit
# weights copied into the implicit part, the stepper evaluates `K_I[i] = S(Y_i)` without a solve
# (`a[i, i] = 0`) and weights it like `K_E[i]`: the explicit RK method applied to `F + S`.
_explicit_implicit_part(bt::IMEXButcherTableau) =
    IMEXButcherTableau(copy(bt.a_t), copy(bt.a_t), copy(bt.c_t), copy(bt.c_t), copy(bt.b_t), copy(bt.b_t))

# Mixed source tuple → sorted
function GeneralRKTimeStepper(pde::HyperbolicPDE{D, M, T}, div_interp, mood::MOOD,
                              all_sources::Tuple{Vararg{AbstractSourceTerm}}, tableau::IMEXButcherTableau{T};
                              kwargs...) where {D, M, T}
    explicit_sts = filter(st -> st isa AbstractExplicitSourceTerm, all_sources)
    implicit_sts = filter(st -> st isa AbstractImplicitSourceTerm, all_sources)
    return GeneralRKTimeStepper(pde, div_interp, mood, explicit_sts, implicit_sts, tableau; kwargs...)
end

# No sources
GeneralRKTimeStepper(pde::HyperbolicPDE{D, M, T}, div_interp, mood::MOOD, tableau::IMEXButcherTableau{T}; kwargs...) where {D, M, T} =
    GeneralRKTimeStepper(pde, div_interp, mood, (), (), tableau; kwargs...)

"""
    mood_flags(ts::GeneralRKTimeStepper) -> AbstractVector{Bool}

Per-step diagnostic: `true` for every particle that dropped an order during the last step. A view of the
stepper's mask over the particles of that step (the buffer behind it is over-allocated).
"""
mood_flags(ts::GeneralRKTimeStepper) = view(parent(ts.work.triggered_step), 1:ts.work.N[])

"""
    mood_inadmissible(ts::GeneralRKTimeStepper) -> Int

Per-step diagnostic: the number of interior candidates (summed over the checkpoints of the last step; with an SSP
tableau the checked candidates are the Euler substeps) that MOOD accepted although they are not admissible
(`is_admissible`, e.g. a negative pressure or a NaN). That happens only where the order cannot drop further
(order 1): the first-order update itself is not admissible there, because the step is above its positivity limit
or because the first-order scheme is not positive on the cloud (e.g. the geometric Rusanov dissipation on an
irregular cloud; `AlignedRusanovFlux` is local-extremum-diminishing). The run goes on with the inadmissible
(sub)step, which often ends it a few steps later. The stepper warns the first time it happens; the count is `0`
without a MOOD criterion (`NoMOOD` checks nothing).
"""
mood_inadmissible(ts::GeneralRKTimeStepper) = ts.work.inadmissible[]

# The moving-grid path (position stages, a neighbour search per step) also serves a static cloud whose boundary
# pieces move: their ghost particles are placed every stage (`place_ghosts!`), the interior keeps `grid_velocity` 0.
@inline _is_moving(pg::ParticleGrid) = !(pg.mover isa NoGridMover) || has_moving_pieces(pg.geometry)

# A step starts with the ghosts of the moving pieces and the region at the step's time: a grid is generated at the
# pieces' poses at t = 0, so a run that starts later (or resumes) first places the ghosts and sweeps the cloud to
# `t` (the ghosts used to move from `t` by one step only and stay behind their piece for the rest of the run).
# Moving pieces need a static interior: the volume estimators integrate the region at t = 0, so a grid mover's
# re-estimated volumes would lose what the pieces uncovered.
function _sync_boundaries!(pg::ParticleGrid{D, M, T}, t::T, max_order::Int) where {D, M, T}
    pg.mover isa NoGridMover ||
        throw(ArgumentError("moving pieces need a static interior cloud (NoGridMover), got $(nameof(typeof(pg.mover))): " *
                            "the volume estimators integrate the region at t = 0"))
    tol = 8 * eps(T) * max(one(T), abs(t))
    (abs(pg.meta.boundary_time - t) <= tol && abs(pg.meta.sweep_time - t) <= tol) && return nothing
    # In sub-intervals over which no piece moves more than a quarter spacing (the sweep's clearance): one sweep over
    # a long way lost the uncovered region farther than R from every particle (Σ V 1.9535 for 1.96 after 6 spacings)
    t0 = pg.meta.boundary_time
    n  = _sync_substeps(pg, t0, t)
    for k in 1:n
        tk = k == n ? t : t0 + (t - t0) * T(k) / T(n)
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, tk)
        pg.neighbor(pg)
        sweep_boundaries!(pg, pg.meta.sweep_time, tk, max_order)
    end
    return nothing
end

# Number of equal sub-intervals of [t0, t] over each of which every piece moves at most a quarter spacing: the
# displacement of the corners of the piece's box in its body frame (the canvas corners for an unbounded piece; with
# the canvas corners always, a ball spinning in place needed 4096 sub-steps to start at t = 5)
function _sync_substeps(pg::ParticleGrid{D, M, T}, t0::T, t::T) where {D, M, T}
    geom, cd = pg.geometry, pg.domain
    lim = minimum(pg.meta.dx) / 4
    box_corners(lo, hi) = [Space{D, T}(ntuple(d -> I[d] == 0 ? lo[d] : hi[d], Val(D))) for I in CartesianIndices(ntuple(_ -> 0:1, Val(D)))]
    body = map(geom.pieces) do piece
        lo, hi = bounding_box(piece.shape)
        all(isfinite, lo) && all(isfinite, hi) ? (box_corners(lo, hi), true) : (box_corners(cd.canvas_mins, cd.canvas_maxs), false)
    end
    function moves(ta, tb)
        dmax = zero(T)
        for (piece, (pts, in_body)) in zip(geom.pieces, body)
            ma, mb = pose(piece, ta), pose(piece, tb)
            for x in pts
                d = in_body ? HyperCloudCore.body_to_world(mb, x) - HyperCloudCore.body_to_world(ma, x) :
                              mb.c + mb.R * (ma.R' * (x - ma.c)) - x
                dmax = max(dmax, sqrt(sum(abs2, d)))
            end
        end
        return dmax
    end
    n = 1
    while n < 2^20 && any(k -> moves(t0 + (t - t0) * T(k - 1) / T(n), t0 + (t - t0) * T(k) / T(n)) > lim, 1:n)
        n *= 2
    end
    return n
end

function update_size!(ts::GeneralRKTimeStepper, N::Int, M_neighbors::Int, moving::Bool)
    ensure_capacity!(ts.rho_n, N)
    ensure_capacity!(ts.rho_cand, N)
    _checks_substeps(ts) && ensure_capacity!(ts.rho_sub, N)
    for i in 1:ts.num_stages
        ensure_capacity!(ts.Y_stages[i], N)
        ensure_capacity!(ts.K_E_stages[i], N)
        ensure_capacity!(ts.K_I_stages[i], N)
    end
    if moving
        ensure_capacity!(ts.x_n, N)
        ensure_capacity!(ts.x_cand, N)
        ensure_capacity!(ts.v_n, N)
        ensure_capacity!(ts.v_cand, N)
        for i in 1:ts.num_stages
            ensure_capacity!(ts.X_stages[i], N)
            ensure_capacity!(ts.K_X_stages[i], N)
            ensure_capacity!(ts.V_stages[i], N)
            ensure_capacity!(ts.K_V_stages[i], N)
        end
    end
    uses_slot_buffers(ts.divergence_interpolator) && update_size!(ts.int_buffer, M_neighbors)
    update_size!(ts.work, N)
    return nothing
end
update_size!(ts::GeneralRKTimeStepper, N::Int, M_neighbors::Int) = update_size!(ts, N, M_neighbors, true)

# =========================================================================
# BUILDING BLOCKS
# =========================================================================

"""
    compute_explicit_rhs!(ts, eq, pg, K_E, K_X, K_V, V_k, Y, t, full)

Scheme pre-pass (cached gradients) and flux pass at the CURRENT grid geometry:
`K_E = -div(F − w ρ) + S_expl` (the PDE wrapped in the moving frame, `_moving_frame`), `K_X = w` when the grid
moves, and `K_V = V div w` (the volume rate the Lagrangian frame writes for the stage volumes `V_k`). The passes
read neighbour states straight from `Y`.
`full = true` covers all interior particles; otherwise gradients are recomputed on `work.changed`
and fluxes on `work.dirty`.
"""
# The signature up to 0.9.1 (the volume buffers of the first stage; untouched on a static grid)
compute_explicit_rhs!(ts::GeneralRKTimeStepper, eq, pg, K_E, K_X, Y, t, full::Bool) =
    compute_explicit_rhs!(ts, eq, pg, K_E, K_X, ts.K_V_stages[1], ts.V_stages[1], Y, t, full)
function compute_explicit_rhs!(ts::GeneralRKTimeStepper{D, M, T}, eq, pg, K_E, K_X, K_V, V_k, Y, t, full::Bool) where {D, M, T}
    N       = pg.meta.N
    interp  = ts.divergence_interpolator
    moving  = _is_moving(pg)
    backend = get_backend()

    prepare_geometry!(interp, pg)   # refreshes geometry caches only if the grid changed

    foreach_particle(_rhs_gather!, backend, N,
        (Y, pg.core.is_boundary, pg.neighbor.ranges, ts.int_buffer, interp, pg, ts.work.changed, full))
    foreach_particle(_rhs_flux!, backend, N,
        (eq, Y, K_E, K_X, K_V, V_k, pg.core.is_boundary, pg.neighbor.ranges, ts.int_buffer, interp, ts.mood.strategy, pg.mover, pg,
         ts.explicit_sources, ts.work.dirty, full, moving, t, zero(State{M, T}), zero(Space{D, T})))   # (20 arguments)
    return nothing
end

# Kernel bodies of the right-hand side (see Backend.jl): the scheme's pre-pass (cached gradients), then the flux pass.
# Boundary particles take part in the full gather: their reconstruction gradients enter the interface states of
# every wall pair. Skipped (gradient zero), the ghost side of a wall pair was piecewise constant and the wall rows
# first order, and a condition that mirrors the numerical solution fed that error back (a channel wall, MUSCL3,
# N = 80/160, t = 1: exact ghosts 1.5e-2/7.8e-3 and MirrorBC 3.8e-2/2.9e-2; gathered: 1.6e-2/2.4e-3 and
# 2.0e-2/2.6e-3). A partial pass (MOOD re-form, same states, changed orders) does not need them: ghost orders
# never change.
@inline function _rhs_gather!(p, Y, is_boundary, nb_slices, ib, interp, pg, grad_mask, full)
    (full || grad_mask[p]) || return nothing
    fi = Y[p]; nb = nb_slices[p]
    update_content!(interp, p, fi, nb, pg, Y, ib)
    return nothing
end

@inline function _rhs_flux!(p, eq, Y, K_E, K_X, K_V, V_k, is_boundary, nb_slices, ib, interp, strategy, mover, pg, sources,
                            flux_mask, full, moving, t, zero_state, zero_space)
    if is_boundary[p]
        if full
            K_E[p] = zero_state
            moving && (K_X[p] = zero_space; K_V[p] = zero(eltype(K_V)))
        end
        return nothing
    end
    (full || flux_mask[p]) || return nothing
    fi = Y[p]; nb = nb_slices[p]
    if moving
        w      = grid_velocity(mover, eq, p, fi, pg, t)
        K_X[p] = w
        K_V[p] = zero(eltype(K_V))                  # (the Lagrangian frame overwrites it with V div w)
        div_F  = interp(_moving_frame(interp, mover, eq, w, K_V, V_k), p, fi, nb, pg, Y, ib, strategy)   # F − w ρ in the moving frame
    else
        div_F  = interp(eq, p, fi, nb, pg, Y, ib, strategy)
    end
    K_E[p] = -div_F + evaluate_sources(sources, fi, p, pg, t)
    return nothing
end

"""
    form_candidate!(ts, eq, pg, i, dt, time) -> (target, t_i)

For `i ≤ s`: builds `Y_i` (explicit predictor, implicit diagonal solve, `K_I[i]`, BCs) and `X_i`.
For `i == s + 1`: builds the final state into `rho_cand` / `x_cand` (BCs applied).
Positions are NOT applied to the grid here; see `set_positions!`.
"""
function form_candidate!(ts::GeneralRKTimeStepper{D, M, T}, eq, pg, i::Int, dt::T, time::Real) where {D, M, T}
    N        = pg.meta.N
    s        = ts.num_stages
    bt       = ts.tableau
    moving   = _is_moving(pg)
    backend  = get_backend()

    is_final = i > s
    target   = is_final ? ts.rho_cand : ts.Y_stages[i]
    x_target = is_final ? ts.x_cand   : ts.X_stages[i]
    v_target = is_final ? ts.v_cand   : ts.V_stages[i]
    t_i      = is_final ? time + dt : time + bt.c_t[i] * dt
    n_prev   = is_final ? s : i - 1

    # (boundary particles keep the state of the step start, `rho_n`: the stepper's copy, not `pg.rhos`, so that the
    # stage can be formed for any state array)
    foreach_particle(_stage_predictor!, backend, N,
        (target, (x_target, v_target), ts.rho_n, (ts.x_n, ts.v_n), ts.K_E_stages, ts.K_I_stages, (ts.K_X_stages, ts.K_V_stages, ts.V_stages),
         ts.rho_n, pg.core.is_boundary, bt, i, n_prev, is_final, dt, moving, _content_form(ts, pg), zero(T), zero(Space{D, T})))

    if !is_final
        pre_solve_updates!(ts.implicit_sources, parent(target), pg, t_i)
        a_ii = bt.a[i, i]
        if abs(a_ii) > T(1e-14)
            # solve and its right-hand side in one pass: S(U_out) = (U_out - U_in) / (dt a_ii)
            foreach_particle(_stage_implicit_step!, backend, N,
                (target, ts.K_I_stages[i], pg.core.is_boundary, ts.implicit_sources, dt * a_ii, inv(dt * a_ii), pg, t_i, zero(State{M, T})))
        else
            foreach_particle(_stage_implicit_rhs!, backend, N,
                (ts.K_I_stages[i], target, pg.core.is_boundary, ts.implicit_sources, pg, t_i, zero(State{M, T})))
        end
    end

    apply_boundary_conditions!(pg, parent(target), ts, eq, t_i, parent(ts.rho_n))
    return target, t_i
end

# Kernel bodies of the stage assembly (see Backend.jl).
# `content`: the stages carry the content V U of an interior particle (a Lagrangian frame tracking its volume rate): the
# stage state is (V_n U_n + Δt Σ a_j V_j K_j) / V_i with the stage volumes V_j of the earlier stages and V_i of this one,
# so that Σ V U is conserved exactly up to the pair defect of the fluxes (the point-value form, U with the volume term
# −U div w and V evolved beside it, left a product-rule error of the two RK sequences at steep fronts).
@inline function _stage_predictor!(p, target, xv_target, U_n, xv_n, K_E, K_I, K_XVV, rhos, is_boundary, bt, i, n_prev,
                                   is_final, dt, moving, content, zero_T, zero_space)
    x_target, v_target = xv_target          # positions and volumes move together (packed: 20 arguments at most)
    x_n, v_n = xv_n
    K_X, K_V, V_st = K_XVV
    if is_boundary[p]
        target[p] = rhos[p]
        moving && (x_target[p] = x_n[p]; v_target[p] = v_n[p])
        return nothing
    end
    X = moving ? x_n[p] : zero_space
    V = moving ? v_n[p] : zero_T
    Y = content ? V * U_n[p] : U_n[p]       # the content, or the point value
    for j in 1:n_prev
        a_t = is_final ? bt.b_t[j] : bt.a_t[i, j]
        a   = is_final ? bt.b[j]   : bt.a[i, j]
        Vj  = content ? (@inbounds V_st[j][p]) : one(zero_T)
        if a_t != zero_T
            Y += (dt * a_t * Vj) * K_E[j][p]
            moving && (X += (dt * a_t) * K_X[j][p]; V += (dt * a_t) * K_V[j][p])
        end
        a != zero_T && (Y += (dt * a * Vj) * K_I[j][p])
    end
    target[p] = content ? Y / V : Y
    moving && (x_target[p] = X; v_target[p] = V)
    return nothing
end

# The stepper runs in content form on a moving interior cloud whose frame tracks the volume rate (the Lagrangian movers
# with the MUSCL scheme); the volumes are remapped at the end of the step as well (`_content_remap!`)
@inline _content_form(ts::GeneralRKTimeStepper, pg::ParticleGrid) =
    !(pg.mover isa NoGridMover) && _conservative_ale(ts.divergence_interpolator) && _lagrangian_type(pg.mover)

# Kernel body (see Backend.jl): the content `V U` of an interior particle, carried with the volumes the scheme's own
# volume rate implies (`v_ev`), is put onto the re-estimated volume `V`
@inline function _content_remap!(p, rhos, v_ev, V, is_boundary)
    @inbounds begin
        (is_boundary[p] || !(V[p] > 0)) && return nothing
        rhos[p] = rhos[p] * (v_ev[p] / V[p])
    end
    return nothing
end

# Implicit stage with a diagonal entry: the solve U_out - dt a_ii S(U_out) = U_in and its right-hand side
# S(U_out) = (U_out - U_in) / (dt a_ii), which spares the second evaluation of the source (for the BGK
# model that is a second discrete Maxwellian per stage). With several implicit sources, which are applied
# one after another, the RHS is still evaluated at U_out, as before.
@inline function _stage_implicit_step!(p, target, K_I_i, is_boundary, sources, dt_ii, inv_dt_ii, pg, t_i, zero_state)
    if is_boundary[p]
        K_I_i[p] = zero_state
        return nothing
    end
    U_in  = target[p]
    U_out = implicit_solve(sources, U_in, dt_ii, p, pg, t_i)
    target[p] = U_out
    K_I_i[p]  = _implicit_rhs(sources, U_out, U_in, inv_dt_ii, p, pg, t_i)
    return nothing
end
@inline _implicit_rhs(::Tuple{AbstractImplicitSourceTerm}, U_out, U_in, inv_dt_ii, p, pg, t) = (U_out - U_in) * inv_dt_ii
@inline _implicit_rhs(sources::Tuple, U_out, U_in, inv_dt_ii, p, pg, t) = evaluate_sources(sources, U_out, p, pg, t)

@inline function _stage_implicit_rhs!(p, K_I_i, target, is_boundary, sources, pg, t_i, zero_state)
    K_I_i[p] = is_boundary[p] ? zero_state : evaluate_sources(sources, target[p], p, pg, t_i)
    return nothing
end

"""
    mood_checkpoint!(ts, eq, pg, i, dt, time)

Forms the candidate for checkpoint `i` (`Y_i`, or the final state for `i == s+1`) and runs the
stage-wise MOOD fixed point against `Y_{i-1}`. The grid geometry must still be that of stage
`i-1` when this is called, because `K_E[i-1]` is recomputed here. Terminates because every
iteration strictly lowers at least one order.
"""
function mood_checkpoint!(ts::GeneralRKTimeStepper{D, M, T}, eq, pg, i::Int, dt::T, time::Real) where {D, M, T}
    interp = ts.divergence_interpolator
    work   = ts.work
    N      = pg.meta.N
    Y_prev = ts.Y_stages[i - 1]
    t_prev = time + ts.tableau.c_t[i - 1] * dt
    reach  = _order_reach(interp, ts.mood.strategy)

    full = true
    while true
        target, _ = form_candidate!(ts, eq, pg, i, dt, time)
        cand = _checks_substeps(ts) ? form_substep!(ts, pg, i, dt) : target
        if !detect_and_drop!(ts.mood, work, interp, pg, Y_prev, cand, full; eq = eq)
            # Accepted. Flagged particles at order 1 were un-flagged by the decrement: count those still inadmissible.
            ts.mood.criterion isa NoMOOD || (work.inadmissible[] += _count_inadmissible(eq, pg, cand, parent(work.scratch)))
            return nothing
        end

        copyto!(parent(work.dirty), 1, parent(work.changed), 1, N)
        expand_mask!(parent(work.dirty), parent(work.scratch), pg, reach)
        compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[i - 1], ts.K_X_stages[i - 1], ts.K_V_stages[i - 1], ts.V_stages[i - 1], Y_prev, t_prev, false)
        full = false
    end
end

# Interior particles whose state is not admissible: one kernel pass into the free MOOD scratch mask and a `count`
# over it (the pattern of the other MOOD reductions; no allocation, device arrays included)
function _count_inadmissible(eq, pg, U, flags)
    N = pg.meta.N
    N == 0 && return 0
    foreach_particle(_inadmissible_body!, get_backend(), N, (flags, pg.core.is_boundary, U, eq))
    return count(view(flags, 1:N))
end
@inline function _inadmissible_body!(p, flags, is_boundary, U, eq)
    @inbounds flags[p] = !is_boundary[p] && !_admissible(eq, U[p])
    return nothing
end

# MOOD checks the Euler substeps (an SSP tableau, a criterion that flags anything)
@inline _checks_substeps(ts::GeneralRKTimeStepper) = ts.substep_ratio > zero(ts.substep_ratio) && !(ts.mood.criterion isa NoMOOD)

"""
    form_substep!(ts, pg, i, dt) -> rho_sub

The forward Euler substep `W_{i-1} = Y_{i-1} + (Δt/C) K_E[i-1]` that MOOD checks at checkpoint `i` (interior
particles; boundary particles keep `Y_{i-1}`, they are not checked).
"""
function form_substep!(ts::GeneralRKTimeStepper{D, M, T}, pg, i::Int, dt::T) where {D, M, T}
    # Implicit sources reach an SSP (explicit) tableau only as explicitly applied sources: their K_I is part of the RHS.
    with_KI = !isempty(ts.implicit_sources)
    foreach_particle(_substep_body!, get_backend(), pg.meta.N,
        (ts.rho_sub, ts.Y_stages[i - 1], ts.K_E_stages[i - 1], ts.K_I_stages[i - 1], (ts.K_V_stages[i - 1], ts.V_stages[i - 1]),
         pg.core.is_boundary, ts.substep_ratio * dt, with_KI, _content_form(ts, pg)))
    return ts.rho_sub
end

# In content form the Euler substep is (V Y + h V K) / (V + h K_V): the compression of the substep included
@inline function _substep_body!(p, W, Y, K, KI, KVV, is_boundary, h, with_KI, content)
    @inbounds if is_boundary[p]
        W[p] = Y[p]
    else
        dU = with_KI ? K[p] + KI[p] : K[p]
        if content
            K_V, V = KVV
            W[p] = (V[p] * Y[p] + h * V[p] * dU) / (V[p] + h * K_V[p])
        else
            W[p] = Y[p] + h * dU
        end
    end
    return nothing
end

# Switch the grid to stage geometry X (frozen topology or full search); the ghosts of moving pieces are placed at
# the stage time first (X holds their positions of the step start)
function _apply_stage_geometry!(ts::GeneralRKTimeStepper, pg, X, t_n, t_i)
    place_ghosts!(pg, X, t_n, t_i)
    set_positions!(pg, X, ts.full_search_per_stage)
    ts.full_search_per_stage && uses_slot_buffers(ts.divergence_interpolator) && update_size!(ts.int_buffer, length(pg.neighbor.indices))
    return nothing
end

# =========================================================================
# TIME STEP
# =========================================================================

function (ts::GeneralRKTimeStepper{D, M, T})(eq::HyperbolicPDE, pg::ParticleGrid, time::Real, dt::T,
                                             U::AbstractVector{State{M, T}} = pg.rhos) where {D, M, T}
    moving = _is_moving(pg)
    (U === pg.rhos || !moving) || throw(ArgumentError("a time step of a separate state array needs a static grid"))
    has_moving_pieces(pg.geometry) && _sync_boundaries!(pg, T(time), _extract_order(ts.divergence_interpolator))
    N      = pg.meta.N
    s      = ts.num_stages
    interp = ts.divergence_interpolator
    orders = pg.core.particle_orders

    update_size!(ts, N, length(pg.neighbor.indices), moving)
    update_size!(interp, N)                       # once per step; gradients persist across checkpoints

    copyto!(parent(ts.rho_n), 1, U, 1, N)
    moving && copyto!(parent(ts.x_n), 1, pg.core.positions, 1, N)
    moving && copyto!(parent(ts.v_n), 1, pg.core.volumes, 1, N)

    max_order = _extract_order(interp)
    fill!(view(orders, 1:N), max_order)
    fill!(view(ts.work.triggered_step, 1:N), false)
    ts.work.inadmissible[] = 0

    for i in 1:s
        if i == 1
            form_candidate!(ts, eq, pg, 1, dt, time)      # X_1 == x_n, grid already there
        else
            mood_checkpoint!(ts, eq, pg, i, dt, time)      # uses geometry of stage i-1
            moving && _apply_stage_geometry!(ts, pg, ts.X_stages[i], time, time + ts.tableau.c_t[i] * dt)
        end
        ts.reset_orders_per_stage && fill!(view(orders, 1:N), max_order)
        compute_explicit_rhs!(ts, eq, pg, ts.K_E_stages[i], ts.K_X_stages[i], ts.K_V_stages[i], ts.V_stages[i], ts.Y_stages[i],
                              time + ts.tableau.c_t[i] * dt, true)
    end

    mood_checkpoint!(ts, eq, pg, s + 1, dt, time)          # uses geometry of stage s
    copyto!(U, 1, parent(ts.rho_cand), 1, N)
    if ts.work.inadmissible[] > 0 && !ts.work.warned[]
        ts.work.warned[] = true
        @warn "MOOD accepted $(ts.work.inadmissible[]) inadmissible candidate(s) at first order in the step at t = $time " *
              "(dt = $dt): the first-order update itself is not admissible there (a step above its positivity limit, or a " *
              "first-order scheme that is not positive on the cloud, e.g. RusanovFlux on an irregular cloud; " *
              "AlignedRusanovFlux is). Reported once per stepper; `mood_inadmissible(ts)` gives the count of each step."
    end

    if moving
        place_ghosts!(pg, parent(ts.x_cand), time, time + dt)
        copyto!(pg.core.positions, 1, parent(ts.x_cand), 1, N)
        pg.neighbor(pg)                                    # full search + periodic wrap for the next step
        if !(pg.mover isa NoGridMover)
            _check_cloud_quality(pg, time + dt)
            # Volumes are re-estimated when the interior moves (a static cloud with moving pieces keeps the volumes the
            # remaps of `sweep_boundaries!` maintain). With the Lagrangian frame the content V U is then remapped onto
            # the re-estimated volumes from the volumes the scheme's own volume rate implies (`v_cand`): the mass
            # equation's right-hand side sums to exactly −Σ U K_V against those (its only pair term, the dissipation,
            # is antisymmetric), so Σ V U is conserved up to the time discretisation of V U and the pair defect of the
            # momentum and energy fluxes. Without an estimator the implied volumes are the volumes.
            remap = _content_form(ts, pg)
            if ts.volume_estimator === nothing
                remap && copyto!(pg.core.volumes, 1, parent(ts.v_cand), 1, N)
            else
                update_volumes!(pg, ts.volume_estimator)
                remap && foreach_particle(_content_remap!, get_backend(), N, (U, ts.v_cand, pg.core.volumes, pg.core.is_boundary))
            end
        end
        has_moving_pieces(pg.geometry) && sweep_boundaries!(pg, time, time + dt, max_order)
    end
    return nothing
end

# A moving interior cloud that the fixed stencil radius can no longer describe: interior particles closer to each other
# than a tenth of the spacing (the MLS derivative weight of a nearly coincident neighbour vanishes with the distance, so the
# pair feels no dissipation, crosses, and its states diverge), or interior particles without any neighbour (a rarefaction
# stretched the cloud beyond R). Both happened on the Woodward-Colella blast with LagrangianMover within 300 steps (pairs
# 0.004 dx apart, gaps of 70 dx): the volumes lost their meaning and the run reported +78 % of mass without failing.
# Checked after the neighbour search of every step; reported once per session.
const _WARNED_CLOUD = Ref(false)
function _check_cloud_quality(pg::ParticleGrid{D, M, T}, t) where {D, M, T}
    _WARNED_CLOUD[] && return nothing
    N = pg.meta.N
    N == 0 && return nothing
    buf = pg.shared.float_buffer
    foreach_particle(_closest_neighbour!, get_backend(), N,
                     (buf, pg.core.is_boundary, pg.neighbor.ranges, pg.neighbor.distances, T(Inf), T(-1)))
    dmin   = minimum(view(buf, 1:N); init = T(Inf))
    lonely = count(==(T(-1)), view(buf, 1:N))
    h = minimum(pg.meta.dx)
    (dmin < T(0.1) * h || lonely > 0) || return nothing
    _WARNED_CLOUD[] = true
    @warn "moving grid at t = $t: the cloud is collapsing or stretching beyond what the stencil radius resolves (closest " *
          "interior pair $(round(dmin / h; sigdigits = 2)) spacings apart, $lonely interior particle(s) without a neighbour). " *
          "Nearly coincident particles feel no mutual dissipation (their MLS weights vanish with the distance), cross, and " *
          "their states diverge; particles without neighbours are not updated; the volumes and the discrete mass lose their " *
          "meaning. Use particle management (ParticleAdaptivity: merging and insertion between steps) and/or a " *
          "ScaledLagrangianMover. Reported once."
    return nothing
end

# Kernel body (see Backend.jl): the distance to the closest neighbour of an interior particle, `none` if it has none,
# `skip` for boundary particles
@inline function _closest_neighbour!(i, out, isb, ranges, distances, skip, none)
    @inbounds begin
        if isb[i]
            out[i] = skip
            return nothing
        end
        d = skip
        for k in ranges[i]
            d = min(d, sqrt(sum(abs2, distances[k])))
        end
        out[i] = isempty(ranges[i]) ? none : d
    end
    return nothing
end
