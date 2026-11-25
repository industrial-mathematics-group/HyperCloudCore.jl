export ParticleAdaptivity, adapt_particles!, is_wall

# =========================================================================
# PARTICLE INSERTION / MERGING WITH A CONSERVATIVE SHAPE-FUNCTION REMAP
# =========================================================================
#
# A topology change (particles added, removed or moved) replaces the partition of unity
# {φ_i^old} by {φ_i^new}. Every particle's conserved content changes by the "swept" amount
#
#     δm_i = ∫ (φ_i^new − φ_i^old)(x) Ũ(x) dx ,       δV_i = ∫ (φ_i^new − φ_i^old) dx ,
#
# where Ũ is the limited reconstruction of the OLD solution. Because both families sum to one
# pointwise, Σ_i δm_i = 0 and Σ_i δV_i = 0 for any quadrature of the integral: the remap is
# exactly conservative by construction, without cell intersections. New particles have
# φ^old = 0, removed particles have φ^new = 0. The integral is evaluated on a small throwaway
# lattice around the event; a limited (bound-clipped) linear reconstruction keeps Ũ in the local
# bounds, and a final repair pass enforces the discrete maximum principle while keeping the sum
# exact. All of this is the particle analogue of swept-region remapping in ALE finite volumes.

"""
    ParticleAdaptivity(; max_gap = nothing, min_gap = 0.5, clearance = nothing, samples_per_dx = 2,
                         exclude_reduced_order = true, max_events = typemax(Int), max_rounds = 8,
                         volume_estimator = nothing)

Rules for keeping a moving point cloud well-spaced. Lengths are multiples of the particle spacing
`dx = minimum(pg.meta.dx)`.
- **Insertion**: a neighbour pair `(i, j)` farther apart than `max_gap·dx` whose midpoint has no
  particle within `clearance·dx` gets a new particle at the midpoint.
- **Merge**: a neighbour pair closer than `min_gap·dx` is replaced by one particle at the midpoint.
- `max_gap = nothing`, `clearance = nothing`: by the dimension of the grid, `1.6` and `0.7` in 1-D and 2-D, `1.6 √(D/2)`
  and `0.7 √(D/2)` above (1.96 and 0.86 in 3-D). A regular lattice must not trigger insertions: its longest diagonal,
  `√D dx`, has to stay below `max_gap`, or its midpoint, the cell centre at `√D/2 dx` from the corners, within the
  clearance. With the 2-D values a 3-D lattice got a particle at every cell centre (+77 % particles in 176 steps of a
  vortex at 24³).
- **Open boundaries** (moving clouds): an interior particle that has left the region is removed with its content
  (the outflow), unless it crossed a wall ([`is_wall`](@ref)) of a shape-based domain, where it is put back onto
  the wall; and the insertion rule also holds between an interior particle and a ghost: the gap a cloud leaves when
  it recedes from a boundary (an inflow) is filled with seeds, whose content enters the region. No event is centred
  outside the region.
- `exclude_reduced_order`: skip events near particles whose MOOD order was lowered in the last step
  (the shock band), where interpolation of any order is unreliable.
- `samples_per_dx`: resolution of the remap quadrature lattice.
- `max_events`: at most this many events per round.
- `max_rounds`: a call repeats detection, remaps and neighbour search until no pair violates the rules, at most this
  many times. The events of one round are kept apart (below), a few per `(9.5 dx)^2` in 2-D at `R = 2.5 dx`, and a pair
  can only be found while it is within the radius `R`: a gap between `max_gap·dx` and `R` that one round per step
  does not reach in time grows beyond `R` and is never filled (an expanding cloud with a small radius lost its
  stencils that way). `1` is the behaviour up to 0.9.9.
- `volume_estimator`: the estimator the time stepper re-estimates the volumes with (a moving grid). After
  the topology change the volumes of the new cloud are estimated with it, and the change of `Σ V U` this
  causes is put back onto the particles the remap changed (bounded by their local extrema, like the remap's
  own repair), so the call is exact for the volumes the next steps use. `nothing` keeps the remap's own
  volumes (right for a static grid, whose volumes are never re-estimated).
- At most one event per neighbourhood and per round. Event centres are kept `(R + dx/2) √D + 3R/2 + 3dx/2` apart
  (see `_event_spacing`), so no event's quadrature sees a particle that another event removes or adds, and every
  remap knows what the others remove: the call is exactly conservative. Call once between time steps.
"""
struct ParticleAdaptivity{T, VE <: Union{Nothing, VolumeEstimator}}
    max_gap::T
    min_gap::T
    clearance::T
    samples_per_dx::Int
    exclude_reduced_order::Bool
    max_events::Int
    max_rounds::Int
    volume_estimator::VE
end

function ParticleAdaptivity(::Type{T} = Float64; max_gap = nothing, min_gap = 0.5, clearance = nothing,
                            samples_per_dx::Int = 2, exclude_reduced_order::Bool = true,
                            max_events::Int = typemax(Int), max_rounds::Int = 8,
                            volume_estimator::Union{Nothing, VolumeEstimator} = nothing) where {T}
    @assert min_gap < 1 && (max_gap === nothing || max_gap > 1) "need min_gap < 1 < max_gap (in units of dx)"
    max_gap   === nothing && (max_gap = NaN)              # (by the dimension of the grid, `_max_gap`)
    clearance === nothing && (clearance = NaN)
    @assert max_events >= 1 "max_events must be at least 1"
    @assert max_rounds >= 1 "max_rounds must be at least 1"
    return ParticleAdaptivity{T, typeof(volume_estimator)}(T(max_gap), T(min_gap), T(clearance), samples_per_dx,
                                                         exclude_reduced_order, max_events, max_rounds, volume_estimator)
end

# The thresholds of the insertion rule on a grid of dimension `D` (the defaults depend on it, see `ParticleAdaptivity`)
@inline _max_gap(ad::ParticleAdaptivity{T}, ::Val{D}) where {T, D} =
    isnan(ad.max_gap) ? (D <= 2 ? T(1.6) : T(1.6) * sqrt(T(D) / 2)) : ad.max_gap
@inline _clearance(ad::ParticleAdaptivity{T}, ::Val{D}) where {T, D} =
    isnan(ad.clearance) ? (D <= 2 ? T(0.7) : T(0.7) * sqrt(T(D) / 2)) : ad.clearance

struct _AdaptEvent{D, T}
    center::Space{D, T}
    removed::Vector{Int}
    added::Vector{Space{D, T}}
end

# The result of a call (`adapt_particles!`)
_adapt_result(::Type{S}; inserted = 0, removed = 0, mass_residual = zero(S), escaped = 0, held = 0, seeded = 0, refused = 0,
              escaped_content = zero(S), seeded_content = zero(S)) where {S} =
    (; inserted, removed, mass_residual, escaped, held, seeded, refused, escaped_content, seeded_content)

# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
"""
    adapt_particles!(pg, ad::ParticleAdaptivity, max_order::Int; states = (), own_volumes = false)
        -> (; inserted, removed, mass_residual, escaped, held, seeded, refused, escaped_content, seeded_content)

Detects holes and clusters, applies the conservative remap for every event on the old topology,
changes the particle set, and rebuilds the neighbour lists, in up to `ad.max_rounds` rounds. `max_order` is the
scheme's maximum consistency order (new particles start there, and `exclude_reduced_order` compares against it).
Requires valid neighbour lists and volumes (`update_volumes!`). `mass_residual` is the change of the interior
content `Σ V U` (boundary particles excluded, which carry no volume) by the events of the call, which should be at
round-off level. Merges and insertions involve interior particles only, and no event is centred outside the region;
`refused` counts the events that could not be remapped and were left out (see `_remap_event!`).

On a moving cloud the call first removes the interior particles that left the region (`escaped`, with the content
`escaped_content`; those that crossed a wall, [`is_wall`](@ref), are put back onto it: `held`) and seeds the gaps
between the cloud and the boundary (`seeded`, bringing `seeded_content`), see [`ParticleAdaptivity`](@ref). The
interior content changes by `seeded_content - escaped_content + mass_residual` over the call.

`own_volumes`: whether the volumes of the cloud are kept by the time stepper between its steps instead of being
re-estimated after each ([`keeps_own_volumes`](@ref): the content form of the Lagrangian movers). The remap's
volumes then stay (the particles' shares of the partition, `Σ V U` exact without a correction), and
`ad.volume_estimator` only gives the seeds their volume. `false` (the stepper re-estimates all volumes after every
step, with the same estimator): all volumes are replaced, as the next step would, and a correction on the changed
particles balances the whole change.

On a grid whose buffers `reserve!` froze, a `CapacityError` is thrown before the grid is changed by the seeds or by a
round: the particle buffers and the neighbour slots the new particle set needs (`_slots_after_events`) are checked
before the first remap of the round (earlier rounds stay applied, each of them conservative).

The event work (detection, remap, compaction) is serial and runs on the host. On a grid built with
`device_grid` a kernel first checks whether any neighbour pair violates the spacing rules; only then
is the grid copied to the host, adapted there, and copied back into the device grid in place (its
arrays grow as needed), where the neighbour search runs. A well-spaced cloud costs one kernel and
one reduction per call.

`states` are further state arrays over the cloud (the blocks of a block state: host arrays of the grid's state type
that can grow, `Vector`s or `Growable`s), remapped, compacted and extended exactly like `pg.rhos` (the geometric
decisions are the same for all, the remap of every array is that of the single state); host grids only.
`mass_residual` is that of `pg.rhos`.
"""
function adapt_particles!(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, max_order::Int; states = (),
                          own_volumes::Bool = false) where {D, M, T}
    nothing_done = _adapt_result(State{M, T})
    pg.meta.N == 0 && return nothing_done
    (ad.volume_estimator !== nothing && has_moving_pieces(pg.geometry)) &&
        throw(ArgumentError("adapt_particles!: a volume estimator together with moving pieces is not supported (the " *
                            "estimators integrate the region at t = 0, the pieces have moved it); use volume_estimator = nothing"))
    escaped = _escaped_particles(pg)
    has_candidates = _has_adapt_candidates(pg, ad)        # (the flags stay in the grid's bit buffer)
    (isempty(escaped) && !has_candidates) && return nothing_done

    on_host = parent(pg.rhos) isa Array
    (on_host || _host_states(states)) ||
        throw(ArgumentError("adapt_particles!: extra state arrays of a device grid must be host arrays (the events run on a host copy)"))
    hg = on_host ? pg : device_grid(Array, pg)            # the event work is serial: on a host copy of a device grid
    st_list = states isa _LazyStates ? states : _state_list(hg, states)
    cand = on_host ? (has_candidates ? findall(view(hg.shared.bit_buffer, 1:hg.meta.N)) : Int[]) : _adapt_candidates(hg, ad)

    # Particles that left the region: put back onto their wall, or removed with their content (an outflow). The
    # removal is applied with the first round below (one change of the particle set, one neighbour search); a particle
    # put back has moved, so the lists are searched at once (rare: a wall has no flow through it)
    mass_before = interior_mass(hg)
    held, lost  = 0, zero(State{M, T})
    gone = Int[]
    if !isempty(escaped)
        geom, t_bnd = hg.geometry, hg.meta.boundary_time
        st0 = _states_now(st_list, hg)
        for i in escaped
            y = _hold_at_wall(geom, hg.core.positions[i], t_bnd)
            if y === nothing
                push!(gone, i)
                lost += hg.core.volumes[i] * st0[1][i]
            else
                hg.core.positions[i] = y
                held += 1
            end
        end
        if held > 0
            mask = hg.shared.bit_buffer
            fill!(view(mask, 1:hg.meta.N), false)
            for i in gone; mask[i] = true; end
            isempty(gone) || _apply_topology!(hg, mask, Space{D, T}[], [State{M, T}[] for _ in st0], T[], max_order; states = st0)
            hg.neighbor(hg)
            empty!(gone)                                  # (applied)
            cand = _adapt_candidates(hg, ad)
        end
    end
    dropped = length(escaped) - held
    mass0   = mass_before - lost
    # The seeds in the gaps next to the boundary, added with the first round as well
    seed_pos, seed_pairs = _seed_candidates(hg, ad, max_order, cand, gone)
    n_seeds = length(seed_pos)
    inflow  = zero(State{M, T})                           # the content the seeds bring (new volume: the gap they fill)

    # Rounds of independent events: detection on the current cloud, the remaps on its topology, the topology change
    # and the neighbour search, until no pair violates the rules (the volumes are re-estimated once, at the end).
    # The pre-check runs once: the candidates of a later round are those left, the neighbours of what was removed and
    # the new particles with their neighbours (`_next_candidates`).
    n_refused, n_removed, n_added = 0, 0, 0
    seeds   = Set{Int}()
    changed = ad.volume_estimator === nothing ? nothing : Dict{Int, Tuple{State{M, T}, State{M, T}}}[]
    for round in 1:ad.max_rounds
        N       = hg.meta.N
        pending = round == 1 && (!isempty(gone) || !isempty(seed_pos))     # the boundary changes go with round 1
        events  = isempty(cand) ? _AdaptEvent{D, T}[] :
                  _detect_events(hg, ad, max_order, cand; dead = (pending ? gone : Int[]),
                                 seeds = (pending ? seed_pos : Space{D, T}[]))
        (isempty(events) && !pending) && break
        all_states = _states_now(st_list, hg)
        n_seed_r   = pending ? length(seed_pos) : 0
        # Capacity first: the remaps below rewrite states and volumes in place, so a `CapacityError` (buffers frozen
        # by `reserve!`) thrown by `_apply_topology!` would leave the grid in a non-conservative intermediate state
        N_new = N + sum(ev -> length(ev.added) - length(ev.removed), events; init = 0) + n_seed_r
        if N_new > N
            ensure_capacity!(hg, N_new)
            on_host || ensure_capacity!(pg, N_new)
        end
        # ... and the neighbour slots the search after the topology change will need (in the layout of `pg`)
        counted = pending ? vcat(events, [_AdaptEvent{D, T}(x, Int[], [x]) for x in seed_pos]) : events
        total_nb, max_nb = _slots_after_events(hg, counted)
        ensure_capacity!(pg.neighbor, _slot_capacity(eltype(pg.neighbor.ranges), total_nb, max_nb, N_new))

        # every particle this round removes, known to all its remaps (their local sets may overlap, `_event_spacing`)
        dead_mask = falses(N)
        for ev in events, r in ev.removed; dead_mask[r] = true; end
        pending && (dead_mask[gone] .= true)
        # (the seeds' states from the pairs as they are now, before the remaps write)
        seed_rho = [[(st[i] + st[j]) / 2 for (i, j) in (pending ? seed_pairs : Tuple{Int, Int}[])] for st in all_states]
        remove_mask = hg.shared.bit_buffer
        fill!(view(remove_mask, 1:N), false)
        new_pos, new_vol = Space{D, T}[], T[]
        new_rho = [State{M, T}[] for _ in all_states]
        df = State{M, T}[]                                # scratch of the remaps (one neighbour list)

        record = changed === nothing ? nothing : [Tuple{Int, State{M, T}, State{M, T}}[] for _ in all_states]
        fill!(view(hg.shared.int_buffer, 1:N), 0)         # the index map of the remaps, cleared once per round
        refused = 0
        for ev in events
            # (an event that cannot be remapped, e.g. next to a particle without a volume, is left out: nothing is written)
            if _remap_event!(hg, ad, ev, new_pos, new_rho, new_vol, df; record = record, states = all_states, strict = false,
                             clear_map = false, removed_mask = dead_mask)
                for r in ev.removed; remove_mask[r] = true; end
            else
                refused += 1
            end
        end
        n_refused += refused
        n_rem_ev = count(view(remove_mask, 1:N))
        n_add_ev = length(new_pos)
        (n_rem_ev + n_add_ev) == 0 && !pending && break   # (every event refused: the next round would find them again)
        if pending
            for i in gone; remove_mask[i] = true; end
            append!(new_pos, seed_pos)                    # (the seeds follow the events' particles)
            append!(new_vol, fill(prod(hg.meta.dx), n_seed_r))
            for s in eachindex(new_rho); append!(new_rho[s], seed_rho[s]); end
            for k in 1:n_seed_r; inflow += prod(hg.meta.dx) * seed_rho[1][k]; end
        end
        n_rem = count(view(remove_mask, 1:N))
        n_removed += n_rem_ev
        n_added   += n_add_ev
        N_kept = N - n_rem
        # the particles written so far, by their indices after this round's compaction
        seeds = _compacted_ids(collect(seeds), remove_mask, N, N_kept)
        for k in 1:n_seed_r; push!(seeds, N_kept + n_add_ev + k); end
        if changed !== nothing
            now = [_compacted_record(r, remove_mask, N, N_kept) for r in record]
            isempty(changed) ? append!(changed, now) :
                (changed .= [_merge_bounds!(_compacted_bounds(changed[s], remove_mask, N), now[s]) for s in eachindex(now)])
        end
        near = Int[]                                      # the neighbours of what is removed (the old lists)
        nbh  = hg.neighbor
        @inbounds for i in 1:N
            remove_mask[i] || continue
            for k in nbh.ranges[i]; push!(near, nbh.indices[k]); end
        end
        cand_kept = _compacted_ids(vcat(cand, near), remove_mask, N, N_kept)
        _apply_topology!(hg, remove_mask, new_pos, new_rho, new_vol, max_order; states = all_states)
        hg.neighbor(hg)
        cand = _next_candidates(hg, cand_kept, N_kept)
    end
    (held + dropped + n_seeds + n_removed + n_added) == 0 && return _adapt_result(State{M, T}; refused = n_refused)

    all_states = _states_now(st_list, hg)
    on_host || (_upload_particles!(pg, hg); pg.neighbor(pg))
    if changed !== nothing && (!isempty(changed) || !isempty(seeds))
        isempty(changed) && append!(changed, (Dict{Int, Tuple{State{M, T}, State{M, T}}}() for _ in all_states))
        inflow += _reestimate_volumes!(pg, hg, ad.volume_estimator, changed; states = all_states, seeds = seeds,
                                       own_volumes = own_volumes)
    end

    return _adapt_result(State{M, T}; inserted = n_added, removed = n_removed,
                         mass_residual = interior_mass(pg) - mass0 - inflow, escaped = dropped, held = held,
                         seeded = n_seeds, refused = n_refused, escaped_content = lost, seeded_content = inflow)
end

# The candidates of the next round on the new particle set: those of this round and the neighbours of the removed
# particles (`kept`, already by their new indices), and the new particles (from `N_kept + 1`) with their neighbours. A
# particle starts an event with a neighbour of a higher index, so a new violation is started by a neighbour of a
# removed particle (its midpoint was blocked by it), by a new particle or by a neighbour of one.
function _next_candidates(hg::ParticleGrid, kept::Set{Int}, N_kept::Int)
    nb = hg.neighbor
    @inbounds for a in (N_kept + 1):hg.meta.N
        push!(kept, a)
        for k in nb.ranges[a]; push!(kept, nb.indices[k]); end
    end
    return sort!(collect(kept))
end

# -------------------------------------------------------------------------
# Particles that left the region (moving clouds)
# -------------------------------------------------------------------------
"""
    _escaped_particles(pg) -> Vector{Int}

The interior particles of a moving cloud that lie outside the region (at `boundary_time`), with a finite position.
Only particles next to the boundary can have crossed it since the last call: a kernel flags the interior particles
with a ghost among their neighbours (or with no neighbour at all), and the region test runs on the host for those.
"""
function _escaped_particles(pg::ParticleGrid{D, M, T}) where {D, M, T}
    out = Int[]
    (pg.mover isa NoGridMover || pg.meta.N_ghost == 0) && return out
    N    = pg.meta.N
    nb   = pg.neighbor
    flag = pg.shared.bit_buffer
    foreach_particle(_near_boundary!, get_backend(), N, (flag, pg.core.is_boundary, nb.ranges, nb.indices))
    any(view(flag, 1:N)) || return out
    host = parent(pg.core.positions) isa Array
    near = host ? flag : Array(view(parent(flag), 1:N))
    pos  = host ? pg.core.positions : Array(view(parent(pg.core.positions), 1:N))
    geom, t = pg.geometry, pg.meta.boundary_time
    @inbounds for i in 1:N
        near[i] || continue
        x = pos[i]
        (all(isfinite, x) && !_is_interior(geom, x, t)) && push!(out, i)   # (a NaN position is a blow-up, not an escape)
    end
    return out
end
# Kernel body (see Backend.jl)
@inline function _near_boundary!(i, flag, isb, ranges, indices)
    near = false
    @inbounds if !isb[i]
        near = isempty(ranges[i])
        for k in ranges[i]
            isb[indices[k]] && (near = true; break)
        end
    end
    @inbounds flag[i] = near
    return nothing
end

"""
    is_wall(bc::AbstractBoundaryCondition) -> Bool

Whether no flow crosses the boundary of the condition `bc`. The particle management of a moving cloud
([`adapt_particles!`](@ref)) puts an interior particle that crossed such a boundary back onto it (the crossing is
the noise of its normal velocity) and removes one that crossed any other boundary with its content (an outflow).
`true` for [`MirrorBC`](@ref), `false` otherwise; define it for a condition type of your own that models a wall.
"""
is_wall(::AbstractBoundaryCondition) = false
is_wall(::MirrorBC) = true

# The tag of a point at time `t` (a domain without a shape has no time dependence)
_point_tag(geom::GeometricDomain, pos, t) = geom.shape === nothing ? geom.get_tag(pos) : geom.get_tag(pos, t)

# The wall point of a particle that crossed a wall (`is_wall`) of a shape-based domain, `nothing` for every other
# boundary: a wall has no flow through it, so the crossing is the noise of the particle's normal velocity
function _hold_at_wall(geom::GeometricDomain{GEO, D, T}, x::Space{D, T}, t) where {GEO, D, T}
    geom.shape === nothing && return nothing
    bc = get(geom.bc_map, _point_tag(geom, x, t), nothing)
    (bc !== nothing && is_wall(bc)) || return nothing
    n, φ = _wall_geometry(at_time(geom.shape, t), x)
    (isfinite(φ) && all(isfinite, n)) || return nothing
    y = x - φ * n
    δ = 64 * eps(T) * max(one(T), maximum(abs, x))
    for _ in 1:8                                          # (round-off can leave the wall point just outside)
        _is_interior(geom, y, t) && return y
        y -= δ * n
        δ *= 4
    end
    return nothing
end

# The state arrays an event changes: the grid's own first, then the extra ones (the blocks of a block state)
# Extra state arrays made on first use (the `states` of `adapt_particles!`/`sweep_boundaries!`): the block pipeline
# copies its blocks to host vectors only when an event remaps them. `vecs` are the arrays made, `list` the state list.
mutable struct _LazyStates{F}
    make::F
    vecs::Any
    list::Any
end
_LazyStates(make) = _LazyStates(make, nothing, nothing)
_states_now(ls::_LazyStates, hg) = ls.list === nothing ? (ls.vecs = ls.make(); ls.list = _state_list(hg, ls.vecs)) : ls.list
_states_now(list, hg) = list
_host_states(::_LazyStates) = true                        # (host vectors)
_host_states(states) = all(st -> parent(st) isa Array, states)

function _state_list(hg::ParticleGrid{D, M, T}, states) where {D, M, T}
    all_states = AbstractVector{State{M, T}}[hg.rhos]
    for st in states
        st isa AbstractVector{State{M, T}} || throw(ArgumentError("adaptivity: an extra state array must hold the grid's state type State{$M, $T}"))
        push!(all_states, st)
    end
    return all_states
end
# Growth of a state array to `n` entries: a `Growable` grows in place, a `Vector` is resized, a view cannot grow
_grow_state!(g::Growable, n::Int) = ensure_capacity!(g, n)
_grow_state!(v::Vector, n::Int) = (length(v) < n && resize!(v, n); nothing)
_grow_state!(v::AbstractVector, n::Int) = length(v) >= n ? nothing :
    throw(ArgumentError("a state array of type $(typeof(v)) cannot grow to $n entries (give a Vector or a Growable)"))

_apply_topology!(pg::ParticleGrid{D, M, T}, remove_mask::AbstractVector{Bool}, new_pos::Vector{Space{D, T}},
                 new_rho::Vector{State{M, T}}, new_vol::Vector{T}, max_order::Int) where {D, M, T} =
    _apply_topology!(pg, remove_mask, new_pos, [new_rho], new_vol, max_order)

"""
    _slots_after_events(pg, events) -> (total, max_nb)

Neighbour slots of the particle set after `events` (old particles minus the removed ones, plus the added ones):
the total and the largest count per particle, from the current neighbour lists and cell list, without changing
anything. Removed particles leave the lists of their neighbours; an added particle gets every surviving or added
particle within `R` and joins their lists. Distances are compared with a slightly larger radius, so the count is
an upper bound (round-off at exactly `R`).
"""
function _slots_after_events(pg::ParticleGrid{D, M, T}, events) where {D, M, T}
    N, nb = pg.meta.N, pg.neighbor
    R_sq  = pg.meta.R^2 * (1 + 16 * eps(T))
    L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
    removed = Set{Int}(r for ev in events for r in ev.removed)
    added   = [x for ev in events for x in ev.added]
    delta   = Dict{Int, Int}()                                    # change of the count of a surviving particle
    for r in removed, k in nb.ranges[r]
        j = nb.indices[k]
        j in removed || (delta[j] = get(delta, j, 0) - 1)
    end
    total, max_nb = 0, 0
    idx, dvec = Int[], Space{D, T}[]
    for (a, xa) in enumerate(added)
        c = 0
        for k in 1:_collect_near!(pg, xa, R_sq, idx, dvec)
            j = idx[k]
            j in removed && continue
            c += 1
            delta[j] = get(delta, j, 0) + 1
        end
        for (b, xb) in enumerate(added)
            b != a && sum(abs2, _wrapped_diff(xb, xa, L, Li)) <= R_sq && (c += 1)
        end
        total += c
        max_nb = max(max_nb, c)
    end
    # the survivors: the current lists, minus those of the removed particles, plus the changes (the largest count of
    # the current lists stands for the unchanged particles: an upper bound if a removed particle held it)
    lens   = view(nb.ranges, 1:N)
    total += sum(length, lens; init = 0)
    max_nb = max(max_nb, maximum(length, lens; init = 0))
    for r in removed; total -= length(nb.ranges[r]); end
    for (i, d) in delta
        total += d
        max_nb = max(max_nb, length(nb.ranges[i]) + d)
    end
    return total, max_nb
end

# The particles the remaps wrote, with their DMP bounds, by index of the compacted cloud (survivors keep their
# order, new particles follow): `index => (lo, hi)`. A survivor of one event that another event removes is
# dropped; a particle written by two events gets the union of the bounds.
function _compacted_record(record::Vector{Tuple{Int, State{M, T}, State{M, T}}}, remove_mask, N::Int, N_kept::Int) where {M, T}
    new_index = zeros(Int, N)
    c = 0
    @inbounds for i in 1:N
        remove_mask[i] || (c += 1; new_index[i] = c)
    end
    out = Dict{Int, Tuple{State{M, T}, State{M, T}}}()
    for (j, lo, hi) in record
        k = j > 0 ? new_index[j] : N_kept - j
        k == 0 && continue
        if haskey(out, k)
            lo0, hi0 = out[k]
            out[k] = (min.(lo0, lo), max.(hi0, hi))
        else
            out[k] = (lo, hi)
        end
    end
    return out
end

"""
    _seed_candidates(hg, ad, max_order, cand, gone) -> (positions, pairs)

The seeds of a call: new interior particles in the gaps between the cloud and the boundary. Where an interior
particle and a ghost among its neighbours are farther apart than `max_gap·dx`, with the midpoint inside the region and
no particle (or earlier seed) within `clearance·dx` of it, a particle is put at the midpoint. A moving cloud recedes
from a boundary with inflow; without the seeds the first interior row loses the ghosts from its stencils, and the
volume estimators extrapolate into the gap.

A seed is not a remap event: the gap is volume no particle of the cloud holds, and what fills it enters the region.
Its state is the mean of the pair's (the ghost's is the boundary datum, the interior particle's the solution next to
it), its volume the nominal `prod(dx)` until an estimator replaces it (`_reestimate_volumes!`). A whole row is
seeded in one call (the event spacing of the conservative remaps allowed a few per boundary and call: a cloud
receding at the flow speed opened the gap faster). `cand`: the particles to look at (the pre-check's flags); `gone`:
particles that are removed with the same change of the particle set (the escaped ones), ignored here. Returns the
positions and the pairs `(interior, ghost)`; the driver adds the particles with its first round.
"""
function _seed_candidates(hg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, max_order::Int, cand, gone::Vector{Int}) where {D, M, T}
    new_pos = Space{D, T}[]
    pairs   = Tuple{Int, Int}[]
    (hg.mover isa NoGridMover || hg.meta.N_ghost == 0 || isempty(cand)) && return new_pos, pairs
    nb      = hg.neighbor
    pos, isb, orders = hg.core.positions, hg.core.is_boundary, hg.core.particle_orders
    dx      = minimum(hg.meta.dx)
    L, Li   = hg.domain.L_wrap, hg.domain.invL_wrap
    geom, t = hg.geometry, hg.meta.boundary_time
    max_gap_sq, clear_sq = (_max_gap(ad, Val(D)) * dx)^2, (_clearance(ad, Val(D)) * dx)^2
    dead    = Set{Int}(gone)
    @inbounds for i in cand
        (isb[i] || i in dead) && continue
        (ad.exclude_reduced_order && 0 < orders[i] < max_order) && continue
        for k in nb.ranges[i]
            j = nb.indices[k]
            (isb[j] && sum(abs2, nb.distances[k]) > max_gap_sq) || continue
            mid = _wrap_point(hg, pos[i] + nb.distances[k] / 2)
            _is_interior(geom, mid, t) || continue
            half_d = nb.distances[k] / 2
            empty  = true
            for k2 in nb.ranges[i]
                if sum(abs2, nb.distances[k2] - half_d) < clear_sq && !(nb.indices[k2] in dead)
                    empty = false; break
                end
            end
            empty || continue
            any(x -> sum(abs2, _wrapped_diff(x, mid, L, Li)) < clear_sq, new_pos) && continue
            push!(new_pos, mid); push!(pairs, (i, j))
            break                                         # one seed per interior particle and call
        end
    end
    return new_pos, pairs
end

# The bounds of an earlier round (`index => (lo, hi)`) by the indices after a compaction (removed particles dropped)
function _compacted_bounds(b::Dict{Int, Tuple{State{M, T}, State{M, T}}}, remove_mask, N::Int) where {M, T}
    new_index = zeros(Int, N)
    c = 0
    @inbounds for i in 1:N
        remove_mask[i] || (c += 1; new_index[i] = c)
    end
    out = Dict{Int, Tuple{State{M, T}, State{M, T}}}()
    for (i, v) in b
        k = new_index[i]
        k == 0 || (out[k] = v)
    end
    return out
end
# The union of two sets of bounds (a particle written twice gets the union of its bounds)
function _merge_bounds!(a::Dict{Int, Tuple{State{M, T}, State{M, T}}}, b::Dict{Int, Tuple{State{M, T}, State{M, T}}}) where {M, T}
    for (k, (lo, hi)) in b
        if haskey(a, k)
            lo0, hi0 = a[k]
            a[k] = (min.(lo0, lo), max.(hi0, hi))
        else
            a[k] = (lo, hi)
        end
    end
    return a
end

# The compacted indices of raw record indices (`j > 0`: an old particle, `-a`: the `a`-th new one), as `_compacted_record`
function _compacted_ids(raw::Vector{Int}, remove_mask, N::Int, N_kept::Int)
    new_index = zeros(Int, N)
    c = 0
    @inbounds for i in 1:N
        remove_mask[i] || (c += 1; new_index[i] = c)
    end
    out = Set{Int}()
    for j in raw
        k = j > 0 ? new_index[j] : N_kept - j
        k == 0 || push!(out, k)
    end
    return out
end

"""
    _reestimate_volumes!(pg, hg, est, changed; seeds) -> inflow

Replaces the remap's volumes by those of `est` (the time stepper's estimator) and puts the change of `Σ V U` this
causes back onto the particles the remap changed (`changed`, by index with their local bounds), with the bounded,
exactly conservative redistribution of the remap's own repair. The call is then exact for the volumes the next
steps use, and the states stay the remap's up to that small correction. Keeping each particle's content instead
(`U_i ← V_i^remap U_i / V_i^est`) would also be exact, but the remap's volumes (integrals of the Shepard partition)
and an estimator's differ by several percent per particle near an event, which that would put into the states.
`hg` is the host grid of the event work (`pg` itself on the host, its host copy for a device grid).
Ghost particles carry no volume, so the interior content is the whole content (up to 0.7.9 the ghosts had volumes
and their content change was moved onto the interior, i.e. across the wall).

`own_volumes` (the stepper keeps its volumes between the steps, the scheme's own of the content form): the remap's
volumes stay, for every particle. They are the particles' shares of the partition after the events, consistent with the
volumes around them, and `Σ V U` is exact without a correction. Replacing all volumes by the estimator's (as up to
0.9.9) put the mismatch between the scheme's and the estimator's volumes of every particle of the cloud onto the
changed particles; estimates for the changed particles alone cost a third of a call and made the states next to the
events worse, not better (C6 at `R = 2.5 dx`, `RK2MUSCL1`: density range 0.66 … 3.10 with them, 0.74 … 3.07 without,
0.81 … 3.0 on the static cloud). Only the seeds are estimated (`_local_volumes`, or the whole cloud for an estimator
without it), between a quarter and twice the nominal volume (a seed fills a gap of about one cell): an estimator of higher degree gives single particles next to a
boundary 1e-3 of their neighbours' volume, and the pair dissipation of the Lagrangian frame weights with the ratio.
Otherwise all volumes are replaced, as the stepper's next re-estimation would, and the correction balances the whole
change (an estimator renormalises all volumes when the particle set changes).

The seeds (`seeds`, see `_seed_candidates`) take their volume without a correction: the gap they fill is new volume,
and its content enters the region. An estimate without a volume (clamped, or a particle no sample point reaches) is
not taken. Returns the content that entered with the seeds' volumes (of the first state array).
"""
function _reestimate_volumes!(pg::ParticleGrid{D, M, T}, hg::ParticleGrid{D, M, T}, est::VolumeEstimator,
                              changed::Vector{Dict{Int, Tuple{State{M, T}, State{M, T}}}}; states = _state_list(hg, ()),
                              seeds::Set{Int} = Set{Int}(), own_volumes::Bool = false) where {D, M, T}
    N = pg.meta.N
    on_host = hg === pg
    V_remap = collect(view(hg.core.volumes, 1:N))               # (hg holds the same cloud and volumes as pg)
    Vh = hg.core.volumes
    if own_volumes
        isempty(seeds) && return zero(State{M, T})
        ids_s = sort!(collect(seeds))
        V_loc = _local_volumes(hg, est, ids_s)
        if V_loc === nothing                                    # (an estimator of the whole cloud only)
            update_volumes!(pg, est)
            on_host || copyto!(parent(hg.core.volumes), 1, parent(pg.core.volumes), 1, N)
            V_loc = [Vh[i] for i in ids_s]
            copyto!(Vh, 1, V_remap, 1, N)
        end
        @inbounds for (k, i) in enumerate(ids_s)
            v = V_loc[k]
            (v > 0 && isfinite(v)) && (Vh[i] = clamp(v, V_remap[i] / 4, 2 * V_remap[i]))
        end
    else
        update_volumes!(pg, est)
        on_host || copyto!(parent(hg.core.volumes), 1, parent(pg.core.volumes), 1, N)
        @inbounds for i in 1:N
            (Vh[i] > 0 && isfinite(Vh[i])) || (Vh[i] = V_remap[i])   # (an estimate without a volume is not taken)
        end
    end
    isb = hg.core.is_boundary
    inflow = zero(State{M, T})                                  # the seeds' change of content (of the first array)
    for i in seeds; inflow += (Vh[i] - V_remap[i]) * states[1][i]; end
    for (s, st) in enumerate(states)
        own_volumes && break                                    # (kept volumes: nothing changed but the seeds)
        ids = [i for i in keys(changed[s]) if !(i in seeds)]
        ε = zero(State{M, T})                                   # the change of content of the whole cloud (without
        @inbounds for i in 1:N                                  # the seeds) to balance
            (isb[i] || i in seeds) || (ε += (V_remap[i] - Vh[i]) * st[i])
        end
        if isempty(ids)                                         # (seeds alone: no remapped particle to balance on)
            s == 1 && (inflow -= ε)
            continue
        end
        iszero(ε) && continue
        V   = [Vh[i] for i in ids]
        Vs  = sum(V)
        Vs > 0 || continue
        U   = [st[i] + ε / Vs for i in ids]                     # Σ V U over the set changes by exactly ε ...
        _dmp_repair!(U, V, [changed[s][i][1] for i in ids], [changed[s][i][2] for i in ids])   # ... and stays bounded
        for (k, i) in enumerate(ids)
            st[i] = U[k]
        end
    end
    if !on_host
        copyto!(parent(pg.core.volumes), 1, parent(hg.core.volumes), 1, N)
        copyto!(parent(pg.rhos), 1, parent(hg.rhos), 1, N)
    end
    return inflow
end

"""
    _has_adapt_candidates(pg, ad) -> Bool

`true` if some interior particle has an interior neighbour that could trigger an event: closer than
`min_gap` (merge) or farther than `max_gap` with an empty midpoint (insertion). A superset of what
`_detect_events` selects (which also spaces events apart and skips the shock band). One kernel and one
reduction, on the grid's backend.
"""
function _has_adapt_candidates(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}) where {D, M, T}
    N    = pg.meta.N
    nb   = pg.neighbor
    dom  = pg.domain
    dx   = minimum(pg.meta.dx)
    flag = pg.shared.bit_buffer
    geom = pg.geometry
    foreach_particle(_adapt_candidate!, _launch_backend(pg), N,
        (flag, pg.core.positions, pg.core.is_boundary, nb.ranges, nb.indices, nb.distances, dom.L_wrap, dom.invL_wrap,
         dom.is_periodic, dom.L, dom.canvas_mins, (ad.min_gap * dx)^2, (_max_gap(ad, Val(D)) * dx)^2,
         (_clearance(ad, Val(D)) * dx)^2,
         (geom.mins, geom.maxs)))
    return any(view(flag, 1:N))
end

# The particles the pre-check flags, in ascending order (a host grid): every particle that can start an event or a
# seed is among them, so the serial passes (`_detect_events`, `_seed_candidates`) visit these instead of the whole
# cloud (their loops over all particles and their neighbours were a fifth of the cost of a call, and are serial)
function _adapt_candidates(hg::ParticleGrid, ad::ParticleAdaptivity)
    _has_adapt_candidates(hg, ad) || return Int[]
    return findall(view(hg.shared.bit_buffer, 1:hg.meta.N))
end

# Kernel body (see Backend.jl): the pair tests of `_detect_events` without the event bookkeeping.
@inline function _adapt_candidate!(i, flag, pos, isb, ranges, indices, dists, L, Li, is_per, Lv, mins,
                                   min_gap_sq, max_gap_sq, clear_sq, box)
    found = false
    if !isb[i]
        @inbounds for k in ranges[i]
            j = indices[k]
            (j > i || isb[j]) || continue
            d2 = sum(abs2, dists[k])
            if d2 < min_gap_sq
                isb[j] || (found = true)
            elseif d2 > max_gap_sq
                half_d = dists[k] / 2
                # (towards a ghost only with the midpoint inside the region's box: the band of ghosts has such pairs
                # at every boundary; the region test itself is the host's, in `_detect_events`)
                (isb[j] && !_in_box(_wrap_coords(pos[i] + half_d, is_per, Lv, mins), box[1], box[2], is_per)) && continue
                # The clearance of the midpoint from the stored distance vectors of the particle's neighbours (no
                # positions, no periodic wrap: most neighbours of a particle are farther than `max_gap`, each of those
                # pairs scans the list, and this loop was half the cost of a time step on a quiet cloud)
                empty = true
                for k2 in ranges[i]
                    if sum(abs2, dists[k2] - half_d) < clear_sq
                        empty = false
                        break
                    end
                end
                empty && (found = true)
            end
            found && break
        end
    end
    flag[i] = found
    return nothing
end

@inline _in_box(x::Space{D, T}, lo, hi, is_per) where {D, T} =
    all(ntuple(d -> is_per[d] || (lo[d] <= x[d] <= hi[d]), Val(D)))

# Copies the adapted particle set of the host copy `hg` back into the device grid `pg` (growing it as needed).
function _upload_particles!(pg::ParticleGrid, hg::ParticleGrid)
    N = hg.meta.N
    ensure_capacity!(pg, N)
    for (dst, src) in ((pg.core.positions, hg.core.positions), (pg.core.is_boundary, hg.core.is_boundary),
                       (pg.core.tags, hg.core.tags), (pg.core.particle_orders, hg.core.particle_orders),
                       (pg.core.volumes, hg.core.volumes), (pg.core.normals, hg.core.normals),
                       (pg.core.wall_distances, hg.core.wall_distances), (pg.core.wall_curvatures, hg.core.wall_curvatures),
                       (pg.rhos, hg.rhos), (pg.curvatures, hg.curvatures))
        copyto!(parent(dst), 1, parent(src), 1, N)
    end
    pg.meta.N          = N
    pg.meta.N_interior = hg.meta.N_interior
    pg.meta.N_ghost    = hg.meta.N_ghost
    return nothing
end

# -------------------------------------------------------------------------
# Detection
# -------------------------------------------------------------------------
"""
    _event_spacing(pg) -> T

Radius around an accepted event centre within which no particle may start another event in the same round:
`(R + dx/2) √D + 3R/2 + 3dx/2`. Events are remapped one after the other on the old topology, so the partition of
one remap is only that of the cloud if its quadrature points see no particle that another event adds or removes.
The quadrature box of an event has the half-width `R + r_ev` (`r_ev < dx/2`: the reach of its particles from its
centre: zero for an insertion, half the pair distance of a merge), a point of it sees the particles within `R`, the
other event's particles lie within `dx/2` of its centre, and that centre lies within `R/2` of the particle that
starts it; `dx` is a margin. Periodic axes wrap (`_collect_near!`).

The local sets of two events (the particles a remap writes: the support of its box and a ring that only takes a
share of its residuals) may overlap. The remaps run one after the other on the values the earlier ones wrote, each
of them exactly conservative on its own set, and every remap of a round knows the particles the others remove
(`removed_mask`): they are no part of its balance and get no share. (Up to 0.9.9 the spacing kept the whole local
sets apart, `R (2 + 2√D) + dx`: 13 instead of 9.5 spacings in 2-D at `R = 2.5 dx`, 14.7 instead of 10.4 in 3-D, i.e.
half respectively a third of the events per round, and every round costs a neighbour search.)
"""
@inline function _event_spacing(pg::ParticleGrid{D, M, T}) where {D, M, T}
    dx = minimum(pg.meta.dx)
    return (pg.meta.R + dx / 2) * sqrt(T(D)) + 3 * pg.meta.R / 2 + 3 * dx / 2
end

function _detect_events(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, max_order::Int, cand = 1:pg.meta.N;
                        dead::Vector{Int} = Int[], seeds::Vector{Space{D, T}} = Space{D, T}[]) where {D, M, T}
    N       = pg.meta.N
    nb      = pg.neighbor
    pos     = pg.core.positions
    isb     = pg.core.is_boundary
    orders  = pg.core.particle_orders
    dx      = minimum(pg.meta.dx)
    R       = pg.meta.R
    L, Li   = pg.domain.L_wrap, pg.domain.invL_wrap
    touched = pg.shared.bit_buffer
    fill!(view(touched, 1:N), false)

    idx, dvec = Int[], Space{D, T}[]
    # Shock band: particles whose order was lowered, plus every particle whose events could reach them. A remap
    # changes its whole local set (radius `R(1 + 2√D) + dx` around the event centre: residual correction and DMP
    # repair), and the centre lies within `R/2` of the particle that starts the event. `4R` (the former radius)
    # covers that in 1-D only.
    band = max(4R, R * (T(3) / 2 + 2 * sqrt(T(D))) + dx)
    if ad.exclude_reduced_order
        for i in 1:N
            if !isb[i] && 0 < orders[i] < max_order   # 0 = never set (fresh grid), not a drop
                n = _collect_near!(pg, pos[i], band^2, idx, dvec)
                for k in 1:n; touched[idx[k]] = true; end
            end
        end
    end

    # `dead`: particles removed with this round (the escaped ones), no part of any event. `seeds`: the positions the
    # round adds particles of its own at: no event puts a particle within the clearance of one (`near_seed` below)
    @inbounds for i in dead; touched[i] = true; end

    max_gap_sq, min_gap_sq, clear_sq = (_max_gap(ad, Val(D)) * dx)^2, (ad.min_gap * dx)^2, (_clearance(ad, Val(D)) * dx)^2
    spacing_sq = _event_spacing(pg)^2
    events = _AdaptEvent{D, T}[]
    geom, t_bnd = pg.geometry, pg.meta.boundary_time
    near_seed(x) = any(y -> sum(abs2, _wrapped_diff(y, x, L, Li)) < clear_sq, seeds)

    @inbounds for i in cand
        (isb[i] || touched[i]) && continue
        ev = nothing

        # Merge: closest too-close neighbour
        best_k, best_d2 = 0, min_gap_sq
        for k in nb.ranges[i]
            j = nb.indices[k]
            d2 = sum(abs2, nb.distances[k])
            if j > i && !isb[j] && !touched[j] && d2 < best_d2
                best_k, best_d2 = k, d2
            end
        end
        mid = best_k == 0 ? pos[i] : _wrap_point(pg, pos[i] + nb.distances[best_k] / 2)
        # (the midpoint of a pair inside a non-convex region can lie outside it: no particle is put there, it would be
        # an interior particle without a volume, out of reach of the boundary conditions)
        if best_k != 0 && _is_interior(geom, mid, t_bnd) && !near_seed(mid)
            ev = _AdaptEvent{D, T}(mid, [i, nb.indices[best_k]], [mid])
        elseif best_k == 0
            # Insertion: a far interior neighbour with an empty midpoint inside the region (the gaps towards the
            # ghosts are filled by the seeds, `_seed_candidates`)
            for k in nb.ranges[i]
                j = nb.indices[k]
                d2 = sum(abs2, nb.distances[k])
                (j > i && !isb[j] && !touched[j] && d2 > max_gap_sq) || continue
                mid = _wrap_point(pg, pos[i] + nb.distances[k] / 2)
                (_is_interior(geom, mid, t_bnd) && !near_seed(mid)) || continue
                half_d = nb.distances[k] / 2
                empty  = true
                for k2 in nb.ranges[i]                    # (as the pre-check: from the stored distance vectors)
                    if sum(abs2, nb.distances[k2] - half_d) < clear_sq
                        empty = false; break
                    end
                end
                if empty
                    ev = _AdaptEvent{D, T}(mid, Int[], [mid])
                    break
                end
            end
        end

        if ev !== nothing
            push!(events, ev)
            n = _collect_near!(pg, ev.center, spacing_sq, idx, dvec)   # independent remaps, see `_event_spacing`
            for k in 1:n; touched[idx[k]] = true; end
            length(events) >= ad.max_events && break
        end
    end
    return events
end

# -------------------------------------------------------------------------
# The remap of one event (on the OLD topology)
# -------------------------------------------------------------------------
#
# `record` (optional) collects `(index, lo, hi)` for every particle the remap wrote, with the local bounds of
# its DMP repair: the old index of a survivor, `-a` for the `a`-th entry of `new_pos`.
# `point_values` (the sweep of moving boundaries): the survivors' volumes follow the event, their states do not (they
# are point values of a collocation scheme; the conservative update is a cell-average operation that perturbs them by
# O(h) per event). The new particles take the reconstruction averaged over their share, clipped to the bounds of the
# old states within R; nothing is redistributed, so the remap is not conservative in this mode.
function _remap_event!(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, ev::_AdaptEvent{D, T},
                       new_pos::Vector{Space{D, T}}, new_rho::Vector{Vector{State{M, T}}}, new_vol::Vector{T},
                       df::Vector{State{M, T}} = State{M, T}[];
                       record::Union{Nothing, Vector{Vector{Tuple{Int, State{M, T}, State{M, T}}}}} = nothing,
                       removed_mask = nothing, point_values::Bool = false, states = _state_list(pg, ()),
                       strict::Bool = true, clear_map::Bool = true) where {D, M, T}
    nb     = pg.neighbor
    pos    = pg.core.positions
    V      = pg.core.volumes
    nS     = length(states)                           # the state arrays (the grid's own first), remapped alike
    wf     = nb.weight_func
    R      = pg.meta.R
    R_sq   = R^2
    dx     = minimum(pg.meta.dx)
    L, Li  = pg.domain.L_wrap, pg.domain.invL_wrap
    geom   = pg.geometry
    is_per = pg.domain.is_periodic
    lmap   = pg.shared.int_buffer                     # global index -> local id (0 = not involved)
    # (the neighbour search leaves prefix sums in here; every remap resets its own entries, so a caller with many
    # events clears the map once, `clear_map = false`: the pass over all particles per event was O(N) each)
    clear_map && fill!(view(lmap, 1:pg.meta.N), 0)
    n_add  = length(ev.added)

    # 1. Local set: every old particle whose shape function can change. The partition changes only within R of a
    #    removed or added particle, so the quadrature box has the half-width R + r_ev (r_ev: the event's farthest
    #    particle from its centre; 0 for the sweep's single-particle events, which were integrated over ± 2R: 8× the
    #    points and local sets of radius 12 h in 3-D) and the local set reaches R beyond the box's corners
    r_ev = zero(T)
    for r in ev.removed; r_ev = max(r_ev, sqrt(sum(abs2, _wrapped_diff(pos[r], ev.center, L, Li)))); end
    for a in ev.added;   r_ev = max(r_ev, sqrt(sum(abs2, _wrapped_diff(a, ev.center, L, Li)))); end
    half = R + r_ev
    loc_idx, loc_d = Int[], Space{D, T}[]
    n_loc = _collect_near!(pg, ev.center, (half * sqrt(T(D)) + R + dx)^2, loc_idx, loc_d)
    @inbounds for l in 1:n_loc; lmap[loc_idx[l]] = l; end
    # `removed_mask` (the sweep of moving boundaries): particles other events of the same call remove, so that no
    # content is given to a particle that the compaction drops
    is_removed = removed_mask === nothing ? falses(n_loc) : [removed_mask[loc_idx[l]] for l in 1:n_loc]
    is_other   = copy(is_removed)                          # removed by another event: gone from the balance below
    for r in ev.removed; is_removed[lmap[r]] = true; is_other[lmap[r]] = false; end
    # Boundary (ghost) particles are not remapped: their states are boundary data (`FixedDirichlet` restores them
    # from `pg.rhos`, `OutflowBC` overwrites them), so the remap must not write them. Their swept share is left out
    # of the balance below, which then puts it onto the interior particles: Σ V U over the interior is exact.
    isb      = pg.core.is_boundary
    is_fixed = [isb[loc_idx[l]] for l in 1:n_loc]

    # 2. Limited linear reconstruction data of the old solution on the local set, for every state array
    grads = [Vector{SVector{D, State{M, T}}}(undef, n_loc) for _ in 1:nS]
    lo    = [Vector{State{M, T}}(undef, n_loc) for _ in 1:nS]
    hi    = [Vector{State{M, T}}(undef, n_loc) for _ in 1:nS]
    scale = dx
    for s in 1:nS                                       # (`states` is a list of arrays: the loops run behind a barrier)
        _local_reconstruction!(grads[s], lo[s], hi[s], states[s], loc_idx, n_loc, nb, df, scale, Val(D))
    end

    # 3. Quadrature over the box center ± half (the region at `boundary_time`: moving pieces have been swept)
    t_bnd = pg.meta.boundary_time
    δV  = zeros(T, n_loc + n_add)
    I_old = zeros(T, n_loc)                             # ∫ φ_old over the box: the share a survivor can lose at most
    δm  = [zeros(State{M, T}, n_loc + n_add) for _ in 1:nS]
    hq  = dx / ad.samples_per_dx
    nq  = ceil(Int, 2half / hq)
    ω   = hq^D
    qidx, qd = Int[], Space{D, T}[]
    w_add = zeros(T, n_add)

    for I in CartesianIndices(ntuple(_ -> nq, Val(D)))
        y_raw = Space{D, T}(ntuple(d -> ev.center[d] - half + (I[d] - T(0.5)) * hq, Val(D)))
        y = _wrap_point(pg, y_raw)
        _is_interior(geom, y, t_bnd) || continue
        n = _collect_near!(pg, y, R_sq, qidx, qd)
        n == 0 && continue

        # S_old, nearest donor, S_new
        # Ghost (fixed) particles are not part of the partition: like the volume estimators, the interior particles
        # share the region among themselves (a ghost's share would leave the interior accounting)
        S_old = zero(T); S_new = zero(T)
        jstar, dstar = 0, T(Inf)
        @inbounds for k in 1:n
            l  = lmap[qidx[k]]
            (l != 0 && is_fixed[l]) && continue
            d2 = sum(abs2, qd[k])
            wk = wf(d2)
            S_old += wk
            removed_k = l != 0 && is_removed[l]
            removed_k || (S_new += wk)
            if l != 0 && !removed_k && d2 < dstar; jstar, dstar = k, d2; end
        end
        # Removed particles may be the only ones nearby: use them as donors too
        if jstar == 0
            @inbounds for k in 1:n
                d2 = sum(abs2, qd[k])
                if lmap[qidx[k]] != 0 && d2 < dstar; jstar, dstar = k, d2; end
            end
        end
        jstar == 0 && continue
        @inbounds for a in 1:n_add
            da = _wrapped_diff(ev.added[a], y, L, Li)
            d2 = sum(abs2, da)
            w_add[a] = d2 <= R_sq ? wf(d2) : zero(T)
            S_new += w_add[a]
        end
        (S_old > 0 && S_new > 0) || continue
        inv_old, inv_new = inv(S_old), inv(S_new)
        lstar = lmap[qidx[jstar]]

        for s in 1:nS
            _swept_point!(δm[s], δV, I_old, s == 1, states[s], grads[s], lo[s], hi[s], loc_idx, lstar, qd[jstar], qidx, qd, n,
                          lmap, is_removed, is_fixed, wf, inv_old, inv_new, ω, w_add, n_loc)
        end
    end

    # 4. New volumes and contents; exact residual correction (the volumes from the first pass, the contents per array)
    V_before = zero(T)
    @inbounds for l in 1:n_loc
        (is_fixed[l] || is_other[l]) && continue           # (other events' removals: counting them once per
        V_before += V[loc_idx[l]]                          # overlapping event handed their volume out repeatedly)
    end
    # A survivor's volume changes by its swept share δV, relative to the larger of its volume and its share of the box
    # (∫ φ_old ≥ −δV): the same as V + δV for volumes that are the partition's (V ≥ ∫ φ_old over the box), and a
    # positive volume for the others. Volumes that are not the remap's partition (an estimator of higher degree, the
    # scheme's own of the content form) can be smaller than the share an event takes away; `max(V + δV, 1e-12 V)` then
    # left a volume of 1e-12 V with the content of V + δV < 0, and the next event near it another factor (volumes of
    # 1e-148 on the Sedov blast, states of content / volume, "Remap produced no volume" in the end). The swept
    # content is scaled alike, and the residual correction below keeps both sums exact.
    Vn = zeros(T, n_loc + n_add)
    fV = ones(T, n_loc)
    V_after = zero(T)
    @inbounds for l in 1:n_loc
        (is_removed[l] || is_fixed[l]) && continue
        j = loc_idx[l]
        fV[l] = (V[j] > 0 && I_old[l] > V[j]) ? V[j] / I_old[l] : one(T)
        Vn[l] = max(V[j] + fV[l] * δV[l], zero(T))
        V_after += Vn[l]
    end
    @inbounds for a in 1:n_add
        Vn[n_loc + a] = max(δV[n_loc + a], zero(T))
        V_after += Vn[n_loc + a]
    end
    # `strict = false` (the adaptivity): an event that cannot be remapped is refused, nothing is written and `false` is
    # returned. That is an event whose particles have no usable volumes (non-finite, or none at all: an interior
    # particle outside the region that the estimator gave nothing), or whose new particle gets no share of the
    # region (it would be a particle with the state 0/0). More quadrature points do not help there.
    if !(V_after > 0 && isfinite(V_after) && isfinite(V_before))
        strict && error("Remap produced no volume at $(ev.center); increase samples_per_dx.")
        @inbounds for l in 1:n_loc; lmap[loc_idx[l]] = 0; end
        return false
    end
    if !strict
        vmin = T(1e-6) * hq^D
        ok = true
        @inbounds for l in 1:n_loc
            (is_removed[l] || is_fixed[l]) && continue
            (Vn[l] > 0 && isfinite(Vn[l])) || (ok = false; break)
        end
        @inbounds for a in 1:n_add
            Vn[n_loc + a] > vmin || (ok = false)
        end
        if !ok
            @inbounds for l in 1:n_loc; lmap[loc_idx[l]] = 0; end
            return false
        end
    end
    εV = V_before - V_after
    mn = [zeros(State{M, T}, n_loc + n_add) for _ in 1:nS]
    for s in 1:nS
        U = states[s]
        m_before = zero(State{M, T}); m_after = zero(State{M, T})
        @inbounds for l in 1:n_loc
            (is_fixed[l] || is_other[l]) && continue
            j = loc_idx[l]
            m_before += V[j] * U[j]
        end
        @inbounds for l in 1:n_loc
            (is_removed[l] || is_fixed[l]) && continue
            j = loc_idx[l]
            mn[s][l] = V[j] * U[j] + fV[l] * δm[s][l]
            m_after += mn[s][l]
        end
        @inbounds for a in 1:n_add
            mn[s][n_loc + a] = δm[s][n_loc + a]
            m_after += mn[s][n_loc + a]
        end
        εm = m_before - m_after
        @inbounds for l in 1:(n_loc + n_add)
            (l <= n_loc && (is_removed[l] || is_fixed[l])) && continue
            mn[s][l] += (Vn[l] / V_after) * εm
        end
    end
    @inbounds for l in 1:(n_loc + n_add)
        (l <= n_loc && (is_removed[l] || is_fixed[l])) && continue
        Vn[l] += (Vn[l] / V_after) * εV
    end

    # 5. DMP repair on the affected set (conservative redistribution), per state array
    aff = Int[]                                         # local ids of interior survivors + added
    for l in 1:n_loc; (is_removed[l] || is_fixed[l]) || push!(aff, l); end
    for a in 1:n_add; push!(aff, n_loc + a); end
    V_aff = [Vn[l] for l in aff]
    Un_s  = Vector{Vector{State{M, T}}}(undef, nS)
    lo_s  = Vector{Vector{State{M, T}}}(undef, nS)
    hi_s  = Vector{Vector{State{M, T}}}(undef, nS)
    for s in 1:nS
        U  = states[s]
        Un = [mn[s][l] / Vn[l] for l in aff]
        lo_a = Vector{State{M, T}}(undef, length(aff)); hi_a = similar(lo_a)
        for (p, l) in enumerate(aff)
            if l <= n_loc
                lo_a[p], hi_a[p] = lo[s][l], hi[s][l]
            else
                # Bounds for a new particle: old states within R of it
                n = _collect_near!(pg, ev.added[l - n_loc], R_sq, qidx, qd)
                if n == 0
                    lo_a[p], hi_a[p] = Un[p], Un[p]
                else
                    a_lo = U[qidx[1]]; a_hi = a_lo
                    for k in 2:n; a_lo = math_min.(a_lo, U[qidx[k]]); a_hi = math_max.(a_hi, U[qidx[k]]); end
                    lo_a[p], hi_a[p] = a_lo, a_hi
                end
            end
        end
        if point_values
            for p in eachindex(aff); Un[p] = math_max.(lo_a[p], math_min.(hi_a[p], Un[p])); end
        else
            _dmp_repair!(Un, V_aff, lo_a, hi_a)
        end
        Un_s[s], lo_s[s], hi_s[s] = Un, lo_a, hi_a
    end
    if !strict && !all(Un -> all(u -> all(isfinite, u), Un), Un_s)      # (non-finite old states in the local set)
        @inbounds for l in 1:n_loc; lmap[loc_idx[l]] = 0; end
        return false
    end

    # 6. Write back (the volumes with the first state array)
    for s in 1:nS
        U = states[s]
        Un, lo_a, hi_a = Un_s[s], lo_s[s], hi_s[s]
        for (p, l) in enumerate(aff)
            if l <= n_loc
                j = loc_idx[l]
                s == 1 && (V[j] = Vn[l])
                point_values || (U[j] = Un[p])
                record === nothing || push!(record[s], (j, lo_a[p], hi_a[p]))
            else
                s == 1 && (push!(new_pos, ev.added[l - n_loc]); push!(new_vol, Vn[l]))
                push!(new_rho[s], Un[p])
                record === nothing || push!(record[s], (-length(new_pos), lo_a[p], hi_a[p]))
            end
        end
    end
    @inbounds for l in 1:n_loc; lmap[loc_idx[l]] = 0; end
    return true
end

# The limited linear reconstruction data of one state array on the local set of an event (`_remap_event!`, step 2): the
# gradient and the bounds of every local particle from its neighbours. `df` is a scratch of one neighbour list.
function _local_reconstruction!(grads::Vector{SVector{D, State{M, T}}}, lo::Vector{State{M, T}}, hi::Vector{State{M, T}},
                                U::AbstractVector{State{M, T}}, loc_idx, n_loc::Int, nb, df::Vector{State{M, T}}, scale::T,
                                ::Val{D}) where {D, M, T}
    @inbounds for l in 1:n_loc
        j  = loc_idx[l]
        sl = nb.ranges[j]
        mn, mx = U[j], U[j]
        # (the differences of this particle's slots, in a scratch of the size of one neighbour list: a scratch over
        # all slots of the grid, allocated per call, was the largest single cost of the particle management)
        length(df) < length(sl) && resize!(df, length(sl))
        for (q, k) in enumerate(sl)
            uk = U[nb.indices[k]]
            df[q] = uk - U[j]
            mn = math_min.(mn, uk); mx = math_max.(mx, uk)
        end
        lo[l], hi[l] = mn, mx
        g = isempty(sl) ? zero(SVector{D, State{M, T}}) :
            Interpolator{D, 1, 1}()(1:length(sl), view(nb.distances, sl), view(nb.weights, sl), df; scale = scale)
        grads[l] = SVector{D, State{M, T}}(ntuple(d -> g[d], Val(D)))
    end
    return nothing
end

# The swept contributions of one quadrature point to one state array (`_remap_event!`, step 3): Ũ(y), the limited
# linear reconstruction of the nearest old particle `lstar` (at `d_star = x − y`), times the change of every shape
# function at the point; the volumes with the first array (`first`)
function _swept_point!(δm::Vector{State{M, T}}, δV::Vector{T}, I_old::Vector{T}, first::Bool, U::AbstractVector{State{M, T}},
                       grads::Vector{SVector{D, State{M, T}}}, lo::Vector{State{M, T}}, hi::Vector{State{M, T}}, loc_idx,
                       lstar::Int, d_star::Space{D, T}, qidx, qd, n::Int, lmap, is_removed, is_fixed, wf, inv_old::T, inv_new::T,
                       ω::T, w_add::Vector{T}, n_loc::Int) where {D, M, T}
    @inbounds begin
        u_rec = U[loc_idx[lstar]]
        for d in 1:D
            u_rec -= d_star[d] * grads[lstar][d]            # d_star = x_j − y  ⇒  y − x_j = −d_star
        end
        u_rec = math_max.(lo[lstar], math_min.(hi[lstar], u_rec))
        for k in 1:n
            l = lmap[qidx[k]]
            (l == 0 || is_removed[l] || is_fixed[l]) && continue
            wk = wf(sum(abs2, qd[k]))
            dφ = wk * inv_new - wk * inv_old
            first && (δV[l] += ω * dφ; I_old[l] += ω * wk * inv_old)
            δm[l] += (ω * dφ) * u_rec
        end
        for a in eachindex(w_add)
            dφ = w_add[a] * inv_new
            first && (δV[n_loc + a] += ω * dφ)
            δm[n_loc + a] += (ω * dφ) * u_rec
        end
    end
    return nothing
end

# The single-state forms (the grid's own state only), as before the state arrays were threaded through
_remap_event!(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, ev::_AdaptEvent{D, T}, new_pos::Vector{Space{D, T}},
              new_rho::Vector{State{M, T}}, new_vol::Vector{T}, df::Vector{State{M, T}} = State{M, T}[];
              kwargs...) where {D, M, T} =
    _remap_event!(pg, ad, ev, new_pos, [new_rho], new_vol, df; kwargs...)

"""
    _dmp_repair!(U, V, lo, hi)

Clips every `U[p]` to `[lo[p], hi[p]]` and redistributes the clipped content `Σ V (U − U_clip)`
to the particles that still have room, keeping `Σ V U` exact. If there is not enough room, the
remainder is spread proportionally to volume (bounds are then violated by that remainder).
"""
function _dmp_repair!(U::Vector{State{M, T}}, V::Vector{T}, lo::Vector{State{M, T}}, hi::Vector{State{M, T}}) where {M, T}
    n = length(U)
    Um = [MVector{M, T}(u) for u in U]
    for m in 1:M
        excess = zero(T)
        for p in 1:n
            u = Um[p][m]
            if u > hi[p][m]
                excess += V[p] * (u - hi[p][m]); Um[p][m] = hi[p][m]
            elseif u < lo[p][m]
                excess += V[p] * (u - lo[p][m]); Um[p][m] = lo[p][m]
            end
        end
        # Round-off relative to the set's content (an absolute eps(T) dropped real excesses of small-volume sets)
        scale = zero(T)
        for p in 1:n
            scale += V[p] * max(abs(hi[p][m]), abs(lo[p][m]))
        end
        abs(excess) <= 4 * eps(T) * scale && continue
        room = zero(T)
        for p in 1:n
            room += excess > 0 ? V[p] * (hi[p][m] - Um[p][m]) : V[p] * (Um[p][m] - lo[p][m])
        end
        if room >= abs(excess) && room > 0
            f = excess / room
            for p in 1:n
                r = excess > 0 ? (hi[p][m] - Um[p][m]) : (Um[p][m] - lo[p][m])
                Um[p][m] += f * r
            end
        else
            # Fill all rooms, spread the remainder by volume
            Vtot = sum(V)
            rem = excess - (excess > 0 ? room : -room)
            for p in 1:n
                Um[p][m] = excess > 0 ? hi[p][m] : lo[p][m]
                Um[p][m] += rem / Vtot
            end
        end
    end
    for p in 1:n; U[p] = State{M, T}(Um[p]); end
    return nothing
end

# -------------------------------------------------------------------------
# Topology change: compaction of removed particles, append of new ones
# -------------------------------------------------------------------------
function _apply_topology!(pg::ParticleGrid{D, M, T}, remove_mask::AbstractVector{Bool},
                          new_pos::Vector{Space{D, T}}, new_rho::Vector{Vector{State{M, T}}}, new_vol::Vector{T},
                          max_order::Int; states = _state_list(pg, ())) where {D, M, T}
    N     = pg.meta.N
    n_add = length(new_pos)
    core  = pg.core
    N_new = N - count(view(remove_mask, 1:N)) + n_add
    ensure_capacity!(pg, N_new)
    ensure_particle_capacity!(pg.neighbor, N_new)
    for st in states
        _grow_state!(st, N_new)
    end

    w = 1
    @inbounds for i in 1:N
        remove_mask[i] && continue
        if w != i
            core.positions[w]       = core.positions[i]
            core.is_boundary[w]     = core.is_boundary[i]
            core.tags[w]            = core.tags[i]
            core.particle_orders[w] = core.particle_orders[i]
            core.volumes[w]         = core.volumes[i]
            core.normals[w]         = core.normals[i]
            core.wall_distances[w]  = core.wall_distances[i]
            core.wall_curvatures[w] = core.wall_curvatures[i]
            for st in states
                st[w] = st[i]
            end
            pg.curvatures[w]        = pg.curvatures[i]
        end
        w += 1
    end
    @inbounds for a in 1:n_add
        core.positions[w]       = new_pos[a]
        core.is_boundary[w]     = false
        core.tags[w]            = 0
        core.particle_orders[w] = max_order
        core.volumes[w]         = new_vol[a]
        core.normals[w], core.wall_distances[w], core.wall_curvatures[w] = _ghost_geometry_curv(pg.geometry, new_pos[a], pg.meta.boundary_time)
        for (s, st) in enumerate(states)
            st[w] = new_rho[s][a]
        end
        pg.curvatures[w]        = zero(State{M, T})
        w += 1
    end
    pg.meta.N          = N_new
    pg.meta.N_interior = count(!, view(core.is_boundary, 1:N_new))
    pg.meta.N_ghost    = N_new - pg.meta.N_interior
    return nothing
end
