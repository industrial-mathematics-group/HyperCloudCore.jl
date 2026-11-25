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
    ParticleAdaptivity(; max_gap = 1.6, min_gap = 0.5, clearance = 0.7, samples_per_dx = 2,
                         exclude_reduced_order = true, max_events = typemax(Int), max_rounds = 8,
                         volume_estimator = nothing)

Rules for keeping a moving point cloud well-spaced. Lengths are multiples of the particle spacing
`dx = minimum(pg.meta.dx)`.
- **Insertion**: a neighbour pair `(i, j)` farther apart than `max_gap·dx` whose midpoint has no
  particle within `clearance·dx` gets a new particle at the midpoint.
- **Merge**: a neighbour pair closer than `min_gap·dx` is replaced by one particle at the midpoint.
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
  many times. The events of one round are kept apart (below), a few per `(13 dx)^D` in 2-D at `R = 2.5 dx`, and a pair
  can only be found while it is within the radius `R`: a gap between `max_gap·dx` and `R` that one round per step
  does not reach in time grows beyond `R` and is never filled (an expanding cloud with a small radius lost its
  stencils that way). `1` is the behaviour up to 0.9.9.
- `volume_estimator`: the estimator the time stepper re-estimates the volumes with (a moving grid). After
  the topology change the volumes of the new cloud are estimated with it, and the change of `Σ V U` this
  causes is put back onto the particles the remap changed (bounded by their local extrema, like the remap's
  own repair), so the call is exact for the volumes the next steps use. `nothing` keeps the remap's own
  volumes (right for a static grid, whose volumes are never re-estimated).
- At most one event per neighbourhood and per round. Event centres are kept `R (2 + 2√D) + dx` apart (see
  `_event_spacing`), so no event's remap region contains a particle that another event removes or adds:
  the remaps are independent and the call is exactly conservative. Call once between time steps.
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

function ParticleAdaptivity(::Type{T} = Float64; max_gap = 1.6, min_gap = 0.5, clearance = 0.7,
                            samples_per_dx::Int = 2, exclude_reduced_order::Bool = true,
                            max_events::Int = typemax(Int), max_rounds::Int = 8,
                            volume_estimator::Union{Nothing, VolumeEstimator} = nothing) where {T}
    @assert min_gap < 1 < max_gap "need min_gap < 1 < max_gap (in units of dx)"
    @assert max_events >= 1 "max_events must be at least 1"
    @assert max_rounds >= 1 "max_rounds must be at least 1"
    return ParticleAdaptivity{T, typeof(volume_estimator)}(T(max_gap), T(min_gap), T(clearance), samples_per_dx,
                                                         exclude_reduced_order, max_events, max_rounds, volume_estimator)
end

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
re-estimated after each ([`keeps_own_volumes`](@ref): the content form of the Lagrangian movers). With
`ad.volume_estimator` only the particles the remaps changed then take the estimator's volumes, and the correction
that keeps `Σ V U` balances their change alone. `false` (the stepper re-estimates all volumes after every step, with
the same estimator): all volumes are replaced, as the next step would, and the correction balances the whole change.

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
    (isempty(escaped) && !_has_adapt_candidates(pg, ad)) && return nothing_done

    on_host = parent(pg.rhos) isa Array
    (on_host || _host_states(states)) ||
        throw(ArgumentError("adapt_particles!: extra state arrays of a device grid must be host arrays (the events run on a host copy)"))
    hg = on_host ? pg : device_grid(Array, pg)            # the event work is serial: on a host copy of a device grid
    st_list = states isa _LazyStates ? states : _state_list(hg, states)

    # Particles that left the region first, then the seeds in the gaps next to the boundary (the events below are
    # detected on a cloud that is inside the region and reaches its boundary)
    held, dropped, lost = 0, 0, zero(State{M, T})
    if !isempty(escaped)
        held, dropped, lost = _handle_escaped!(hg, escaped, max_order, _states_now(st_list, hg))
    end
    mass0    = interior_mass(hg)
    seed_ids = _seed_boundary!(pg, hg, ad, max_order, st_list)
    inflow   = interior_mass(hg) - mass0                  # the content the seeds brought (new volume: the gap they fill)

    # Rounds of independent events: detection on the current cloud, the remaps on its topology, the topology change
    # and the neighbour search, until no pair violates the rules (the volumes are re-estimated once, at the end)
    n_refused, n_removed, n_added = 0, 0, 0
    seeds   = Set{Int}(seed_ids)
    changed = ad.volume_estimator === nothing ? nothing : Dict{Int, Tuple{State{M, T}, State{M, T}}}[]
    for _ in 1:ad.max_rounds
        events = _detect_events(hg, ad, max_order)
        isempty(events) && break
        N = hg.meta.N
        all_states = _states_now(st_list, hg)
        # Capacity first: the remaps below rewrite states and volumes in place, so a `CapacityError` (buffers frozen
        # by `reserve!`) thrown by `_apply_topology!` would leave the grid in a non-conservative intermediate state
        N_new = N + sum(ev -> length(ev.added) - length(ev.removed), events)
        if N_new > N
            ensure_capacity!(hg, N_new)
            on_host || ensure_capacity!(pg, N_new)
        end
        # ... and the neighbour slots the search after the topology change will need (in the layout of `pg`)
        total_nb, max_nb = _slots_after_events(hg, events)
        ensure_capacity!(pg.neighbor, _slot_capacity(eltype(pg.neighbor.ranges), total_nb, max_nb, N_new))

        remove_mask = hg.shared.bit_buffer
        fill!(view(remove_mask, 1:N), false)
        new_pos, new_vol = Space{D, T}[], T[]
        new_rho = [State{M, T}[] for _ in all_states]
        df = Vector{State{M, T}}(undef, length(hg.neighbor.indices))   # neighbour-slot scratch shared by all events

        record = changed === nothing ? nothing : [Tuple{Int, State{M, T}, State{M, T}}[] for _ in all_states]
        refused = 0
        for ev in events
            # (an event that cannot be remapped, e.g. next to a particle without a volume, is left out: nothing is written)
            if _remap_event!(hg, ad, ev, new_pos, new_rho, new_vol, df; record = record, states = all_states, strict = false)
                for r in ev.removed; remove_mask[r] = true; end
            else
                refused += 1
            end
        end
        n_refused += refused
        n_rem = count(view(remove_mask, 1:N))
        (n_rem + length(new_pos)) == 0 && break           # (every event refused: the next round would find them again)
        n_removed += n_rem
        n_added   += length(new_pos)
        # the particles written so far, by their indices after this round's compaction
        seeds = _compacted_ids(collect(seeds), remove_mask, N, N - n_rem)
        if changed !== nothing
            now = [_compacted_record(r, remove_mask, N, N - n_rem) for r in record]
            isempty(changed) ? append!(changed, now) :
                (changed .= [_merge_bounds!(_compacted_bounds(changed[s], remove_mask, N), now[s]) for s in eachindex(now)])
        end
        _apply_topology!(hg, remove_mask, new_pos, new_rho, new_vol, max_order; states = all_states)
        hg.neighbor(hg)
    end
    (held + dropped + length(seed_ids) + n_removed + n_added) == 0 && return _adapt_result(State{M, T}; refused = n_refused)

    all_states = _states_now(st_list, hg)
    on_host || (_upload_particles!(pg, hg); pg.neighbor(pg))
    if changed !== nothing && (!isempty(changed) || !isempty(seeds))
        isempty(changed) && append!(changed, (Dict{Int, Tuple{State{M, T}, State{M, T}}}() for _ in all_states))
        inflow += _reestimate_volumes!(pg, hg, ad.volume_estimator, changed; states = all_states, seeds = seeds,
                                       own_volumes = own_volumes)
    end

    return _adapt_result(State{M, T}; inserted = n_added, removed = n_removed,
                         mass_residual = interior_mass(pg) - mass0 - inflow, escaped = dropped, held = held,
                         seeded = length(seeds), refused = n_refused, escaped_content = lost, seeded_content = inflow)
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

# Puts the escaped particles `escaped` back onto their wall or removes them (with their content, which leaves the
# region: an outflow), and searches the neighbours again. Returns the numbers held and removed and the content removed.
function _handle_escaped!(hg::ParticleGrid{D, M, T}, escaped::Vector{Int}, max_order::Int, states) where {D, M, T}
    geom, t = hg.geometry, hg.meta.boundary_time
    pos, V  = hg.core.positions, hg.core.volumes
    N       = hg.meta.N
    mask    = hg.shared.bit_buffer
    fill!(view(mask, 1:N), false)
    held, dropped, lost = 0, 0, zero(State{M, T})
    for i in escaped
        y = _hold_at_wall(geom, pos[i], t)
        if y === nothing
            mask[i] = true
            dropped += 1
            lost += V[i] * states[1][i]
        else
            pos[i] = y
            held += 1
        end
    end
    dropped > 0 && _apply_topology!(hg, mask, Space{D, T}[], [State{M, T}[] for _ in states], T[], max_order; states = states)
    hg.neighbor(hg)
    return held, dropped, lost
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
    for i in 1:N
        i in removed && continue
        n = length(nb.ranges[i]) + get(delta, i, 0)
        total += n
        max_nb = max(max_nb, n)
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
    _seed_boundary!(pg, hg, ad, max_order, states) -> Vector{Int}

New interior particles in the gaps between the cloud and the boundary: where an interior particle and a ghost among
its neighbours are farther apart than `max_gap·dx`, with the midpoint inside the region and no particle (or earlier
seed) within `clearance·dx` of it. A moving cloud recedes from a boundary with inflow; without the seeds the first
interior row loses the ghosts from its stencils, and the volume estimators extrapolate into the gap.

A seed is not a remap event: the gap is volume no particle of the cloud holds, and what fills it enters the region.
Its state is the mean of the pair's (the ghost's is the boundary datum, the interior particle's the solution next to
it), its volume the nominal `prod(dx)` until an estimator replaces it (`_reestimate_volumes!`). A whole row is
seeded in one call (the event spacing of the conservative remaps allowed a few per boundary and call: a cloud
receding at the flow speed opened the gap faster). Returns the indices of the seeds; the neighbours are searched
again if there are any.
"""
function _seed_boundary!(pg::ParticleGrid{D, M, T}, hg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, max_order::Int, st_list) where {D, M, T}
    (hg.mover isa NoGridMover || hg.meta.N_ghost == 0) && return Int[]
    N, nb   = hg.meta.N, hg.neighbor
    pos, isb, orders = hg.core.positions, hg.core.is_boundary, hg.core.particle_orders
    dx      = minimum(hg.meta.dx)
    L, Li   = hg.domain.L_wrap, hg.domain.invL_wrap
    geom, t = hg.geometry, hg.meta.boundary_time
    max_gap_sq, clear_sq = (ad.max_gap * dx)^2, (ad.clearance * dx)^2
    new_pos = Space{D, T}[]
    pairs   = Tuple{Int, Int}[]
    @inbounds for i in 1:N
        isb[i] && continue
        (ad.exclude_reduced_order && 0 < orders[i] < max_order) && continue
        for k in nb.ranges[i]
            j = nb.indices[k]
            (isb[j] && sum(abs2, nb.distances[k]) > max_gap_sq) || continue
            mid = _wrap_point(hg, pos[i] + nb.distances[k] / 2)
            _is_interior(geom, mid, t) || continue
            empty = true
            for k2 in nb.ranges[i]
                if sum(abs2, _wrapped_diff(pos[nb.indices[k2]], mid, L, Li)) < clear_sq
                    empty = false; break
                end
            end
            empty || continue
            any(x -> sum(abs2, _wrapped_diff(x, mid, L, Li)) < clear_sq, new_pos) && continue
            push!(new_pos, mid); push!(pairs, (i, j))
            break                                         # one seed per interior particle and call
        end
    end
    isempty(new_pos) && return Int[]
    states = _states_now(st_list, hg)
    N_new  = N + length(new_pos)
    # (capacity before the grid is changed, as for the events)
    ensure_capacity!(hg, N_new)
    hg === pg || ensure_capacity!(pg, N_new)
    total_nb, max_nb = _slots_after_events(hg, [_AdaptEvent{D, T}(x, Int[], [x]) for x in new_pos])
    ensure_capacity!(pg.neighbor, _slot_capacity(eltype(pg.neighbor.ranges), total_nb, max_nb, N_new))
    new_rho = [[(st[i] + st[j]) / 2 for (i, j) in pairs] for st in states]
    mask = hg.shared.bit_buffer
    fill!(view(mask, 1:N), false)
    _apply_topology!(hg, mask, new_pos, new_rho, fill(prod(hg.meta.dx), length(new_pos)), max_order; states = states)
    hg.neighbor(hg)
    return collect((N + 1):N_new)
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

`own_volumes` (the stepper keeps its volumes between the steps, the scheme's own of the content form): only the
particles the remaps changed and the seeds take the estimator's volumes, every other particle keeps its volume, and
the correction balances the change of the changed particles. Replacing all volumes there (as up to 0.9.9) put the
mismatch between the scheme's and the estimator's volumes of every particle of the cloud onto the changed particles.
Otherwise all volumes are replaced, as the stepper's next re-estimation would, and the correction balances the whole
change (an estimator renormalises all volumes when the particle set changes).

The seeds (`seeds`, see `_seed_boundary!`) take their volume without a correction: the gap they fill is new volume,
and its content enters the region. An estimate without a volume (clamped, or a particle no sample point reaches) is
not taken. Returns the content that entered with the seeds' volumes (of the first state array).
"""
function _reestimate_volumes!(pg::ParticleGrid{D, M, T}, hg::ParticleGrid{D, M, T}, est::VolumeEstimator,
                              changed::Vector{Dict{Int, Tuple{State{M, T}, State{M, T}}}}; states = _state_list(hg, ()),
                              seeds::Set{Int} = Set{Int}(), own_volumes::Bool = false) where {D, M, T}
    N = pg.meta.N
    on_host = hg === pg
    V_remap = collect(view(hg.core.volumes, 1:N))               # (hg holds the same cloud and volumes as pg)
    update_volumes!(pg, est)
    on_host || copyto!(parent(hg.core.volumes), 1, parent(pg.core.volumes), 1, N)
    touched = Set{Int}(seeds)
    for c in changed; union!(touched, keys(c)); end
    Vh = hg.core.volumes
    @inbounds for i in 1:N
        ((!own_volumes || i in touched) && Vh[i] > 0 && isfinite(Vh[i])) || (Vh[i] = V_remap[i])
        # A stepper that keeps the volumes uses them as they are (the pair-antisymmetric dissipation of the Lagrangian
        # frame weights a pair with V_j / V_i), and an estimator of higher degree gives single particles next to a
        # boundary or a hole volumes of 1e-3 of their neighbours' (MLS of degree 2, C6 at R = 2.5 dx: one such particle
        # went to a density of −18 in the next step). The estimate is taken within a factor 8 of the remap's volume
        # (the particle's share of the partition, the nominal volume of a seed).
        (own_volumes && V_remap[i] > 0) && (Vh[i] = clamp(Vh[i], V_remap[i] / 8, 8 * V_remap[i]))
    end
    isb = hg.core.is_boundary
    inflow = zero(State{M, T})                                  # the seeds' change of content (of the first array)
    for i in seeds; inflow += (Vh[i] - V_remap[i]) * states[1][i]; end
    for (s, st) in enumerate(states)
        ids = [i for i in keys(changed[s]) if !(i in seeds)]
        ε = zero(State{M, T})                                   # the change of content to balance: of the changed
        if own_volumes                                          # particles, or of the whole cloud (without the seeds)
            for i in ids; ε += (V_remap[i] - Vh[i]) * st[i]; end
        else
            @inbounds for i in 1:N
                (isb[i] || i in seeds) || (ε += (V_remap[i] - Vh[i]) * st[i])
            end
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
    foreach_particle(_adapt_candidate!, get_backend(), N,
        (flag, pg.core.positions, pg.core.is_boundary, nb.ranges, nb.indices, nb.distances, dom.L_wrap, dom.invL_wrap,
         dom.is_periodic, dom.L, dom.canvas_mins, (ad.min_gap * dx)^2, (ad.max_gap * dx)^2, (ad.clearance * dx)^2,
         (geom.mins, geom.maxs)))
    return any(view(flag, 1:N))
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
                mid   = _wrap_coords(pos[i] + dists[k] / 2, is_per, Lv, mins)
                # (towards a ghost only with the midpoint inside the region's box: the band of ghosts has such pairs
                # at every boundary; the region test itself is the host's, in `_detect_events`)
                (isb[j] && !_in_box(mid, box[1], box[2], is_per)) && continue
                empty = true
                for k2 in ranges[i]
                    if sum(abs2, _wrapped_diff(pos[indices[k2]], mid, L, Li)) < clear_sq
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

Radius around an accepted event centre within which no particle may start another event in the same call:
`R (2 + 2√D) + dx`. Events are remapped one after the other on the old topology, so two remaps are only
independent if neither sees a particle that the other removes or adds. The remap of an event at `c'` reads
every particle within `(R + r_ev) √D + R + dx` of `c'` (its local set: the support of its quadrature box of
half-width `R + r_ev`, `r_ev ≤ R/2` the reach of the event's particles; the former box of half-width `2R` gave
`R (1 + 2√D) + dx`, which this spacing keeps) and gives a share of its swept content to each of them; a particle
that an earlier event removed would take that share with it when the particle set is compacted. The changed
particles of an event lie within `R/2` of its centre, and a later event's centre lies within `R/2` of the particle
that starts it,
hence the extra `R`. Periodic axes wrap (`_collect_near!`). In 1-D this is close to the former `4R`; in 2-D and
3-D the former `4R` let the local sets of events about `3.5R` apart overlap, which is where calls with
several events were not exactly conservative.
"""
@inline _event_spacing(pg::ParticleGrid{D, M, T}) where {D, M, T} =
    pg.meta.R * (2 + 2 * sqrt(T(D))) + minimum(pg.meta.dx)

function _detect_events(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, max_order::Int) where {D, M, T}
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

    max_gap_sq, min_gap_sq, clear_sq = (ad.max_gap * dx)^2, (ad.min_gap * dx)^2, (ad.clearance * dx)^2
    spacing_sq = _event_spacing(pg)^2
    events = _AdaptEvent{D, T}[]
    geom, t_bnd = pg.geometry, pg.meta.boundary_time

    @inbounds for i in 1:N
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
        if best_k != 0 && _is_interior(geom, mid, t_bnd)
            ev = _AdaptEvent{D, T}(mid, [i, nb.indices[best_k]], [mid])
        elseif best_k == 0
            # Insertion: a far interior neighbour with an empty midpoint inside the region (the gaps towards the
            # ghosts are filled by `_seed_boundary!`)
            for k in nb.ranges[i]
                j = nb.indices[k]
                d2 = sum(abs2, nb.distances[k])
                (j > i && !isb[j] && !touched[j] && d2 > max_gap_sq) || continue
                mid = _wrap_point(pg, pos[i] + nb.distances[k] / 2)
                _is_interior(geom, mid, t_bnd) || continue
                empty = true
                for k2 in nb.ranges[i]
                    if sum(abs2, _wrapped_diff(pos[nb.indices[k2]], mid, L, Li)) < clear_sq
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
                       df::Vector{State{M, T}} = Vector{State{M, T}}(undef, length(pg.neighbor.indices));
                       record::Union{Nothing, Vector{Vector{Tuple{Int, State{M, T}, State{M, T}}}}} = nothing,
                       removed_mask = nothing, point_values::Bool = false, states = _state_list(pg, ()),
                       strict::Bool = true) where {D, M, T}
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
    fill!(view(lmap, 1:pg.meta.N), 0)                 # the neighbour search leaves prefix sums in here
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
    for s in 1:nS
        U = states[s]
        @inbounds for l in 1:n_loc
            j  = loc_idx[l]
            sl = nb.ranges[j]
            mn, mx = U[j], U[j]
            for k in sl
                uk = U[nb.indices[k]]
                df[k] = uk - U[j]
                mn = math_min.(mn, uk); mx = math_max.(mx, uk)
            end
            lo[s][l], hi[s][l] = mn, mx
            g = isempty(sl) ? zero(SVector{D, State{M, T}}) :
                Interpolator{D, 1, 1}()(sl, nb.distances, nb.weights, df; scale = scale)
            grads[s][l] = SVector{D, State{M, T}}(ntuple(d -> g[d], Val(D)))
        end
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
            U = states[s]
            # Ũ(y): limited linear reconstruction of the nearest old particle
            u_rec = U[loc_idx[lstar]]
            @inbounds for d in 1:D
                u_rec -= qd[jstar][d] * grads[s][lstar][d]  # qd = x_j − y  ⇒  y − x_j = −qd
            end
            u_rec = math_max.(lo[s][lstar], math_min.(hi[s][lstar], u_rec))

            # Accumulate swept contributions (the volumes once, with the first state array)
            @inbounds for k in 1:n
                l = lmap[qidx[k]]
                (l == 0 || is_removed[l] || is_fixed[l]) && continue
                wk = wf(sum(abs2, qd[k]))
                dφ = wk * inv_new - wk * inv_old
                s == 1 && (δV[l] += ω * dφ; I_old[l] += ω * wk * inv_old)
                δm[s][l] += (ω * dφ) * u_rec
            end
            @inbounds for a in 1:n_add
                dφ = w_add[a] * inv_new
                s == 1 && (δV[n_loc + a] += ω * dφ)
                δm[s][n_loc + a] += (ω * dφ) * u_rec
            end
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

# The single-state forms (the grid's own state only), as before the state arrays were threaded through
_remap_event!(pg::ParticleGrid{D, M, T}, ad::ParticleAdaptivity{T}, ev::_AdaptEvent{D, T}, new_pos::Vector{Space{D, T}},
              new_rho::Vector{State{M, T}}, new_vol::Vector{T}, df::Vector{State{M, T}} = Vector{State{M, T}}(undef, length(pg.neighbor.indices));
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
