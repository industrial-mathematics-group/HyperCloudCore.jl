export place_ghosts!, sweep_boundaries!

# =========================================================================
# MOVING BOUNDARIES — ghost particles that follow their piece, interior particles that make way
# =========================================================================
#
# A `Moving` piece of the domain's shape is a rigid body with a prescribed motion. The cloud stays Eulerian:
#   * its ghost particles are placed at the piece's pose at every stage time (`place_ghosts!`, exact rigid
#     motion of the positions and normals from the step start: no drift);
#   * after each step the interior particles the piece has covered, or come closer to than a quarter spacing,
#     are removed, and the region it has left is refilled with the lattice points of the canvas that are at
#     least half a spacing inside and half a spacing from every particle (`sweep_boundaries!`); both through
#     the conservative remap of the adaptivity, so the content of the region is preserved exactly.
# The boundary conditions see the ghosts at the time of the state they build (`tag_motion`), with the wall
# velocity in the `WallPoint`.

"""
    place_ghosts!(pg, X, t_from, t_to)

Moves the ghost particles of the moving pieces in the position buffer `X` (their positions at `t_from`) to the
pieces' poses at `t_to`, rotates their stored normals from `pg.meta.boundary_time` to `t_to` and sets
`boundary_time = t_to`. Exact rigid motion (no integration). A kernel; nothing happens on a domain without
moving pieces.
"""
function place_ghosts!(pg::ParticleGrid{D, M, T}, X, t_from, t_to) where {D, M, T}
    geom = pg.geometry
    has_moving_pieces(geom) || return nothing
    tm_pos = tag_motions(geom, t_from, t_to)
    tm_nrm = tag_motions(geom, pg.meta.boundary_time, t_to)
    foreach_particle(_place_ghost!, get_backend(), pg.meta.N,
                     (X, pg.core.normals, pg.core.is_boundary, pg.core.tags, tm_pos, tm_nrm))
    pg.meta.boundary_time = T(t_to)
    return nothing
end

# Kernel body (see Backend.jl): a ghost takes its piece's displacement; interior particles are left alone.
@inline function _place_ghost!(p, X, normals, isb, tags, tm_pos, tm_nrm)
    @inbounds begin
        isb[p] || return nothing
        k = tags[p] + 1
        X[p]       = move(tm_pos[k], X[p])
        normals[p] = rotate(tm_nrm[k], normals[p])
    end
    return nothing
end

"""
    sweep_boundaries!(pg, t_old, t_new, max_order; clearance = 0.25, min_gap = 0.5, samples_per_dx = 2)
        -> (; inserted, removed, mass_residual)

Makes the interior cloud follow the moving pieces of the domain from their poses at `t_old` to those at `t_new`
(their ghosts must already be at `t_new`: `place_ghosts!`, or the time stepper, which calls this after every
step). Interior particles that a piece now covers, or that lie closer than `clearance` spacings to it, are
removed; lattice points of the canvas that are at least `min_gap` spacings inside the region and at least
`min_gap` spacings from every remaining particle are inserted. Each change goes through the remap of the
adaptivity (`_remap_event!` with the region at both times, quadrature with `samples_per_dx` points per spacing),
used as a consistent interpolation, not as a conservation step: an inserted particle carries the limited linear
reconstruction of its neighbours, a covered particle vanishes with its content, the particles around a change
blend their state with the reconstruction on the volume they gain, and the particle volumes follow the region
(a covered point's share leaves with the piece, an uncovered one's arrives), so the interior volumes sum to the
region's volume at `t_new` up to the quadrature. The content of the region is therefore *not* conserved by the
sweep; `mass_residual` reports its change. That is the consistent treatment of a collocation scheme at a moving
wall: the transport of the swept fluid is the equations' business (the wall velocity enters through the
[`WallPoint`](@ref) of the conditions), and enforcing the old content on the new region instead flattened the
solution around every insertion. Nothing is renormalised to `domain_volume`, which is the volume at `t = 0`;
the volumes of particles the sweep does not touch keep the region of the last sweep that changed them
(`pg.meta.sweep_time`).

Requires valid neighbour lists. Returns the counts and the content change of the call; `(0, 0, 0)` on a domain
without moving pieces.
"""
function sweep_boundaries!(pg::ParticleGrid{D, M, T}, t_old, t_new, max_order::Int;
                           clearance::Real = 0.25, min_gap::Real = 0.5, samples_per_dx::Int = 2) where {D, M, T}
    nothing_done = (inserted = 0, removed = 0, mass_residual = zero(State{M, T}))
    geom = pg.geometry
    (has_moving_pieces(geom) && pg.meta.N > 0) || return nothing_done
    t = T(t_new)
    pg.meta.boundary_time == t ||
        throw(ArgumentError("sweep_boundaries!: the ghosts are at t = $(pg.meta.boundary_time), not at $t (call place_ghosts! first)"))
    t_old = pg.meta.sweep_time                          # the region the volumes near the pieces belong to
    mass0 = interior_mass(pg)

    on_host = parent(pg.rhos) isa Array
    hg = on_host ? pg : device_grid(Array, pg)
    N  = hg.meta.N
    shape_t = at_time(geom.shape, t)
    dx  = minimum(hg.meta.dx)
    R   = hg.meta.R
    pos = hg.core.positions
    isb = hg.core.is_boundary
    cd  = hg.domain
    clear, gap = T(clearance) * dx, T(min_gap) * dx

    # The region the pieces can have changed: their boxes at `t_old` and `t`, widened by the band and a spacing
    boxes = map(geom.pieces) do piece
        lo0, hi0 = bounding_box(at_time(piece, t_old))      # unbounded pieces (a piston's half space): the canvas
        lo1, hi1 = bounding_box(at_time(piece, t))
        (max.(min.(lo0, lo1) .- (R + 2dx), cd.canvas_mins), min.(max.(hi0, hi1) .+ (R + 2dx), cd.canvas_maxs))
    end
    inbox(x) = any(b -> all(b[1] .<= x .<= b[2]), boxes)

    # 1. Covered: interior particles inside a piece or closer than `clearance` to it
    remove_mask = hg.shared.bit_buffer
    fill!(view(remove_mask, 1:N), false)
    removed = Int[]
    for i in 1:N
        (isb[i] || !inbox(pos[i])) && continue
        if signed_distance(shape_t, pos[i]) > -clear
            remove_mask[i] = true
            push!(removed, i)
        end
    end

    # 2. Uncovered: lattice points of the canvas inside the boxes, at least `min_gap` inside the region and
    #    away from every particle that stays
    added = Space{D, T}[]
    idx, dvec = Int[], Space{D, T}[]
    for b in boxes
        lo, hi = b
        rng = ntuple(d -> ceil(Int, (lo[d] - cd.canvas_mins[d]) / hg.meta.dx[d]):floor(Int, (hi[d] - cd.canvas_mins[d]) / hg.meta.dx[d]), Val(D))
        for I in CartesianIndices(rng)
            x = _wrap_point(hg, Space{D, T}(ntuple(d -> cd.canvas_mins[d] + I[d] * hg.meta.dx[d], Val(D))))
            all(cd.canvas_mins .<= x .<= cd.canvas_maxs) || continue
            signed_distance(shape_t, x) <= -gap || continue
            n = _collect_near!(hg, x, gap^2, idx, dvec)
            any(k -> !remove_mask[idx[k]], 1:n) && continue
            push!(added, x)
        end
    end
    (isempty(removed) && isempty(added)) && return nothing_done

    # 3. Remaps (sequential: later events see the states the earlier ones left; no content goes to a particle
    #    that is removed in this call), then the topology change and the neighbour search
    N_new = N - length(removed) + length(added)
    if N_new > N
        ensure_capacity!(hg, N_new)
        on_host || ensure_capacity!(pg, N_new)
    end
    events = [_AdaptEvent{D, T}(pos[i], [i], Space{D, T}[]) for i in removed]
    append!(events, [_AdaptEvent{D, T}(x, Int[], [x]) for x in added])
    total_nb, max_nb = _slots_after_events(hg, events)
    ensure_capacity!(pg.neighbor, _slot_capacity(eltype(pg.neighbor.ranges), total_nb, max_nb, N_new))
    ad = ParticleAdaptivity(T; samples_per_dx = samples_per_dx)
    new_pos, new_rho, new_vol = Space{D, T}[], State{M, T}[], T[]
    df = Vector{State{M, T}}(undef, length(hg.neighbor.indices))
    region = (t_old, t)
    for ev in events
        _remap_event!(hg, ad, ev, new_pos, new_rho, new_vol, df; removed_mask = remove_mask, region = region)
    end
    _apply_topology!(hg, remove_mask, new_pos, new_rho, new_vol, max_order)
    on_host || _upload_particles!(pg, hg)
    pg.neighbor(pg)
    pg.meta.sweep_time = t
    return (inserted = length(new_pos), removed = length(removed), mass_residual = interior_mass(pg) - mass0)
end
