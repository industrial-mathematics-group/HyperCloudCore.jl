export place_ghosts!, sweep_boundaries!

const _WARNED_SWEEP = Ref(false)

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

# The region changed between `t_old` and `t`: the lattice points (spacing dx / samples_per_dx) a piece has covered
# leave the interior particles that shared them, volume and content alike (the content there is the limited linear
# reconstruction of the nearest interior particle), and the points it has uncovered join them with the reconstructed
# state. Once per sweep over the union of the boxes, before the events, so that no point is counted twice (the
# remaps of neighbouring events overlap). The partition is that of the interior particles (ghosts excluded, as in
# the volume estimators). A particle whose whole share is gone keeps its state; it is removed right after.
function _update_region!(hg::ParticleGrid{D, M, T}, geom, t_old::T, t::T, boxes, samples_per_dx::Int) where {D, M, T}
    N    = hg.meta.N
    nb   = hg.neighbor
    pos  = hg.core.positions
    isb  = hg.core.is_boundary
    V    = hg.core.volumes
    U    = hg.rhos
    wf   = nb.weight_func
    R_sq = hg.meta.R^2
    dx   = minimum(hg.meta.dx)
    hq   = dx / samples_per_dx
    ω    = hq^D
    cd   = hg.domain
    inbox(x) = any(b -> all(b[1] .<= x .<= b[2]), boxes)
    near = [i for i in 1:N if !isb[i] && inbox(pos[i])]
    isempty(near) && return nothing
    local_id = hg.shared.int_buffer
    fill!(view(local_id, 1:N), 0)
    for (l, i) in enumerate(near); local_id[i] = l; end
    # Limited linear reconstruction of the interior particles in the boxes (as the remap does for its local set)
    n_near = length(near)
    grads  = Vector{SVector{D, State{M, T}}}(undef, n_near)
    lo     = Vector{State{M, T}}(undef, n_near)
    hi     = Vector{State{M, T}}(undef, n_near)
    df     = Vector{State{M, T}}(undef, length(nb.indices))
    for (l, j) in enumerate(near)
        sl = nb.ranges[j]
        mn, mx = U[j], U[j]
        for k in sl
            uk = U[nb.indices[k]]
            df[k] = uk - U[j]
            mn = math_min.(mn, uk); mx = math_max.(mx, uk)
        end
        lo[l], hi[l] = mn, mx
        g = isempty(sl) ? zero(SVector{D, State{M, T}}) : Interpolator{D, 1, 1}()(sl, nb.distances, nb.weights, df; scale = dx)
        grads[l] = SVector{D, State{M, T}}(ntuple(d -> g[d], Val(D)))
    end
    δV = zeros(T, n_near)
    δm = zeros(State{M, T}, n_near)
    idx, dvec = Int[], Space{D, T}[]
    visited = length(boxes) > 1 ? Set{NTuple{D, Int}}() : nothing
    for b in boxes
        lo_b, hi_b = b
        rng = ntuple(d -> ceil(Int, (lo_b[d] - cd.canvas_mins[d]) / hq):floor(Int, (hi_b[d] - cd.canvas_mins[d]) / hq), Val(D))
        for I in CartesianIndices(rng)
            if visited !== nothing
                key = Tuple(I)
                key in visited && continue
                push!(visited, key)
            end
            y = _wrap_point(hg, Space{D, T}(ntuple(d -> cd.canvas_mins[d] + (I[d] - T(0.5)) * hq, Val(D))))
            in_old = _is_interior(geom, y, t_old)
            in_new = _is_interior(geom, y, t)
            in_old == in_new && continue
            n = _collect_near!(hg, y, R_sq, idx, dvec)
            S = zero(T)
            jstar, dstar = 0, T(Inf)
            @inbounds for k in 1:n
                l = local_id[idx[k]]
                l == 0 && continue
                d2 = sum(abs2, dvec[k])
                S += wf(d2)
                d2 < dstar && (jstar = k; dstar = d2)
            end
            (S > 0 && jstar != 0) || continue
            lstar = local_id[idx[jstar]]
            u_rec = U[near[lstar]]
            @inbounds for d in 1:D
                u_rec -= dvec[jstar][d] * grads[lstar][d]          # dvec = x_j − y
            end
            u_rec = math_max.(lo[lstar], math_min.(hi[lstar], u_rec))
            sgn = in_new ? one(T) : -one(T)
            @inbounds for k in 1:n
                l = local_id[idx[k]]
                l == 0 && continue
                dφ = sgn * ω * wf(sum(abs2, dvec[k])) / S
                δV[l] += dφ
                δm[l] += dφ * u_rec
            end
        end
    end
    for (l, j) in enumerate(near)
        Vn = V[j] + δV[l]
        Vn > T(1e-12) * V[j] || continue
        U[j] = math_max.(lo[l], math_min.(hi[l], (V[j] * U[j] + δm[l]) / Vn))
        V[j] = Vn
    end
    fill!(view(local_id, 1:N), 0)
    return nothing
end

"""
    sweep_boundaries!(pg, t_old, t_new, max_order; clearance = 0.25, min_gap = 0.5, samples_per_dx = 2)
        -> (; inserted, removed, mass_residual)

Makes the interior cloud follow the moving pieces of the domain from their poses at `t_old` to those at `t_new`
(their ghosts must already be at `t_new`: `place_ghosts!`, or the time stepper, which calls this after every
step). Interior particles that a piece now covers, or that lie closer than `clearance` spacings to it, are
removed; lattice points of the canvas that are at least `min_gap` spacings inside the region and at least
`min_gap` spacings from every remaining particle are inserted.

Two steps. First the **region**: on a lattice with `samples_per_dx` points per spacing, the points a piece has
covered since the last sweep (`pg.meta.sweep_time`) leave the interior particles that shared them, volume and
content alike (the content there is the limited linear reconstruction of the nearest interior particle), and the
points it has uncovered join them with the reconstructed state. The particle volumes therefore follow the region
(the interior volumes sum to its volume up to the quadrature), the content of the covered part is gone and the
uncovered part appears with the local state: a consistent interpolation, as a collocation scheme needs at a moving
wall. The transport of the swept fluid is the equations' business, through the wall velocity in the
[`WallPoint`](@ref) of the conditions; enforcing the old content on the new region instead flattened the
solution around every insertion. Then the **topology**: each removal and insertion is the conservative remap of
the adaptivity on the new region (`_remap_event!`): a removed particle's remaining share of the region and its
content go to its neighbours, an inserted particle takes its share from them. `mass_residual` is the change of
`interior_mass` over the call: the content of what was covered and uncovered. Nothing is renormalised to
`domain_volume`, which is the volume at `t = 0`.

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

    # 1. The region itself: what the pieces covered leaves the interior particles, what they uncovered joins them
    _update_region!(hg, geom, t_old, t, boxes, samples_per_dx)

    # 2. Covered: interior particles inside a piece or closer than `clearance` to it
    remove_mask = hg.shared.bit_buffer
    fill!(view(remove_mask, 1:N), false)
    removed = Int[]
    n_inside = 0
    for i in 1:N
        (isb[i] || !inbox(pos[i])) && continue
        φ, _, _, piece = _probe(shape_t, pos[i])
        (piece !== nothing && φ > -clear) || continue        # the nearest piece must be a moving one: particles on
        remove_mask[i] = true                               # a static wall inside the box stay
        push!(removed, i)
        φ > zero(T) && (n_inside += 1)
    end
    # A piece that moves more than the clearance in one step covers particles before they are removed: their last
    # stage was computed inside the solid. Keep |w| dt below `clearance` spacings.
    n_inside > 0 && !_WARNED_SWEEP[] && (_WARNED_SWEEP[] = true;
        @warn "sweep_boundaries!: $n_inside interior particle(s) were inside a moving piece at t = $t before being removed; " *
              "the piece moved more than $clearance spacings in one step. Reduce the time step (|wall velocity| dt <= $clearance dx). " *
              "Reported once.")

    # 3. Uncovered: lattice points of the canvas inside the boxes, at least `min_gap` inside the region and
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
    if isempty(removed) && isempty(added)
        on_host || copyto!(parent(pg.core.volumes), 1, parent(hg.core.volumes), 1, N)
        on_host || copyto!(parent(pg.rhos), 1, parent(hg.rhos), 1, N)
        pg.meta.sweep_time = t
        return (inserted = 0, removed = 0, mass_residual = interior_mass(pg) - mass0)
    end

    # 4. Remaps on the new region (the conservative remap of the adaptivity: a removed particle's share of the
    #    region and its content go to its neighbours, an inserted one takes its share from them), sequential so
    #    that later events see the states the earlier ones left and no content goes to a particle removed in this
    #    call; then the topology change and the neighbour search
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
    for ev in events
        _remap_event!(hg, ad, ev, new_pos, new_rho, new_vol, df; removed_mask = remove_mask)
    end
    _apply_topology!(hg, remove_mask, new_pos, new_rho, new_vol, max_order)
    on_host || _upload_particles!(pg, hg)
    pg.neighbor(pg)
    pg.meta.sweep_time = t
    return (inserted = length(new_pos), removed = length(removed), mass_residual = interior_mass(pg) - mass0)
end
