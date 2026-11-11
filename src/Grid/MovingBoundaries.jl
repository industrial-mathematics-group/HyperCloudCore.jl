export place_ghosts!, sweep_boundaries!

const _WARNED_SWEEP = Ref(false)

# =========================================================================
# MOVING BOUNDARIES — ghost particles that follow their piece, interior particles that make way
# =========================================================================
#
# A `Moving` piece of the domain's shape is a rigid body with a prescribed motion. The cloud stays Eulerian:
#   * its ghost particles are placed at the piece's pose at every stage time (`place_ghosts!`, exact rigid
#     motion of the positions and normals from the step start: no drift);
#   * after each step the region's volumes follow the pieces (what they covered leaves the interior particles,
#     what they uncovered joins them with the reconstructed state), then the interior particles a piece has
#     covered, or come closer to than a quarter spacing, are removed and the region it has left is refilled with
#     the lattice points of the canvas that are at least half a spacing inside and half a spacing from every
#     particle, through the conservative remap of the adaptivity in rounds of independent events; the static
#     walls' ghosts are added where a piece uncovered their band (`sweep_boundaries!`).
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
    # The lattice points that changed side. Each once: the boxes overlap (a set over several boxes), and on a
    # periodic axis the cell index runs over one period only (index 0 and n are one point after wrapping; a box
    # spanning the axis counted the seam twice and grew the region by 1/n of the swept volume).
    n_per   = ntuple(d -> cd.is_periodic[d] ? max(1, round(Int, cd.L[d] / hq)) : 0, Val(D))
    visited = length(boxes) > 1 ? Set{NTuple{D, Int}}() : nothing
    changed = Tuple{Space{D, T}, Bool}[]
    for b in boxes
        lo_b, hi_b = b
        rng = ntuple(Val(D)) do d
            a, z = ceil(Int, (lo_b[d] - cd.canvas_mins[d]) / hq), floor(Int, (hi_b[d] - cd.canvas_mins[d]) / hq)
            n_per[d] > 0 ? (max(a, 1):min(z, n_per[d])) : (a:z)
        end
        for I in CartesianIndices(rng)
            if visited !== nothing
                key = Tuple(I)
                key in visited && continue
                push!(visited, key)
            end
            y = _wrap_point(hg, Space{D, T}(ntuple(d -> cd.canvas_mins[d] + (I[d] - T(0.5)) * hq, Val(D))))
            in_new = _is_interior(geom, y, t)
            _is_interior(geom, y, t_old) == in_new || push!(changed, (y, in_new))
        end
    end
    isempty(changed) && return nothing
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
    for (y, in_new) in changed
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
    for (l, j) in enumerate(near)
        Vn = V[j] + δV[l]
        if Vn > T(1e-12) * V[j]
            U[j] = math_max.(lo[l], math_min.(hi[l], (V[j] * U[j] + δm[l]) / Vn))
            V[j] = Vn
        else
            V[j] = zero(T)          # its whole share is covered: it is removed below and hands out nothing (keeping
        end                         # the old volume put region that no longer exists back onto its neighbours)
    end
    fill!(view(local_id, 1:N), 0)
    return nothing
end

"""
    sweep_boundaries!(pg, t_old, t_new, max_order; clearance = 0.25, min_gap = 0.5, samples_per_dx = 2)
        -> (; inserted, removed, mass_residual)

Makes the interior cloud follow the moving pieces of the domain from their poses at the last sweep
(`pg.meta.sweep_time`; the argument `t_old` is not used) to those at `t_new` (their ghosts must already be at
`t_new`: `place_ghosts!`, or the time stepper, which calls this after every step). Interior particles that a piece
now covers, or that lie closer than `clearance` spacings to it, are removed; lattice points of the canvas that are
inside the region, at least `min_gap` spacings from every moving piece and from every remaining particle are
inserted (the lattice points of a static wall lie on the wall, as in the initial cloud).

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
content go to its neighbours, an inserted particle takes its share from them. The removals, then the insertions,
run in rounds of independent events (centres at least the adaptivity's event spacing apart, `_event_spacing`),
each round followed by the topology change and a neighbour search, so that every remap sees the particles the
earlier ones removed and inserted. Last, the ghosts of the static walls are added where a piece uncovered their
band (lattice points outside a static wall, within the band of the interior, state of the nearest ghost of the
same tag). `mass_residual` is the change of `interior_mass` over the call: the content of what was covered and
uncovered. Nothing is renormalised to `domain_volume`, which is the volume at `t = 0`. A piece whose box reaches
across a periodic end is refused (pieces have no periodic images).

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
    _check_periodic_pieces(hg, geom, t)
    posed = map(pc -> at_time(pc, t), geom.pieces)

    # 1. The region itself: what the pieces covered leaves the interior particles, what they uncovered joins them
    _update_region!(hg, geom, t_old, t, boxes, samples_per_dx)

    # 2. Covered: interior particles inside a piece or closer than `clearance` to it, removed in rounds of
    #    independent remaps (below). Particles whose whole share the piece covered have no volume and no content
    #    left: they go first, without a remap (as survivors of other events they made the remap divide 0 by 0)
    ad      = ParticleAdaptivity(T; samples_per_dx = samples_per_dx)
    spacing = _event_spacing(hg)
    df      = State{M, T}[]                              # neighbour-slot scratch of the remaps, shared by the rounds
    n_removed, n_inside = _drop_empty!(hg, max_order), -1
    n_removed > 0 && (n_inside = n_removed)
    while true
        cand = _covered_particles(hg, geom, posed, t, inbox, clear)
        n_inside < 0 && (n_inside = count(i -> !_is_interior(geom, hg.core.positions[i], t), cand))
        isempty(cand) && break
        pos = hg.core.positions
        sel = _spaced(hg, [pos[i] for i in cand], spacing)
        _sweep_round!(hg, ad, [_AdaptEvent{D, T}(pos[cand[a]], [cand[a]], Space{D, T}[]) for a in sel], max_order, df)
        n_removed += length(sel)
    end
    # A piece that moves more than the clearance in one step covers particles before they are removed: their last
    # stage was computed inside the solid. Keep |w| dt below `clearance` spacings.
    n_inside > 0 && !_WARNED_SWEEP[] && (_WARNED_SWEEP[] = true;
        @warn "sweep_boundaries!: $n_inside interior particle(s) were inside a moving piece at t = $t before being removed; " *
              "the piece moved more than $clearance spacings in one step. Reduce the time step (|wall velocity| dt <= $clearance dx). " *
              "Reported once.")

    # 3. Uncovered: lattice points of the canvas inside the boxes and inside the region, at least `min_gap` from
    #    every moving piece and every particle, inserted in rounds as well (each round sees the particles the
    #    earlier ones inserted, so a lattice point reached from both ends of a periodic axis is inserted once).
    #    Also at least `min_gap` inside a static wall: inserted wall-row particles (on the wall, as `get_points`
    #    makes them) got the remap's share of a half cell (0.6 h² against 0.97 h² in a generated cloud) and kept
    #    an error of 0.07 behind a receding piston, the uncovered part being less accurate than without them.
    n_inserted = 0
    while true
        cand = _uncovered_points(hg, geom, shape_t, posed, boxes, gap, R + 2dx)
        isempty(cand) && break
        sel = _spaced(hg, cand, spacing)
        _sweep_round!(hg, ad, [_AdaptEvent{D, T}(cand[a], Int[], [cand[a]]) for a in sel], max_order, df)
        n_inserted += length(sel)
    end

    # 4. The ghosts of the static walls next to what the pieces uncovered (the initial band only covers the
    #    region at t = 0)
    n_ghosts = _add_static_ghosts!(hg, geom, shape_t, t, boxes, gap, max_order)
    n_ghosts > 0 && _refresh_static_ghosts!(hg, geom, t, inbox)

    if n_removed + n_inserted + n_ghosts == 0
        on_host || copyto!(parent(pg.core.volumes), 1, parent(hg.core.volumes), 1, N)
        on_host || copyto!(parent(pg.rhos), 1, parent(hg.rhos), 1, N)
        pg.meta.sweep_time = t
        return (inserted = 0, removed = 0, mass_residual = interior_mass(pg) - mass0)
    end
    if !on_host
        total_nb, max_nb = _slots_after_events(hg, _AdaptEvent{D, T}[])     # the host copy is searched already
        ensure_capacity!(pg.neighbor, _slot_capacity(eltype(pg.neighbor.ranges), total_nb, max_nb, hg.meta.N))
        _upload_particles!(pg, hg)
        pg.neighbor(pg)
    end
    pg.meta.sweep_time = t
    return (inserted = n_inserted, removed = n_removed, mass_residual = interior_mass(pg) - mass0)
end

# The pieces are evaluated without periodic images (the cell lists wrap, the shapes do not): a piece whose box
# reaches across a periodic end would lose its wrapped part (the region grew to the whole channel while the
# piece's ghosts stayed, and interior particles overlapped them). Refused instead of a silently wrong region.
# The box (conservative for rotated bodies) is only a first test; the piece is refused if it has points on a periodic
# end, sampled at half a spacing over the box (a ball spinning about its centre near the seam has a box that reaches
# r√2 from the centre and was refused although it never crosses).
function _check_periodic_pieces(hg::ParticleGrid, geom::GeometricDomain{GEO, D, T}, t) where {GEO, D, T}
    hq = minimum(hg.meta.dx) / 2
    for (k, piece) in enumerate(geom.pieces), d in 1:D
        geom.is_periodic[d] || continue
        posed  = at_time(piece, t)
        lo, hi = bounding_box(posed)
        (isfinite(lo[d]) && isfinite(hi[d])) || continue
        (lo[d] < geom.mins[d] || hi[d] > geom.maxs[d]) || continue
        any(i -> (lo[i] > hg.domain.canvas_maxs[i] || hi[i] < hg.domain.canvas_mins[i]), 1:D) && continue
        for e in (geom.mins[d], geom.maxs[d])
            (lo[d] <= e <= hi[d]) || continue
            rng = ntuple(i -> i == d ? (0:0) : (0:max(0, ceil(Int, (min(hi[i], hg.domain.canvas_maxs[i]) - max(lo[i], hg.domain.canvas_mins[i])) / hq))), Val(D))
            for I in CartesianIndices(rng)
                y = Space{D, T}(ntuple(i -> i == d ? e : max(lo[i], hg.domain.canvas_mins[i]) + I[i] * hq, Val(D)))
                signed_distance(posed, y) < zero(T) &&
                    throw(ArgumentError("sweep_boundaries!: moving piece $k reaches across the periodic ends of axis $d at t = $t " *
                                        "(at $y); moving pieces must stay inside the periodic period (no periodic images of pieces)"))
            end
        end
    end
    return nothing
end

# Interior particles whose volume the region pass took entirely (covered by a piece): removed without a remap
function _drop_empty!(hg::ParticleGrid{D, M, T}, max_order::Int) where {D, M, T}
    N, isb, V = hg.meta.N, hg.core.is_boundary, hg.core.volumes
    mask = hg.shared.bit_buffer
    n = 0
    @inbounds for i in 1:N
        mask[i] = !isb[i] && iszero(V[i])
        n += mask[i]
    end
    n == 0 && return 0
    _apply_topology!(hg, mask, Space{D, T}[], State{M, T}[], T[], max_order)
    hg.neighbor(hg)
    return n
end

# Interior particles in the boxes that a moving piece covers or comes closer to than `clear`, by the pieces' own
# distances (the composite's nearest surface left a particle on a static wall next to an advancing face until the
# face covered it: the static wall won the tie at distance 0)
function _covered_particles(hg::ParticleGrid{D, M, T}, geom, posed, t::T, inbox, clear::T) where {D, M, T}
    pos, isb = hg.core.positions, hg.core.is_boundary
    out = Int[]
    for i in 1:hg.meta.N
        (isb[i] || !inbox(pos[i])) && continue
        x = pos[i]
        (!_is_interior(geom, x, t) || any(p -> abs(signed_distance(p, x)) < clear, posed)) && push!(out, i)
    end
    return out
end

# The static ghosts in the boxes take the wall geometry at `t` (those near where a piece was kept its normals and
# distances of t = 0 after it left)
function _refresh_static_ghosts!(hg::ParticleGrid, geom, t, inbox)
    pot = geom.piece_of_tag
    for i in 1:hg.meta.N
        hg.core.is_boundary[i] && inbox(hg.core.positions[i]) || continue
        tag = hg.core.tags[i]
        (tag + 1 <= length(pot) && pot[tag + 1] != 0) && continue
        hg.core.normals[i], hg.core.wall_distances[i] = wall_geometry(geom, hg.core.positions[i], t)
    end
    return nothing
end

# The canvas lattice (spacing dx; a non-periodic wall's lattice point on the wall, as in `get_points`) inside `box`
function _sweep_lattice(f, hg::ParticleGrid{D, M, T}, geom, box) where {D, M, T}
    cd, dx = hg.domain, hg.meta.dx
    wall_lo = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.mins[d], cd.canvas_mins[d], dx[d]), Val(D))
    wall_hi = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.maxs[d], cd.canvas_mins[d], dx[d]), Val(D))
    lo, hi = box
    tol = T(1e-6)                     # (canvas_max - canvas_min) / dx = 25.999999999999996 dropped the last row
    rng = ntuple(d -> ceil(Int, (lo[d] - cd.canvas_mins[d]) / dx[d] - tol):floor(Int, (hi[d] - cd.canvas_mins[d]) / dx[d] + tol), Val(D))
    for I in CartesianIndices(rng)
        x = Space{D, T}(ntuple(Val(D)) do d
            j = I[d]
            j == wall_lo[d] ? geom.mins[d] : j == wall_hi[d] ? geom.maxs[d] : cd.canvas_mins[d] + j * dx[d]
        end)
        x = _wrap_point(hg, x)
        all(cd.canvas_mins .<= x .<= cd.canvas_maxs) && f(x)
    end
    return nothing
end

# Lattice points in the boxes that are at least `gap` inside the region and from every particle, within `reach` of
# a moving piece (what a piece can have uncovered; holes of a jittered cloud elsewhere in a piece's box, a whole
# canvas for a piston, are the adaptivity's business)
function _uncovered_points(hg::ParticleGrid{D, M, T}, geom, shape_t, posed, boxes, gap::T, reach::T) where {D, M, T}
    out = Space{D, T}[]
    idx, dvec = Int[], Space{D, T}[]
    for b in boxes
        _sweep_lattice(hg, geom, b) do x
            signed_distance(shape_t, x) <= -gap || return
            any(p -> abs(signed_distance(p, x)) <= reach, posed) || return
            _collect_near!(hg, x, gap^2, idx, dvec) == 0 || return
            push!(out, x)
        end
    end
    return out
end

# A subset of `centres` (by index) no two of which are closer than `spacing` (periodic axes wrap): remaps with
# such centres are independent (`_event_spacing`)
function _spaced(hg::ParticleGrid{D, M, T}, centres::Vector{Space{D, T}}, spacing::T) where {D, M, T}
    L, Li = hg.domain.L_wrap, hg.domain.invL_wrap
    s2 = spacing^2
    sel = Int[]
    for a in eachindex(centres)
        any(b -> sum(abs2, _wrapped_diff(centres[b], centres[a], L, Li)) < s2, sel) || push!(sel, a)
    end
    return sel
end

# One round of independent events: the conservative remaps on the current topology, the topology change and the
# neighbour search, so that the next round sees the particles this one removed and inserted. (All events of a
# sweep at once, on the old topology, gave each remap the shares of every particle removed in its box: in 2-D a
# row of removals changed states by 0.2 and volumes by a factor up to 2.8 as far as 12 spacings from the face.)
function _sweep_round!(hg::ParticleGrid{D, M, T}, ad, events::Vector{_AdaptEvent{D, T}}, max_order::Int,
                       df::Vector{State{M, T}}) where {D, M, T}
    N     = hg.meta.N
    N_new = N + sum(ev -> length(ev.added) - length(ev.removed), events)
    N_new > N && ensure_capacity!(hg, N_new)
    total_nb, max_nb = _slots_after_events(hg, events)
    ensure_capacity!(hg.neighbor, _slot_capacity(eltype(hg.neighbor.ranges), total_nb, max_nb, N_new))
    mask = hg.shared.bit_buffer
    fill!(view(mask, 1:N), false)
    for ev in events, r in ev.removed
        mask[r] = true
    end
    new_pos, new_rho, new_vol = Space{D, T}[], State{M, T}[], T[]
    length(df) < length(hg.neighbor.indices) && resize!(df, length(hg.neighbor.indices))
    for ev in events
        _remap_event!(hg, ad, ev, new_pos, new_rho, new_vol, df; removed_mask = mask)
    end
    _apply_topology!(hg, mask, new_pos, new_rho, new_vol, max_order)
    hg.neighbor(hg)
    return nothing
end

# Static-wall ghosts where the pieces uncovered the band: lattice points outside the region whose nearest surface
# is a static wall, at least half a spacing outside it (nearer points are snapped onto the wall in `get_points`),
# within `R + dx` of an interior particle (the band of `get_points`) and at least `gap` from every particle. Their
# state is that of the nearest ghost of the same tag (a constant Dirichlet band stays constant), their volume zero.
# Returns the number added; the neighbour lists are searched again if any.
function _add_static_ghosts!(hg::ParticleGrid{D, M, T}, geom, shape_t, t::T, boxes, gap::T, max_order::Int) where {D, M, T}
    isb, tags = hg.core.is_boundary, hg.core.tags
    cutoff_sq = (hg.meta.R + maximum(hg.meta.dx))^2
    h_snap = minimum(hg.meta.dx) / 2
    L, Li = hg.domain.L_wrap, hg.domain.invL_wrap
    idx, dvec = Int[], Space{D, T}[]
    xs, tg, us = Space{D, T}[], Int[], State{M, T}[]
    for b in boxes
        _sweep_lattice(hg, geom, b) do x
            _is_interior(geom, x, t) && return
            φ, _, tag, pc = _probe(shape_t, x)
            (pc === nothing && tag != 0 && φ >= h_snap) || return
            _collect_near!(hg, x, gap^2, idx, dvec) == 0 || return
            any(y -> sum(abs2, _wrapped_diff(y, x, L, Li)) < gap^2, xs) && return
            n = _collect_near!(hg, x, cutoff_sq, idx, dvec)
            any(k -> !isb[idx[k]], 1:n) || return
            jbest, dbest, jany, dany = 0, T(Inf), 0, T(Inf)
            for k in 1:n
                d2 = sum(abs2, dvec[k])
                d2 < dany && (jany = idx[k]; dany = d2)
                (isb[idx[k]] && tags[idx[k]] == tag && d2 < dbest) && (jbest = idx[k]; dbest = d2)
            end
            push!(xs, x); push!(tg, tag); push!(us, hg.rhos[jbest != 0 ? jbest : jany])
        end
    end
    isempty(xs) && return 0
    N, n = hg.meta.N, length(xs)
    total_nb, max_nb = _slots_after_events(hg, [_AdaptEvent{D, T}(x, Int[], [x]) for x in xs])
    ensure_capacity!(hg, N + n)
    ensure_particle_capacity!(hg.neighbor, N + n)
    ensure_capacity!(hg.neighbor, _slot_capacity(eltype(hg.neighbor.ranges), total_nb, max_nb, N + n))
    core = hg.core
    for a in 1:n
        p = N + a
        core.positions[p]       = xs[a]
        core.is_boundary[p]     = true
        core.tags[p]            = tg[a]
        core.particle_orders[p] = max_order
        core.volumes[p]         = zero(T)
        core.normals[p], core.wall_distances[p] = wall_geometry(geom, xs[a], t)
        hg.rhos[p]              = us[a]
        hg.curvatures[p]        = zero(State{M, T})
    end
    hg.meta.N       = N + n
    hg.meta.N_ghost = hg.meta.N - hg.meta.N_interior
    hg.neighbor(hg)
    return n
end
