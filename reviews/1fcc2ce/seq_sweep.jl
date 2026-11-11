# Probe-only runtime override (not a repo edit): sweep_boundaries! with one event per remap and a neighbour search
# between events (the independence the adaptivity's event spacing guarantees), insertions checked against each other.
@eval HyperCloudCore function sweep_boundaries!(pg::ParticleGrid{D, M, T}, t_old, t_new, max_order::Int;
                           clearance::Real = 0.25, min_gap::Real = 0.5, samples_per_dx::Int = 2) where {D, M, T}
    geom = pg.geometry
    (has_moving_pieces(geom) && pg.meta.N > 0) || return (inserted = 0, removed = 0, mass_residual = zero(State{M, T}))
    t = T(t_new)
    t_old = pg.meta.sweep_time
    mass0 = interior_mass(pg)
    hg = pg
    dx = minimum(hg.meta.dx); R = hg.meta.R; cd = hg.domain
    clear, gap = T(clearance) * dx, T(min_gap) * dx
    shape_t = at_time(geom.shape, t)
    boxes = map(geom.pieces) do piece
        lo0, hi0 = bounding_box(at_time(piece, t_old)); lo1, hi1 = bounding_box(at_time(piece, t))
        (max.(min.(lo0, lo1) .- (R + 2dx), cd.canvas_mins), min.(max.(hi0, hi1) .+ (R + 2dx), cd.canvas_maxs))
    end
    inbox(x) = any(b -> all(b[1] .<= x .<= b[2]), boxes)
    _update_region!(hg, geom, t_old, t, boxes, samples_per_dx)
    ad = ParticleAdaptivity(T; samples_per_dx = samples_per_dx)
    nrem = 0; nins = 0
    function do_event!(ev)
        N = hg.meta.N
        mask = hg.shared.bit_buffer; fill!(view(mask, 1:N), false)
        for r in ev.removed; mask[r] = true; end
        ensure_capacity!(hg, N + length(ev.added))
        total_nb, max_nb = _slots_after_events(hg, [ev])
        ensure_capacity!(hg.neighbor, _slot_capacity(eltype(hg.neighbor.ranges), total_nb, max_nb, N + length(ev.added)))
        np, nr, nv = Space{D, T}[], State{M, T}[], T[]
        df = Vector{State{M, T}}(undef, length(hg.neighbor.indices))
        _remap_event!(hg, ad, ev, np, nr, nv, df; removed_mask = mask)
        _apply_topology!(hg, mask, np, nr, nv, max_order)
        hg.neighbor(hg)
    end
    while true
        pos = hg.core.positions; isb = hg.core.is_boundary
        i = findfirst(i -> !isb[i] && inbox(pos[i]) && (p = _probe(shape_t, pos[i]); p[4] !== nothing && p[1] > -clear), 1:hg.meta.N)
        i === nothing && break
        do_event!(_AdaptEvent{D, T}(pos[i], [i], Space{D, T}[])); nrem += 1
    end
    idx, dvec = Int[], Space{D, T}[]
    while true
        found = nothing
        for b in boxes
            lo, hi = b
            rng = ntuple(d -> ceil(Int, (lo[d] - cd.canvas_mins[d]) / hg.meta.dx[d]):floor(Int, (hi[d] - cd.canvas_mins[d]) / hg.meta.dx[d]), Val(D))
            for I in CartesianIndices(rng)
                x = _wrap_point(hg, Space{D, T}(ntuple(d -> cd.canvas_mins[d] + I[d] * hg.meta.dx[d], Val(D))))
                all(cd.canvas_mins .<= x .<= cd.canvas_maxs) || continue
                signed_distance(shape_t, x) <= -gap || continue
                _collect_near!(hg, x, gap^2, idx, dvec) == 0 || continue
                found = x; break
            end
            found === nothing || break
        end
        found === nothing && break
        do_event!(_AdaptEvent{D, T}(found, Int[], [found])); nins += 1
    end
    pg.meta.sweep_time = t
    return (inserted = nins, removed = nrem, mass_residual = interior_mass(pg) - mass0)
end
