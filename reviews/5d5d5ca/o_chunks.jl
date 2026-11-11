# Probe-only override (VARIANT=chunks): insertions as joint events (all candidates within R of a seed in one remap, so
# that they share the quadrature's S_new), the joint events in rounds of independent remaps.
if get(ENV, "VARIANT", "") == "chunks"
    @eval H function _round_body!(hg::ParticleGrid{D, M, T}, ad, events::Vector{_AdaptEvent{D, T}}, max_order::Int,
                           df::Vector{State{M, T}}) where {D, M, T}
        N     = hg.meta.N
        N_new = N + sum(ev -> length(ev.added) - length(ev.removed), events)
        N_new > N && ensure_capacity!(hg, N_new)
        total_nb, max_nb = _slots_after_events(hg, events)
        ensure_capacity!(hg.neighbor, _slot_capacity(eltype(hg.neighbor.ranges), total_nb, max_nb, N_new))
        mask = hg.shared.bit_buffer
        fill!(view(mask, 1:N), false)
        for ev in events, r in ev.removed; mask[r] = true; end
        new_pos, new_rho, new_vol = Space{D, T}[], State{M, T}[], T[]
        length(df) < length(hg.neighbor.indices) && resize!(df, length(hg.neighbor.indices))
        for ev in events; _remap_event!(hg, ad, ev, new_pos, new_rho, new_vol, df; removed_mask = mask); end
        _apply_topology!(hg, mask, new_pos, new_rho, new_vol, max_order)
        hg.neighbor(hg)
        return nothing
    end
    @eval H function _sweep_round!(hg::ParticleGrid{D, M, T}, ad, events::Vector{_AdaptEvent{D, T}}, max_order::Int,
                           df::Vector{State{M, T}}) where {D, M, T}
        if all(ev -> isempty(ev.removed), events)
            pts = [x for ev in events for x in ev.added]
            L, Li = hg.domain.L_wrap, hg.domain.invL_wrap
            R = hg.meta.R
            joint = _AdaptEvent{D, T}[]
            left = trues(length(pts))
            for a in eachindex(pts)
                left[a] || continue
                grp = [b for b in eachindex(pts) if left[b] && sum(abs2, _wrapped_diff(pts[b], pts[a], L, Li)) <= R^2]
                left[grp] .= false
                push!(joint, _AdaptEvent{D, T}(pts[a], Int[], pts[grp]))
            end
            rest = joint
            while !isempty(rest)
                sel = _spaced(hg, [ev.center for ev in rest], _event_spacing(hg))
                _round_body!(hg, ad, rest[sel], max_order, df)
                rest = rest[setdiff(eachindex(rest), sel)]
            end
        else
            _round_body!(hg, ad, events, max_order, df)
        end
        return nothing
    end
end
