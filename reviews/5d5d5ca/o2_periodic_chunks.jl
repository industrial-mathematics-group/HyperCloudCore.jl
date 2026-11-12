# Probe-only in-process overrides of rr-moving3 (5d5d5ca) to find what changed the receding-piston errors:
#   VARIANT=base       unchanged
#   VARIANT=nowall     insertion also needs signed_distance(region, x) <= -gap (8d41d25's rule: no wall-row particles)
#   VARIANT=rounds     insertions in rounds of `_spaced` events again (8d41d25's scheme), wall rows kept
# CASE=p14 (r2's data, ∂ₙu ≠ 0 at the face) | r8 (consistent data, uncovered part by distance) | r8c (ExtrapolationBC face)
include("common.jl"); include("o_chunks.jl")
const VARIANT = get(ENV, "VARIANT", "base")
const CASE = get(ENV, "CASE", "p14")
const NS = parse.(Int, split(get(ENV, "NS", "20,40,80"), ","))
if VARIANT == "nowall"
    @eval H function _uncovered_points(hg::ParticleGrid{D, M, T}, geom, shape_t, posed, boxes, gap::T, reach::T, t::T) where {D, M, T}
        out = Space{D, T}[]
        idx, dvec = Int[], Space{D, T}[]
        L, Li = hg.domain.L_wrap, hg.domain.invL_wrap
        for b in boxes
            _sweep_lattice(hg, geom, b) do x
                _is_interior(geom, x, t) || return
                signed_distance(shape_t, x) <= -gap || return
                dmin = minimum(p -> abs(signed_distance(p, x)), posed)
                gap <= dmin <= reach || return
                _collect_near!(hg, x, gap^2, idx, dvec) == 0 || return
                any(y -> sum(abs2, _wrapped_diff(y, x, L, Li)) < gap^2, out) && return
                push!(out, x)
            end
        end
        return out
    end
elseif VARIANT == "rounds"
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
            rest = events
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
# p6's case: channel periodic in y, receding co-moving piston, Gaussian pulse; volumes and where the extremes are
u0(x, t) = 1 + exp(-((x[1] - 0.8 - t)^2 + (x[2] - 0.25)^2) / 0.02)
for N in NS
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
    geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, true)), piston); is_periodic = (false, true),
                           mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 5 => MirrorBC(; degree = 1)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4h; t = 0.0
    while t < 0.5 - 1e-12; d = min(dt, 0.5 - t); ts(eq, pg, t, d); t += d; end
    face = 1 + t
    ii = interior(pg)
    V = [pg.core.volumes[i] / h^2 for i in ii]
    e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in ii]
    k = argmax(e)
    cols(sel) = sort(unique(round.([(pg.core.positions[ii[j]][1] - 1) / h for j in sel], digits = 2)))
    @printf("p6 %s N=%d: err %.4f at x = 1 + %.2f h (face - %.2f h); V/h² in [%.3f, %.3f]; columns (x - 1)/h with V < 0.8 h²: %s, > 1.2 h²: %s\n",
            VARIANT, N, e[k], (pg.core.positions[ii[k]][1] - 1) / h, (face - pg.core.positions[ii[k]][1]) / h, extrema(V)...,
            cols(findall(<(0.8), V)), cols(findall(>(1.2), V)))
end
