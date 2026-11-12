# Probe-only in-process overrides of rr-moving3 (5d5d5ca) to find what changed the receding-piston errors:
#   VARIANT=base       unchanged
#   VARIANT=nowall     insertion also needs signed_distance(region, x) <= -gap (8d41d25's rule: no wall-row particles)
#   VARIANT=rounds     insertions in rounds of `_spaced` events again (8d41d25's scheme), wall rows kept
# CASE=p14 (r2's data, ∂ₙu ≠ 0 at the face) | r8 (consistent data, uncovered part by distance) | r8c (ExtrapolationBC face)
include("common.jl")
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
# docs: the plate with 1 + ½ cos 5π(x − t) (zero normal derivative on the plate's faces)
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
plate_shape = Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4)
function run(N, tend, u0)
    h = 1 / N
    chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - Moving(plate_shape, translating((1.0, 0.0)))
    geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                            bc_map = BCD(t => MirrorBC(; degree = 1) for t in (3, 4, 5, 6, 7, 8)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eq, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eq, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq, pg, t, d); t += d; end
    inner = interior(pg)
    dpl(x) = signed_distance(plate_shape, x - SVector(t, 0.0))
    e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in inner]
    dd = [dpl(pg.core.positions[i]) for i in inner]
    @printf("  N=%3d max err %.4g; within 12h of the plate %.4g; beyond 12h %.4g\n", N, maximum(e),
            maximum(e[k] for k in eachindex(e) if dd[k] < 12h), maximum(e[k] for k in eachindex(e) if dd[k] >= 12h; init=0.0))
end
uc(x, t) = 1 + 0.5 * cos(5π * (x[1] - t))
println("== plate, u0 = 1 + ½ cos 5π(x - t), t = 0.5")
for N in (20, 40, 80); run(N, 0.5, uc); end
