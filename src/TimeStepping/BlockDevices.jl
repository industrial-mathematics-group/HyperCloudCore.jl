# =========================================================================
# A BLOCK STATE SPLIT OVER DEVICES (the velocity decomposition)
#
# Every device holds the whole particle cloud and a range of blocks of the state, with its own stage buffers and
# scheme; the flux of a block depends on that block alone, so the right-hand side needs no exchange. What couples the
# blocks are per-particle reductions (the source parameters, the frame velocity of a moving grid, the cross-block
# admissibility of MOOD, the kinetic walls' parameters): the devices exchange their partial sums, (D + 2) N numbers
# at a time, through host buffers (summed in block order), plus the MOOD flags (an OR) and, for a kinetic wall whose
# operator reads other blocks of a ghost, the mirrored ghost rows. The step runs in lockstep phases over the parts.
# =========================================================================
export BlockDevices, BlockPart, host_state

"""
    BlockPart(pg, ts::BlockTimeStepper, bs::BlockState; activate = nothing)

One device's share of a split block state (see [`BlockDevices`](@ref)): its grid (the whole cloud, on the device's
storage), its stepper and its state, both built for the same block range (`blocks` of [`BlockTimeStepper`](@ref) and
[`BlockState`](@ref)), and `activate`, a callable that makes the device current before the part's launches
(`() -> CUDA.device!(d)`; `nothing` for a host part or a single device).
"""
struct BlockPart{PG <: ParticleGrid, TS <: BlockTimeStepper, BS <: BlockState, F}
    pg::PG
    ts::TS
    bs::BS
    activate::F
    scratch::Dict{DataType, Any}    # the exchange array per partial type (a `Growable`, on the part's storage)
    rows::Dict{Int, Any}            # per wall tag: the gathered mirrored ghost rows of all blocks and the ghost map
end
function BlockPart(pg::ParticleGrid{D, W, T}, ts::BlockTimeStepper{D, W, T}, bs::BlockState{W, T}; activate = nothing) where {D, W, T}
    ts.blocks == bs.blocks || throw(ArgumentError("BlockPart: the stepper runs the blocks $(ts.blocks), the state holds $(bs.blocks)"))
    return BlockPart(pg, ts, bs, activate, Dict{DataType, Any}(), Dict{Int, Any}())
end
@inline _activate(pt::BlockPart) = (pt.activate === nothing || pt.activate(); nothing)
_storage(pt::BlockPart) = typeof(parent(pt.bs.data)).name.wrapper
function _part_scratch!(pt::BlockPart, ::Type{P}, n::Int) where {P}
    g = get!(pt.scratch, P) do
        Growable{P}(_storage(pt))
    end::Growable{P}
    ensure_capacity!(g, n)
    return g
end

"""
    BlockDevices(eq, parts::Vector{<:BlockPart})
    (devs::BlockDevices)(time, dt)
    get_time_step(devs::BlockDevices; cached_weights = true)

A block state of the block equation `eq` split over devices: `parts` hold consecutive block ranges that cover
`1:num_blocks(eq)` in order, each on its device (its storage) with the whole cloud, its stepper and its state. A step
runs the phases of [`BlockTimeStepper`](@ref) in lockstep over the parts and exchanges, through host buffers, the
per-particle partial sums of the reductions, the MOOD flags and the mirrored ghost rows of the kinetic walls:

- the sources give their reduction in the additive form ([`reduction_partial_type`](@ref)), the frame velocity of a
  Lagrangian mover in [`fluid_velocity_partial_type`](@ref), the cross-block admissibility of MOOD in
  [`admissible_partial_type`](@ref) (an equation that implements `block_admissible` without it is refused), the
  operator of a kinetic wall in [`wall_partial_type`](@ref); the sums are formed in block order on the host;
- the MOOD flag pass runs per part, the flags are OR-ed over the parts, the drop pass runs on every part (the same
  orders everywhere), as does every host-side decision;
- the parameters of a kinetic wall come from the exchanged partials, and `block_wall_apply` reads the other blocks of
  a ghost from the gathered ghost rows of all blocks (per tag: `nb × ghosts` states, exchanged per application).

The parts' grids are copies of one cloud (a `device_grid` per device; copies, not the shared arrays of
`device_grid(Array, pg)`: every part moves its own positions and orders). Not supported on a split state: moving
boundary pieces, the particle adaptivity and the reordering (host events of the single stepper). The result agrees
with the single stepper to round-off (the partial sums are added in a different order).
"""
struct BlockDevices{D, W, T, EQ <: HyperbolicPDE{D, W, T}, P <: BlockPart}
    pde::EQ
    parts::Vector{P}
    host::Dict{Any, Any}            # host exchange buffers: a partial type → (recv, acc); a wall tag → the ghost rows
    ghosts::Dict{Int, Any}          # per wall tag: (the ghost indices, the geometry version they belong to)
end
function BlockDevices(eq::HyperbolicPDE{D, W, T}, parts::Vector{P}) where {D, W, T, P <: BlockPart}
    isempty(parts) && throw(ArgumentError("BlockDevices: no parts"))
    nb = num_blocks(eq)
    first(first(parts).bs.blocks) == 1 && last(last(parts).bs.blocks) == nb ||
        throw(ArgumentError("BlockDevices: the parts must cover the blocks 1:$nb, they hold $(map(pt -> pt.bs.blocks, parts))"))
    for k in 2:length(parts)
        first(parts[k].bs.blocks) == last(parts[k - 1].bs.blocks) + 1 ||
            throw(ArgumentError("BlockDevices: the parts' blocks must be consecutive, got $(parts[k - 1].bs.blocks) then $(parts[k].bs.blocks)"))
    end
    N = first(parts).pg.meta.N
    all(pt -> pt.pg.meta.N == N && pt.bs.N[] == N, parts) || throw(ArgumentError("BlockDevices: the parts' grids and states must hold the same particles"))
    p1 = first(parts)
    for pt in parts
        pt.ts.pde === eq || pt.ts.pde == eq || throw(ArgumentError("BlockDevices: every part's stepper must be built for the equation"))
        (pt.ts.inner.num_stages == p1.ts.inner.num_stages && pt.ts.inner.tableau == p1.ts.inner.tableau &&
         length(pt.ts.explicit_sources) == length(p1.ts.explicit_sources) && length(pt.ts.implicit_sources) == length(p1.ts.implicit_sources)) ||
            throw(ArgumentError("BlockDevices: the parts' steppers must share the tableau and the sources"))
        has_moving_pieces(pt.pg.geometry) && throw(ArgumentError("BlockDevices: moving boundary pieces are not supported on a split state"))
    end
    # the additive forms the exchange needs
    for st in (p1.ts.explicit_sources..., p1.ts.implicit_sources...)
        reduction_partial_type(st.term) === nothing &&
            throw(ArgumentError("BlockDevices: the source $(typeof(st.term)) gives no additive form of its reduction (`reduction_partial_type`)"))
    end
    if !(p1.ts.inner.mood.criterion isa NoMOOD) && admissible_partial_type(eq) === nothing && _overrides_block_admissible(eq)
        throw(ArgumentError("BlockDevices: $(typeof(eq)) implements `block_admissible` without its additive form (`admissible_partial_type`), which a split state needs"))
    end
    _lagrangian_type(p1.pg.mover) && fluid_velocity_partial_type(eq) === nothing &&
        throw(ArgumentError("BlockDevices: a Lagrangian mover on a split state needs the additive form of `block_fluid_velocity` (`fluid_velocity_partial_type`)"))
    for (tag, bc) in p1.pg.geometry.bc_map
        _is_block_wall(bc, eq) && wall_partial_type(bc.op, eq) === nothing &&
            throw(ArgumentError("BlockDevices: the wall operator $(typeof(bc.op)) of tag $tag gives no additive form of its reduction (`wall_partial_type`)"))
    end
    return BlockDevices{D, W, T, typeof(eq), P}(eq, parts, Dict{Any, Any}(), Dict{Int, Any}())
end
# Whether an equation implements `block_admissible` itself (the default is `true`, additive form or not)
function _overrides_block_admissible(eq::HyperbolicPDE{D, W, T}) where {D, W, T}
    m = which(block_admissible, Tuple{typeof(eq), Vector{State{W, T}}, Int, Int, Int})
    return Base.unwrap_unionall(m.sig).parameters[2] !== HyperbolicPDE
end

"""
    host_state(devs::BlockDevices) -> Vector{State{W, T}}

The whole block-major state (`nb N` entries, block `b` at `(b − 1) N + 1 … b N`) gathered from the parts onto the host.
"""
function host_state(devs::BlockDevices{D, W, T}) where {D, W, T}
    N   = first(devs.parts).pg.meta.N
    out = Vector{State{W, T}}(undef, num_blocks(devs.pde) * N)
    for pt in devs.parts
        _activate(pt)
        copyto!(out, (first(pt.bs.blocks) - 1) * N + 1, parent(pt.bs.data), 1, pt.bs.nb * N)
    end
    return out
end

# =========================================================================
# THE EXCHANGES
# =========================================================================
# The host buffers of a partial type: the receive buffer and the accumulator
function _host_buffers!(devs::BlockDevices, ::Type{P}, N::Int) where {P}
    recv, acc = get!(devs.host, P) do
        (Vector{P}(undef, 0), Vector{P}(undef, 0))
    end::Tuple{Vector{P}, Vector{P}}
    length(recv) < N && (resize!(recv, N); resize!(acc, N))
    return recv, acc
end
# The sum over the parts of their exchange arrays of type `P` (one entry per particle), in part order, written back into
# every part's array; returns the host sum
function _exchange_sum!(devs::BlockDevices, ::Type{P}, N::Int) where {P}
    recv, acc = _host_buffers!(devs, P, N)
    fill!(view(acc, 1:N), zero(P))
    for pt in devs.parts
        _activate(pt)
        copyto!(recv, 1, parent(pt.scratch[P]::Growable{P}), 1, N)
        @inbounds for i in 1:N
            acc[i] += recv[i]
        end
    end
    for pt in devs.parts
        _activate(pt)
        copyto!(parent(pt.scratch[P]::Growable{P}), 1, acc, 1, N)
    end
    return acc
end
# The OR over the parts of a Bool array of each (`get(pt)`), written back into every part's array; returns the host OR
function _exchange_or!(devs::BlockDevices, get::F, N::Int) where {F}
    recv, acc = _host_buffers!(devs, Bool, N)
    fill!(view(acc, 1:N), false)
    for pt in devs.parts
        _activate(pt)
        copyto!(recv, 1, get(pt), 1, N)
        @inbounds for i in 1:N
            acc[i] |= recv[i]
        end
    end
    for pt in devs.parts
        _activate(pt)
        copyto!(get(pt), 1, acc, 1, N)
    end
    return acc
end

# Kernel bodies (see Backend.jl) of the exchanged reductions: the partial over the part's blocks (the part's array seen
# with global block indices), and the finalization from the summed partials
@inline function _block_partial_body!(p, out, st, Y, N, nb, blocks, pg, is_boundary, t)
    is_boundary[p] && return nothing
    out[p] = block_partial(st, p, Y, N, nb, blocks, pg, t)
    return nothing
end
@inline function _block_finalize_body!(p, params, st, sums, pg, is_boundary, t, dt_coeff)
    is_boundary[p] && return nothing
    params[p] = finalize_reduction(st, p, sums[p], pg, t, dt_coeff)
    return nothing
end
@inline function _velocity_partial_body!(p, out, eq, Y, N, nb, blocks, is_boundary)
    is_boundary[p] && return nothing
    out[p] = block_fluid_velocity_partial(eq, Y, N, nb, blocks, p)
    return nothing
end
@inline function _velocity_finalize_body!(p, w, eq, sums, alpha, is_boundary, zero_space)
    w[p] = is_boundary[p] ? zero_space : alpha * finalize_fluid_velocity(eq, sums[p])
    return nothing
end
@inline function _admissible_partial_body!(p, out, eq, Y, N, nb, blocks, is_boundary, dirty, full)
    (is_boundary[p] || !(full || dirty[p])) && return nothing
    out[p] = block_admissible_partial(eq, Y, N, nb, blocks, p)
    return nothing
end
@inline function _admissible_finalize_body!(p, triggered, eq, sums, is_boundary, dirty, full)
    (is_boundary[p] || !(full || dirty[p])) && return nothing
    finalize_admissible(eq, sums[p]) || (triggered[p] = true)
    return nothing
end
@inline function _wall_partial_body!(g, out, op, eq, S, N, nb, blocks, pos, is_boundary, tags, normals, wdist, curvs, gm, tag)
    (is_boundary[g] && tags[g] == tag) || return nothing
    out[g] = block_wall_partial(op, eq, S, N, nb, blocks, g, _ghost_wall_point(g, pos, normals, wdist, curvs, gm))
    return nothing
end
@inline function _wall_finalize_body!(g, params, op, eq, sums, pos, is_boundary, tags, normals, wdist, curvs, gm, tag)
    (is_boundary[g] && tags[g] == tag) || return nothing
    params[g] = finalize_wall_reduction(op, eq, sums[g], _ghost_wall_point(g, pos, normals, wdist, curvs, gm))
    return nothing
end
# The local flags of the inadmissible count: a block of the part not admissible (the cross-block check is exchanged)
@inline function _local_inadmissible_body!(p, flags, is_boundary, eq, Y, N, nb, off)
    if is_boundary[p]
        flags[p] = false
        return nothing
    end
    ok = true
    for b in 1:nb
        ok &= block_is_admissible(eq, Y[(b - 1) * N + p], b + off)
    end
    flags[p] = !ok
    return nothing
end
# The gathered mirrored ghost rows of all blocks of a wall tag (`rows[(b − 1) ng + j]`: block `b` of the `j`-th ghost)
# seen as the block-major scratch of the whole state (`x[(b − 1) N + g]`, `gmap[g] = j`); a bits kernel argument
struct _GhostRows{A, M}
    rows::A
    gmap::M
    N::Int
    ng::Int
end
@inline function Base.getindex(x::_GhostRows, i::Int)
    b = (i - 1) ÷ x.N + 1
    g = (i - 1) % x.N + 1
    return @inbounds x.rows[(b - 1) * x.ng + x.gmap[g]]
end
Base.eltype(::Type{_GhostRows{A, M}}) where {A, M} = eltype(A)
Adapt.adapt_structure(to, x::_GhostRows) = _GhostRows(adapt(to, x.rows), adapt(to, x.gmap), x.N, x.ng)
@inline function _gather_rows_body!(q, rows, S, ghost_idx, N, ng)
    b = (q - 1) ÷ ng + 1
    j = (q - 1) % ng + 1
    @inbounds rows[q] = S[(b - 1) * N + ghost_idx[j]]
    return nothing
end

# The ghosts of a tag (host indices), cached by the geometry version of the first part's grid, with the per-part
# device copies of the index list and of the map particle → position in the list
function _ghost_list!(devs::BlockDevices, tag::Int, N::Int)
    pg1 = first(devs.parts).pg
    v   = pg1.meta.geometry_version
    cached = get(devs.ghosts, tag, nothing)
    cached !== nothing && cached[2] == v && return cached[1]::Vector{Int}
    isb  = Array(view(pg1.core.is_boundary, 1:N))
    tags = Array(view(pg1.core.tags, 1:N))
    list = findall(i -> isb[i] && tags[i] == tag, 1:N)
    gmap = zeros(Int, N)
    for (j, g) in enumerate(list)
        gmap[g] = j
    end
    for pt in devs.parts
        _activate(pt)
        AT = _storage(pt)
        pt.rows[tag] = (Growable{eltype(parent(pt.ts.ghost_scratch))}(AT), Growable(adapt(AT, list)), Growable(adapt(AT, gmap)), Growable{eltype(parent(pt.ts.ghost_scratch))}(AT))
    end
    devs.ghosts[tag] = (list, v)
    return list
end
# The mirrored ghost rows of all blocks of `tag`, gathered from the parts' scratches and given to every part
function _gather_ghost_rows!(devs::BlockDevices{D, W, T}, tag::Int, N::Int) where {D, W, T}
    list = _ghost_list!(devs, tag, N)
    ng   = length(list)
    nb   = num_blocks(devs.pde)
    full = get!(devs.host, (:rows, tag)) do
        Vector{State{W, T}}(undef, 0)
    end::Vector{State{W, T}}
    length(full) < nb * ng && resize!(full, nb * ng)
    ng == 0 && return nothing
    for pt in devs.parts
        _activate(pt)
        rows_all, idx, gmap, rows_local = pt.rows[tag]
        nbl = pt.bs.nb
        ensure_capacity!(rows_local, nbl * ng)
        foreach_particle(_gather_rows_body!, get_backend(), nbl * ng, (rows_local, parent(pt.ts.ghost_scratch), idx, N, ng))
        copyto!(full, (first(pt.bs.blocks) - 1) * ng + 1, parent(rows_local), 1, nbl * ng)
    end
    for pt in devs.parts
        _activate(pt)
        rows_all = pt.rows[tag][1]
        ensure_capacity!(rows_all, nb * ng)
        copyto!(parent(rows_all), 1, full, 1, nb * ng)
    end
    return nothing
end
_ghost_rows(pt::BlockPart, tag::Int, N::Int, ng::Int) = (r = pt.rows[tag]; _GhostRows(parent(r[1]), parent(r[3]), N, ng))

# =========================================================================
# THE STEP IN LOCKSTEP OVER THE PARTS
# =========================================================================
function (devs::BlockDevices{D, W, T})(time::Real, dt::T) where {D, W, T}
    eq    = devs.pde
    parts = devs.parts
    for pt in parts
        _activate(pt)
        _block_step_begin!(pt.ts, eq, pt.pg, pt.bs, time, dt)
    end
    p1     = first(parts)
    inner1 = p1.ts.inner
    s      = inner1.num_stages
    bt     = inner1.tableau
    N      = p1.pg.meta.N
    moving = _block_moving(p1.pg)
    max_order = _extract_order(inner1.divergence_interpolator)
    for i in 1:s
        if i == 1
            _devices_form_candidate!(devs, 1, dt, time)
        else
            _devices_checkpoint!(devs, i, dt, time)
            if moving
                for pt in parts
                    _activate(pt)
                    _apply_stage_geometry!(pt.ts.inner, pt.pg, pt.ts.inner.X_stages[i], time, time + bt.c_t[i] * dt)
                end
            end
        end
        if inner1.reset_orders_per_stage
            for pt in parts
                _activate(pt)
                fill!(view(pt.pg.core.particle_orders, 1:N), max_order)
            end
        end
        _devices_rhs!(devs, i, time + bt.c_t[i] * dt, true)
    end
    _devices_checkpoint!(devs, s + 1, dt, time)
    for pt in parts
        _activate(pt)
        _block_step_end!(pt.ts, eq, pt.pg, pt.bs, time, dt)
    end
    return nothing
end

# The stage array of checkpoint / stage `i` of a part
@inline _part_stage(pt::BlockPart, i::Int) = (inner = pt.ts.inner; i > inner.num_stages ? inner.rho_cand : inner.Y_stages[i])

# The reduction of source `n` (explicit or implicit) at the stage array of index `i`: the partials over every part's
# blocks, the sum, the parameters on every part
function _devices_reduce!(devs::BlockDevices{D, W, T}, which::Symbol, n::Int, i::Int, t_i, dt_coeff) where {D, W, T}
    N  = first(devs.parts).pg.meta.N
    nb = num_blocks(devs.pde)
    src(pt) = which === :implicit ? pt.ts.implicit_sources[n] : pt.ts.explicit_sources[n]
    P = reduction_partial_type(src(first(devs.parts)).term)
    for pt in devs.parts
        _activate(pt)
        st  = src(pt)
        buf = _part_scratch!(pt, P, N)
        off = first(pt.bs.blocks) - 1
        foreach_particle(_block_partial_body!, get_backend(), N,
            (buf, st.term, _OffsetBlocks(parent(_part_stage(pt, i)), off * N), N, nb, pt.bs.blocks, pt.pg, pt.pg.core.is_boundary, t_i))
    end
    _exchange_sum!(devs, P, N)
    for pt in devs.parts
        _activate(pt)
        st = src(pt)
        ensure_capacity!(st.params, N)
        foreach_particle(_block_finalize_body!, get_backend(), N,
            (st.params, st.term, pt.scratch[P], pt.pg, pt.pg.core.is_boundary, t_i, dt_coeff))
    end
    return nothing
end

function _devices_form_candidate!(devs::BlockDevices{D, W, T}, i::Int, dt::T, time::Real) where {D, W, T}
    parts  = devs.parts
    eq     = devs.pde
    inner1 = first(parts).ts.inner
    s      = inner1.num_stages
    bt     = inner1.tableau
    N      = first(parts).pg.meta.N
    is_final = i > s
    t_i = is_final ? time + dt : time + bt.c_t[i] * dt
    for pt in parts
        _activate(pt)
        _launch_predictor!(pt.ts, pt.pg, N, pt.bs.nb, i, dt)
    end
    if !is_final && !isempty(first(parts).ts.implicit_sources)
        a_ii = bt.a[i, i]
        implicit = abs(a_ii) > T(1e-14)
        for n in eachindex(first(parts).ts.implicit_sources)
            _devices_reduce!(devs, :implicit, n, i, t_i, implicit ? dt * a_ii : zero(T))
            for pt in parts
                _activate(pt)
                src = pt.ts.implicit_sources[n]
                if implicit
                    _launch_implicit_step!(pt.ts, pt.pg, src, _part_stage(pt, i), pt.ts.inner.K_I_stages[i], N, pt.bs.nb, dt * a_ii, t_i, n == 1)
                else
                    _launch_implicit_rhs!(pt.ts, pt.pg, src, _part_stage(pt, i), pt.ts.inner.K_I_stages[i], N, pt.bs.nb, t_i, n == 1)
                end
            end
        end
    end
    _devices_apply_bcs!(devs, i, t_i)
    return t_i
end

# The boundary conditions of the stage array `i` on every part: a kinetic wall with the exchange of its partials and
# the gather of the mirrored ghost rows, every other condition per part
function _devices_apply_bcs!(devs::BlockDevices{D, W, T}, i::Int, t) where {D, W, T}
    parts = devs.parts
    eq    = devs.pde
    N     = first(parts).pg.meta.N
    nb    = num_blocks(eq)
    for (tag, bc) in first(parts).pg.geometry.bc_map
        if _is_block_wall(bc, eq)
            P = wall_partial_type(bc.op, eq)
            for pt in parts
                _activate(pt)
                _launch_wall_fit!(pt.ts, bc, pt.pg, _part_stage(pt, i), tag, eq, t, pt.bs.nb)
                buf = _part_scratch!(pt, P, N)
                c   = pt.pg.core
                gm  = _wall_motion(pt.pg, tag, t)
                foreach_particle(_wall_partial_body!, get_backend(), N,
                    (buf, bc.op, eq, _OffsetBlocks(parent(pt.ts.ghost_scratch), (first(pt.bs.blocks) - 1) * N), N, nb, pt.bs.blocks,
                     c.positions, c.is_boundary, c.tags, c.normals, c.wall_distances, c.wall_curvatures, gm, tag))
            end
            _exchange_sum!(devs, P, N)
            _gather_ghost_rows!(devs, tag, N)
            ng = length(devs.ghosts[tag][1])
            for pt in parts
                _activate(pt)
                params = _wall_params!(pt.ts, bc, tag, eq)
                ensure_capacity!(params, N)
                c  = pt.pg.core
                gm = _wall_motion(pt.pg, tag, t)
                foreach_particle(_wall_finalize_body!, get_backend(), N,
                    (params, bc.op, eq, pt.scratch[P], c.positions, c.is_boundary, c.tags, c.normals, c.wall_distances, c.wall_curvatures, gm, tag))
                _launch_wall_apply!(pt.ts, bc, pt.pg, _part_stage(pt, i), tag, eq, t, N, pt.bs.nb, params, _ghost_rows(pt, tag, N, ng))
            end
        else
            for pt in parts
                _activate(pt)
                _block_bc!(bc, pt.ts, pt.pg, _part_stage(pt, i), tag, eq, t, N, pt.bs.nb)
            end
        end
    end
    return nothing
end

# The right-hand side of stage `i` on every part: the frame velocity of a moving grid (exchanged for the Lagrangian
# movers), the flux pass per part, the explicit sources with their exchanged reductions
function _devices_rhs!(devs::BlockDevices{D, W, T}, i::Int, t_i, full::Bool) where {D, W, T}
    parts = devs.parts
    eq    = devs.pde
    N     = first(parts).pg.meta.N
    nb    = num_blocks(eq)
    pg1   = first(parts).pg
    if _block_moving(pg1)
        if _lagrangian_type(pg1.mover)
            P = fluid_velocity_partial_type(eq)
            for pt in parts
                _activate(pt)
                buf = _part_scratch!(pt, P, N)
                foreach_particle(_velocity_partial_body!, get_backend(), N,
                    (buf, eq, _OffsetBlocks(parent(pt.ts.inner.Y_stages[i]), (first(pt.bs.blocks) - 1) * N), N, nb, pt.bs.blocks, pt.pg.core.is_boundary))
            end
            _exchange_sum!(devs, P, N)
            for pt in parts
                _activate(pt)
                alpha = pt.pg.mover isa ScaledLagrangianMover ? pt.pg.mover.alpha : one(T)
                foreach_particle(_velocity_finalize_body!, get_backend(), N,
                    (pt.ts.grid_w, eq, pt.scratch[P], alpha, pt.pg.core.is_boundary, zero(Space{D, T})))
            end
        else
            for pt in parts
                _activate(pt)
                foreach_particle(_block_velocity_body!, get_backend(), N,
                    (pt.ts.grid_w, pt.pg.mover, eq, parent(pt.ts.inner.Y_stages[i]), N, pt.bs.nb, pt.pg, pt.pg.core.is_boundary, t_i, zero(Space{D, T})))
            end
        end
    end
    for pt in parts
        _activate(pt)
        _launch_flux!(pt.ts, eq, pt.pg, i, t_i, full, N, pt.bs.nb)
    end
    for n in eachindex(first(parts).ts.explicit_sources)
        full && _devices_reduce!(devs, :explicit, n, i, t_i, zero(T))
        for pt in parts
            _activate(pt)
            _launch_explicit_source!(pt.ts, pt.pg, pt.ts.explicit_sources[n], i, N, pt.bs.nb, t_i, full)
        end
    end
    return nothing
end

# The MOOD checkpoint on every part (see `_block_checkpoint!`): the same fixed point everywhere, the flags exchanged
function _devices_checkpoint!(devs::BlockDevices{D, W, T}, i::Int, dt::T, time::Real) where {D, W, T}
    parts  = devs.parts
    inner1 = first(parts).ts.inner
    N      = first(parts).pg.meta.N
    t_prev = time + inner1.tableau.c_t[i - 1] * dt
    reach  = _order_reach(inner1.divergence_interpolator, inner1.mood.strategy)
    full = true
    while true
        _devices_form_candidate!(devs, i, dt, time)
        cand_sub = _checks_substeps(inner1)
        if cand_sub
            for pt in parts
                _activate(pt)
                _block_form_substep!(pt.ts, pt.pg, i, dt)
            end
        end
        if !_devices_detect_and_drop!(devs, i, cand_sub, full)
            inner1.mood.criterion isa NoMOOD || (inner1.work.inadmissible[] += _devices_count_inadmissible(devs, i, cand_sub))
            return nothing
        end
        for pt in parts
            _activate(pt)
            work = pt.ts.inner.work
            copyto!(parent(work.dirty), 1, parent(work.changed), 1, N)
            expand_mask!(parent(work.dirty), parent(work.scratch), pt.pg, reach)
        end
        _devices_rhs!(devs, i - 1, t_prev, false)
        full = false
    end
end

# The partials of the cross-block admissibility of the candidate `i` on every part, summed (the host sum returned)
function _devices_admissible_sums!(devs::BlockDevices, i::Int, cand_sub::Bool, full::Bool, N::Int)
    eq = devs.pde
    P  = admissible_partial_type(eq)
    nb = num_blocks(eq)
    for pt in devs.parts
        _activate(pt)
        buf = _part_scratch!(pt, P, N)
        foreach_particle(_admissible_partial_body!, get_backend(), N,
            (buf, eq, _OffsetBlocks(_block_candidate_array(pt.ts, i, cand_sub), (first(pt.bs.blocks) - 1) * N), N, nb, pt.bs.blocks,
             pt.pg.core.is_boundary, parent(pt.ts.inner.work.dirty), full))
    end
    return _exchange_sum!(devs, P, N)
end

function _devices_detect_and_drop!(devs::BlockDevices, i::Int, cand_sub::Bool, full::Bool)
    parts  = devs.parts
    eq     = devs.pde
    inner1 = first(parts).ts.inner
    inner1.mood.criterion isa NoMOOD && return false
    N = first(parts).pg.meta.N
    for pt in parts
        _activate(pt)
        _mood_reset!(pt.ts.inner.work, N)
        _launch_flag_pass!(pt.ts, eq, pt.pg, i, cand_sub, full, N, pt.bs.nb)
    end
    P = admissible_partial_type(eq)
    if P !== nothing
        _devices_admissible_sums!(devs, i, cand_sub, full, N)
        for pt in parts
            _activate(pt)
            work = pt.ts.inner.work
            foreach_particle(_admissible_finalize_body!, get_backend(), N,
                (parent(work.triggered), eq, pt.scratch[P], pt.pg.core.is_boundary, parent(work.dirty), full))
        end
    end
    _exchange_or!(devs, pt -> parent(pt.ts.inner.work.triggered), N)
    changed = false
    for pt in parts
        _activate(pt)
        changed = _mood_drop_pass!(pt.ts.inner.mood, pt.ts.inner.work, pt.pg)   # (the same on every part)
    end
    return changed
end

# Interior particles whose candidate is not admissible: a block of any part, or the cross-block check of the summed partials
function _devices_count_inadmissible(devs::BlockDevices, i::Int, cand_sub::Bool)
    parts = devs.parts
    eq    = devs.pde
    N     = first(parts).pg.meta.N
    N == 0 && return 0
    for pt in parts
        _activate(pt)
        foreach_particle(_local_inadmissible_body!, get_backend(), N,
            (parent(pt.ts.inner.work.scratch), pt.pg.core.is_boundary, eq, _block_candidate_array(pt.ts, i, cand_sub), N, pt.bs.nb, first(pt.bs.blocks) - 1))
    end
    flags = copy(_exchange_or!(devs, pt -> parent(pt.ts.inner.work.scratch), N))
    P = admissible_partial_type(eq)
    if P !== nothing
        sums = _devices_admissible_sums!(devs, i, cand_sub, true, N)
        isb  = Array(view(first(parts).pg.core.is_boundary, 1:N))
        for p in 1:N
            isb[p] && continue
            flags[p] |= !finalize_admissible(eq, sums[p])
        end
    end
    return count(view(flags, 1:N))
end

"""
    get_time_step(devs::BlockDevices; cached_weights = true) -> T

The CFL step of a split block state: the smallest over the parts (`get_time_step` of every part's state).
"""
function get_time_step(devs::BlockDevices{D, W, T}; cached_weights::Bool = true) where {D, W, T}
    dt = T(Inf)
    for pt in devs.parts
        _activate(pt)
        dt = min(dt, get_time_step(pt.pg, devs.pde, pt.ts.inner.divergence_interpolator, pt.bs; cached_weights))
    end
    return dt
end

"""
    set_block_states!(f, devs::BlockDevices)

Sets every part's blocks from `f(x, b)` (the global block index `b`), as `set_block_states!` of a single state.
"""
function set_block_states!(f, devs::BlockDevices)
    for pt in devs.parts
        _activate(pt)
        set_block_states!(f, pt.bs, pt.pg)
    end
    return nothing
end
