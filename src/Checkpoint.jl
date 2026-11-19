# =========================================================================
# CHECKPOINTS (restart of a run that was stopped, e.g. preempted on a cluster)
#
# A run resumes from the state at the end of a step. What cannot be recomputed is saved: the states, and the particle
# set with its geometry (positions in memory order, volumes, the boundary data, the counts), which particle movers,
# adaptivity and the sweep of moving pieces change during a run, and the neighbour lists with their weights and
# distances: they are derived, but not reproducibly anywhere (a run's grid is built on the host and searched on the
# device afterwards, an adaptivity event searches a host copy, and a GPU's `exp` rounds differently from the host's).
# Everything else is rebuilt: the cell list (a search over the restored positions, whose lists are then replaced by the
# saved ones), the geometry caches of the schemes (stale after the restore, `geometry_version`), the stepper's stage
# buffers and MOOD masks (set at the start of every step). A run rebuilt from its configuration and given the checkpoint
# continues bit for bit as the run that wrote it would have (same backend and device model; a GPU of another model may
# round differently).
# =========================================================================
export checkpoint_state, restore_checkpoint!

# The layout of the named tuples below; a checkpoint of another format is refused
const CHECKPOINT_FORMAT = 1

# The first `n` entries of a grid array as a host `Vector` (always a copy: a checkpoint must not alias the grid)
_host_copy(a, n::Int) = Array(view(parent(a), 1:n))

"""
    checkpoint_state(pg::ParticleGrid) -> NamedTuple
    checkpoint_state(pg::ParticleGrid, bs::BlockState) -> NamedTuple
    checkpoint_state(devs::BlockDevices) -> NamedTuple

The state of a run at the end of a step, as host arrays (copies, device grids included), to restore with
[`restore_checkpoint!`](@ref) into the same run rebuilt from its configuration:

- the particle set: `positions` (in memory order), `volumes`, `is_boundary`, `tags`, `particle_orders`, `normals`,
  `wall_distances`, `wall_curvatures` and the counts `N`, `N_interior`, `N_ghost`;
- `boundary_time` and `sweep_time` (the time the moving pieces' ghosts and the swept region belong to);
- the neighbour lists: `neighbor_counts` (per particle) and `neighbor_indices`, `neighbor_weights`, `neighbor_distances`
  (particle by particle, in slot order; independent of the grid's slot layout);
- the states: `rhos`, and for a block state `blocks` (the block-major array of all blocks, `nb N` entries) with `nb`;
- for the consistency check of the restore: `format`, `D`, `M`, `T` (the float type's name), `R` and `dx`.

The neighbour lists are saved although a search would find them again: their weights are not reproducible across
devices (a run's grid is built and searched on the host, then stepped on a GPU whose `exp` rounds differently). The
caches derived from the geometry are not part of it (the restore makes them stale). A split
block state ([`BlockDevices`](@ref)) is saved as one block state: the grid of the first part (every part holds the same
cloud) and the blocks of all parts ([`host_state`](@ref)). Plain data only (`Vector`s of static vectors, numbers,
strings), for any file format (`JLD2`, `Serialization`).
"""
function checkpoint_state(pg::ParticleGrid{D, M, T}) where {D, M, T}
    N, c, m = pg.meta.N, pg.core, pg.meta
    return (format = CHECKPOINT_FORMAT, D = D, M = M, T = string(T), R = m.R, dx = Tuple(m.dx),
            N = N, N_interior = m.N_interior, N_ghost = m.N_ghost, boundary_time = m.boundary_time, sweep_time = m.sweep_time,
            positions = _host_copy(c.positions, N), volumes = _host_copy(c.volumes, N), is_boundary = _host_copy(c.is_boundary, N),
            tags = _host_copy(c.tags, N), particle_orders = _host_copy(c.particle_orders, N), normals = _host_copy(c.normals, N),
            wall_distances = _host_copy(c.wall_distances, N), wall_curvatures = _host_copy(c.wall_curvatures, N),
            _neighbour_lists(pg)..., rhos = _host_copy(pg.rhos, N))
end

# The neighbour lists particle by particle in slot order (what `with_slot_layout` preserves), from host copies
function _neighbour_lists(pg::ParticleGrid{D, M, T}) where {D, M, T}
    N, nb = pg.meta.N, pg.neighbor
    ranges = _host_copy(nb.ranges, N)
    S = maximum(r -> isempty(r) ? 0 : last(r), ranges; init = 0)      # the last slot in use (either layout)
    idx, w, dist = _host_copy(nb.indices, S), _host_copy(nb.weights, S), _host_copy(nb.distances, S)
    counts = Int[length(r) for r in ranges]
    n = sum(counts; init = 0)
    I, Wt, Dd = Vector{Int}(undef, n), Vector{T}(undef, n), Vector{Space{D, T}}(undef, n)
    k = 0
    @inbounds for i in 1:N, s in ranges[i]
        k += 1
        I[k], Wt[k], Dd[k] = idx[s], w[s], dist[s]
    end
    return (neighbor_counts = counts, neighbor_indices = I, neighbor_weights = Wt, neighbor_distances = Dd)
end

function checkpoint_state(pg::ParticleGrid{D, W, T}, bs::BlockState{W, T}) where {D, W, T}
    bs.N[] == pg.meta.N || throw(ArgumentError("checkpoint_state: the block state was set for $(bs.N[]) particles, the grid has $(pg.meta.N)"))
    return (; checkpoint_state(pg)..., nb = bs.nb, blocks = _host_copy(bs.data, bs.nb * bs.N[]))
end

function checkpoint_state(devs::BlockDevices)
    p1 = first(devs.parts)
    _activate(p1)
    return (; checkpoint_state(p1.pg)..., nb = num_blocks(devs.pde), blocks = host_state(devs))
end

"""
    restore_checkpoint!(pg::ParticleGrid, ck) -> pg
    restore_checkpoint!(pg::ParticleGrid, bs::BlockState, ck) -> pg
    restore_checkpoint!(devs::BlockDevices, ck) -> devs

Puts the state of a [`checkpoint_state`](@ref) back into a run built from the same configuration (the same equation,
grid parameters, scheme and stepper; the initial conditions may be set or not, they are overwritten): the particle set
and the states are copied (the grid grows if the checkpoint holds more particles), the cell list is rebuilt by a search
over the restored positions without wrapping them again (`pg.neighbor(pg; wrap = false)`), and the neighbour lists are
then replaced by the saved ones, in the grid's slot layout; every geometry cache is stale afterwards. The next step then continues the run that wrote the checkpoint bit for bit (same backend and
device model).

A split block state takes a checkpoint of any split or of a single block state (the blocks are distributed by the
parts' block ranges) and vice versa. Throws an `ArgumentError` if the checkpoint belongs to another kind of run: another
format, dimension, state length, float type, support radius or spacing, or a block state for a grid alone (or the
other way round).
"""
function restore_checkpoint!(pg::ParticleGrid{D, M, T}, ck) where {D, M, T}
    _check_checkpoint(pg, ck)
    hasproperty(ck, :blocks) && throw(ArgumentError("restore_checkpoint!: the checkpoint holds a block state; restore it with the block state (restore_checkpoint!(pg, bs, ck))"))
    _restore_grid!(pg, ck)
    copyto!(pg.rhos, 1, ck.rhos, 1, ck.N)
    return pg
end

function restore_checkpoint!(pg::ParticleGrid{D, W, T}, bs::BlockState{W, T}, ck) where {D, W, T}
    _check_checkpoint(pg, ck)
    first(bs.blocks) == 1 || throw(ArgumentError("restore_checkpoint!: the block state holds the blocks $(bs.blocks) of a split state; restore the BlockDevices"))
    _check_blocks(ck, bs.nb)
    _restore_grid!(pg, ck)
    copyto!(pg.rhos, 1, ck.rhos, 1, ck.N)
    _restore_blocks!(bs, ck)
    return pg
end

function restore_checkpoint!(devs::BlockDevices, ck)
    nb = num_blocks(devs.pde)
    for pt in devs.parts
        _check_checkpoint(pt.pg, ck)
    end
    _check_blocks(ck, nb)
    _foreach_part(devs.parts) do pt
        _restore_grid!(pt.pg, ck)
        copyto!(pt.pg.rhos, 1, ck.rhos, 1, ck.N)
        _restore_blocks!(pt.bs, ck)
    end
    return devs
end

function _check_checkpoint(pg::ParticleGrid{D, M, T}, ck) where {D, M, T}
    bad(what) = throw(ArgumentError("restore_checkpoint!: the checkpoint belongs to another run ($what)"))
    (hasproperty(ck, :format) && ck.format == CHECKPOINT_FORMAT) ||
        bad("format $(hasproperty(ck, :format) ? ck.format : "unknown"), this version reads format $CHECKPOINT_FORMAT")
    ck.D == D || bad("D = $(ck.D), the grid has D = $D")
    ck.M == M || bad("a state of $(ck.M) components, the grid's has $M")
    ck.T == string(T) || bad("float type $(ck.T), the grid's is $T")
    ck.R == pg.meta.R || bad("support radius $(ck.R), the grid's is $(pg.meta.R)")
    ck.dx == Tuple(pg.meta.dx) || bad("spacing $(ck.dx), the grid's is $(Tuple(pg.meta.dx))")
    n = ck.N
    all(a -> length(a) == n, (ck.positions, ck.volumes, ck.is_boundary, ck.tags, ck.particle_orders, ck.normals,
                              ck.wall_distances, ck.wall_curvatures, ck.rhos)) || bad("arrays of the wrong length for N = $n")
    ck.N_interior + ck.N_ghost == n || bad("N_interior + N_ghost ≠ N")
    (length(ck.neighbor_counts) == n && all(>=(0), ck.neighbor_counts) &&
     all(a -> length(a) == sum(ck.neighbor_counts; init = 0), (ck.neighbor_indices, ck.neighbor_weights, ck.neighbor_distances))) ||
        bad("neighbour lists of the wrong length")
    all(j -> 1 <= j <= n, ck.neighbor_indices) || bad("neighbour indices outside 1:N")
    return nothing
end

# `nb`: the number of blocks of the run (all of the equation's blocks)
function _check_blocks(ck, nb::Int)
    hasproperty(ck, :blocks) || throw(ArgumentError("restore_checkpoint!: the checkpoint holds no block state (it is the checkpoint of a grid alone)"))
    ck.nb == nb || throw(ArgumentError("restore_checkpoint!: the checkpoint holds $(ck.nb) blocks, the run has $nb"))
    length(ck.blocks) == ck.nb * ck.N || throw(ArgumentError("restore_checkpoint!: the checkpoint's blocks have the wrong length"))
    return nothing
end

# The particle set and its geometry, the cell list by a search over the restored positions (no second wrap), then the saved
# neighbour lists in the grid's slot layout
function _restore_grid!(pg::ParticleGrid, ck)
    N = ck.N
    ensure_capacity!(pg, N)
    c = pg.core
    for (dst, src) in ((c.positions, ck.positions), (c.volumes, ck.volumes), (c.is_boundary, ck.is_boundary), (c.tags, ck.tags),
                       (c.particle_orders, ck.particle_orders), (c.normals, ck.normals), (c.wall_distances, ck.wall_distances),
                       (c.wall_curvatures, ck.wall_curvatures))
        copyto!(dst, 1, src, 1, N)
    end
    m = pg.meta
    m.N, m.N_interior, m.N_ghost = N, ck.N_interior, ck.N_ghost
    m.boundary_time, m.sweep_time = ck.boundary_time, ck.sweep_time
    pg.neighbor(pg; wrap = false)
    _restore_neighbours!(pg, ck)
    return nothing
end

function _restore_neighbours!(pg::ParticleGrid{D, M, T}, ck) where {D, M, T}
    nd, N = pg.neighbor, ck.N
    R = eltype(nd.ranges)                                    # the slot layout of the grid
    counts = ck.neighbor_counts
    max_nb = maximum(counts; init = 0)
    cap = _slot_capacity(R, sum(counts; init = 0), max_nb, N)
    incl = cumsum(counts)
    ranges = R[_slot_range(R, i, counts[i], incl[i], N) for i in 1:N]
    push!(ranges, _slot_sentinel(R, cap, N))
    I, Wt, Dd = zeros(Int, cap), zeros(T, cap), zeros(Space{D, T}, cap)     # (the padding of the slot-major layout: zeros)
    k = 0
    @inbounds for i in 1:N, s in ranges[i]
        k += 1
        I[s], Wt[s], Dd[s] = ck.neighbor_indices[k], ck.neighbor_weights[k], ck.neighbor_distances[k]
    end
    ensure_capacity!(nd, cap)
    copyto!(nd.ranges, 1, ranges, 1, N + 1)
    copyto!(nd.indices, 1, I, 1, cap)
    copyto!(nd.weights, 1, Wt, 1, cap)
    copyto!(nd.distances, 1, Dd, 1, cap)
    copyto!(nd.counts, 1, counts, 1, N)
    pg.meta.max_nb = max_nb
    pg.meta.geometry_version += 1
    return nothing
end

# The blocks a state holds (all, or the range of a part), from the block-major array of all blocks
function _restore_blocks!(bs::BlockState, ck)
    N = ck.N
    bs.N[] = N
    ensure_capacity!(bs.data, bs.nb * N)
    copyto!(parent(bs.data), 1, ck.blocks, (first(bs.blocks) - 1) * N + 1, bs.nb * N)
    return nothing
end
