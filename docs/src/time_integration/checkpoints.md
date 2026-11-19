# Checkpoints

A run on a shared cluster can be stopped at any time (preemption by a job of higher priority, the time limit
of the partition). [`checkpoint_state`](@ref) takes the state of a run at the end of a step as plain host
arrays, and [`restore_checkpoint!`](@ref) puts it back into the same run rebuilt from its configuration: the
next step continues the run that wrote the checkpoint **bit for bit** (on the same backend and device model; a
GPU of another model may round differently). Writing the file, how often, and resuming the time loop are the
driver's business (HyperCloud's `solve_equation` with a `Checkpointing`); this package only defines what the
state is.

## What is saved and what is rebuilt

A step starts from the states and the particle set and recomputes everything else, so that is what a checkpoint
holds:

| saved | why |
|:------|:----|
| `rhos`, the block state (`blocks`, block-major, all blocks) | the solution |
| `positions` in memory order, `N`, `N_interior`, `N_ghost` | movers, adaptivity and the sweep of moving pieces change the particle set and its order |
| `volumes` | re-estimated, remapped (content form, adaptivity, sweeps): not recomputable from the positions alone |
| `is_boundary`, `tags`, `normals`, `wall_distances`, `wall_curvatures` | change with the particle set (inserted, removed and swept particles, ghosts of moving pieces) |
| `particle_orders` | the orders of the last step (a step resets them at its start; kept so the restored grid is the grid at the end of the step) |
| `boundary_time`, `sweep_time` | the time the moving pieces' ghosts and the swept region belong to (`_sync_boundaries!` compares them with the step's time) |

Rebuilt on restore: the neighbour lists, distances and weights (a search over the restored positions, without
wrapping them into the periodic box again), and through `geometry_version` every cache derived from them (the
MLS moment matrices, the slot weights, the ghost geometry of the block pipeline). The stepper's stage buffers and
MOOD masks are set at the start of every step and are not part of the state. The checkpoint also records `D`,
`M`, the float type, the support radius and the spacing: a checkpoint of another kind of run is refused.

The neighbour search is deterministic (the particles of every bin are sorted), the per-particle sums run in a
fixed order, and the partial sums of a split block state are added on the host in block order, which is why a
restart is exact rather than close.

## Usage

```julia
ck = checkpoint_state(pg)                 # or checkpoint_state(pg, bs), checkpoint_state(devs)
# ... save `ck` (a NamedTuple of Vectors and numbers) with JLD2 or Serialization; the run is stopped ...
# ... later: build the run from the same configuration as before ...
restore_checkpoint!(pg, ck)               # or restore_checkpoint!(pg, bs, ck), restore_checkpoint!(devs, ck)
```

A split block state ([`BlockDevices`](@ref)) is saved as one block state (the grid of its first part, every part
holds the same cloud, and the blocks of all parts), so a checkpoint moves between a single block state and splits
over any number of parts.

```@docs
checkpoint_state
restore_checkpoint!
```
