# Block States

The state of a particle is an `SVector` of `M` components, and `M` is a type parameter of every struct of
the package: operations on a state unroll at compile time, which is what makes small systems fast and
allocation free, and what makes large ones impossible. A discrete-velocity model with two species of
``30^2`` velocities and the Chu reduction has 3600 components per particle: the compile time of the schemes
grows faster than ``M``, the per-slot flux buffers hold ``(D + 1) M`` values per neighbour, and a kernel
holding a 3600-wide `SVector` spills every register.

The **block pipeline** keeps the `SVector` pipeline as it is and composes it. It applies to **block
equations**: systems whose transport is block diagonal, so that the components couple through source terms
and per-particle reductions only. Every discrete-velocity model is one (each velocity is transported on its
own, the collision couples them through the moments); a system whose flux Jacobian couples its components
densely is not, and such systems are not the large ones in practice.

## Block equations

A block equation is a `HyperbolicPDE{D, W, T}` whose state type is one block of width ``W``, with
``n_b`` blocks in total. It implements the three functions of the contract: `num_blocks`, `block_flux` and
`block_max_eigenvalue`. Every `HyperbolicPDE` is a block equation with one block (the defaults forward to
`flux` and `max_eigenvalue`), so the block pipeline reproduces the standard one for any equation,
bit for bit.

[`BlockOf`](@ref)`(eq, b)` is block `b` as an ordinary PDE: `flux` and `max_eigenvalue` forward to the block
functions with the block index as a run-time field, so a single compiled kernel serves every block.

## The block state and the stepper

[`BlockState`](@ref) holds the ``n_b`` blocks of all particles in one array of length ``n_b N`` in
block-major order (block ``b`` of particle ``i`` at index ``(b - 1) N + i``): every block is a contiguous
vector over the particles, the layout devices want. The width ``W`` is a compile-time constant, the block
count a run-time number.

[`BlockTimeStepper`](@ref) owns a `GeneralRKTimeStepper` for ``W`` components and runs its stage kernels
block by block on the views of the block state, with the equation wrapped as `BlockOf(eq, b)`: the grid,
the neighbour search, the MLS weights and the scheme's pre-pass are shared by all blocks. The stage
functions of `GeneralRKTimeStepper` take their state arrays as arguments for that reason (the boundary
conditions receive the reference state of the step start the same way, `apply_boundary_conditions!`);
the stepper's own step advances any state array, `pg.rhos` by default.

```julia
eq  = MyDVM(...)                                   # a HyperbolicPDE{D, W, T} with num_blocks(eq) = nb
pg  = ParticleGrid(geom, dxs, weight, W)           # W components per particle
bs  = BlockState(eq, pg)
set_block_states!((x, b) -> initial_block(x, b), bs, pg)
ts  = BlockTimeStepper(eq, MUSCL(T, D, W, 2, MinmodLimiter(:hard), RusanovFlux()), RK3_SSP_Tableau(T))
dt  = 0.3 * get_time_step(pg, eq, ts.inner.divergence_interpolator, bs)
ts(eq, pg, bs, t, dt)
f_b = block_view(bs, b)                            # the states of block b
```

## Source terms in two passes

A source of a block equation needs the whole distribution of a particle (the moments) but updates every
block on its own. The contract of [`AbstractBlockSource`](@ref) splits it accordingly: `block_reduce` turns
the blocks of a particle into a small bits value of parameters (for a BGK model the moments and the
exponential parameters of the Maxwellian, for the mixture the species' velocities and temperatures after
the stage), stored per particle in a [`BlockSourceTerm`](@ref); `block_source` and `block_implicit_solve`
then update one block from those parameters. The stepper runs the reduction once per stage and source
(with `dt_coeff = Δt a_ii` for an implicit solve, `0` for an explicit evaluation) and the per-block pass
after it. Implicit terms need an IMEX tableau; at its stages without an implicit diagonal they are
evaluated explicitly, as in the standard stepper, and the right-hand side of a solved stage is
``(U_{out} - U_{in}) / (\Delta t\, a_{ii})``. Several implicit terms are applied one after another (their
increments add up).

## MOOD on blocks

MOOD works as in the standard stepper, with the particle's order shared by all its blocks: at every
checkpoint the flag pass runs once per block (the criterion on the block's components against the block's
previous stage, the per-block admissibility that every block is finite), [`block_admissible`](@ref) adds the
cross-block check (for a discrete-velocity model the positivity of the moments), and one drop pass lowers the
order of every flagged particle; the previous stage's right-hand side is then recomputed on the affected
particles for all blocks (the explicit sources reuse their parameters there) and the candidates are re-formed,
until nothing is flagged. The Euler substeps of SSP tableaus, the forcing halos and the diagnostics
(`mood_flags`, `mood_inadmissible`) are the same. `MOODu2`'s curvature test reads the scheme's gradient cache,
which holds the last block's gradients, so the block's gradients at the previous stage are gathered before its
flag pass (one pre-pass per block and checkpoint).

## Moving grids on blocks

On a moving grid the frame velocity of every particle is reduced before each flux pass: a prescribed mover gives
it as usual, the Lagrangian movers through [`block_fluid_velocity`](@ref) (the mean velocity of the moments of a
discrete-velocity model). The flux pass then runs in the frozen-velocity ALE frame of the particle, `F − w_i U`
with the wave speed bound [`block_max_eigenvalue`](@ref)`(eq, U, d, b, w_i)` (the exact `|λ − w|` of a
block's velocities makes the dissipation vanish where the block moves with the frame), the positions are stage
variables as in the standard stepper and the volumes are re-estimated at the end of the step. The conservative
Lagrangian frame of the standard stepper, which evaluates the frame velocity of every interface state, has no
block form: a block cannot give the fluid velocity. Moving boundary pieces are not available yet.

## Walls on blocks

The boundary conditions of the grid apply block by block with the block's state of the step start as the
reference. A `MirrorBC` whose operator needs the whole distribution of a ghost (a diffuse kinetic wall: the
density that returns the incident flux, summed over all velocities; a specular wall: a velocity reflected into
another block) implements the wall in two passes, [`block_wall_reduce`](@ref) and [`block_wall_apply`](@ref)
with the parameter type [`wall_reduction_type`](@ref): the mirrored interior state of every block is fitted into
a scratch array first (the even reflection), the reduction runs once per ghost over the mirrored blocks, and the
application gives every block's ghost state from its mirrored value, the parameters and, where it needs them,
the other mirrored blocks of the ghost. An operator without a reduction acts block by block.

## Adaptivity, reordering and moving pieces on blocks

The host routines that change the particle set take further state arrays and treat them exactly like the grid's
own state (the geometric decisions are shared, the remap of every array is that of the single state, the
compaction and the new particles' states follow): `adapt_particles!(pg, ad, order; states)`,
`sweep_boundaries!(pg, t_old, t, order; states)` and the reordering `pg.reorder(pg; extra)`. The block
pipeline uses them with its blocks: [`adapt_particles!`](@ref)`(pg, ad, order, bs)` between steps (the blocks
are copied into per-block vectors around an event, since the block-major layout depends on the particle count,
and laid out again afterwards; nothing is copied when no pair violates the spacing rules), the moving pieces
inside the step (the ghosts placed at every stage, the sweep at the end with every block remapped), and
[`reorder!`](@ref)`(pg, bs)`. Checked: with a compressing prescribed mover and the adaptivity, with a piston
sweeping a tube, the block runs reproduce the standard runs bit for bit, particle set included.

## Devices and the velocity decomposition

The block pipeline runs on any storage: the block state, the stepper's buffers, the parameter arrays of the sources
and of the walls are `Growable`s of the `storage` array type (the stepper moves the parameter arrays of its sources to
the storage of its stage buffers), the kernels are those of the standard stepper, and the host-side passes (the
initial conditions, the snapshots, the adaptivity and the sweep of moving pieces) copy blocks between the device and
the host, the last two only when an event changes the particle set. [`memory_estimate`](@ref) and
[`reserve!`](@ref) take the block stepper and the block state (`reserve!(pg, ts, bs, scheme; limit)`): the state
arrays count `nb` elements per particle. Checked on the KernelAbstractions CPU backend (PoCL) and with CUDA.

A state whose blocks are split over several devices exchanges, per stage, only the per-particle reductions: for
that the sources give their reduction in the **additive form** of [`reduction_partial_type`](@ref),
[`block_partial`](@ref) (the contribution of a block range, which adds up over ranges: the moments of those
velocities) and [`finalize_reduction`](@ref) (the parameters from the sum); a source that gives it gets
`block_reduce` for free, so the single-device stepper runs the same arithmetic. The orchestration across devices
(one set of stage buffers per device over its block range, the sum of the partials per particle, the same split
for the frame velocity, the MOOD flags and the wall reductions) is not part of this version.

Scope of this version: host and single-device storage. Checked against the standard pipeline: a one-block Burgers run is
bit-identical, a discrete transport with four blocks of two velocities reproduces the eight-component run
to round-off (bit for bit with `MinmodLimiter`; the Venkatakrishnan limiter's tolerance depends on the
state vector as a whole) and converges to the exact solution at second order, in 1-D and 2-D, with inflow
and outflow conditions, on the Polyester and the KernelAbstractions CPU backends. With a source that needs
the reduction (every component relaxing towards the particle's mean, the structure of the BGK collision)
the block run agrees with the standard stepper and the equivalent `AbstractImplicitSourceTerm` to round-off
(the stage states and implicit right-hand sides bit for bit, the flux pass to an ulp), with ARS222 (also in the
stiff limit) and with an explicit tableau and source. With MOOD (`MOODu1` on a box per component with the unlimited
third-order scheme, the Euler substeps of RK3, the forcing halo, ARS222 with a source, `OnlyMOOD`, and the
cross-block admissibility of a guarded equation) the same particles drop their orders as in the standard run
and the states agree to round-off.

Checked: a prescribed mover gives the standard run bit for bit (positions and volumes included); a Lagrangian
mover with one common velocity transports every state exactly; a reflecting wall operator that reads the mirror
partner's block reproduces the permutation operator of the standard run to round-off.

What comes next: the orchestration of a block state split over several devices (above), and a conservative
Lagrangian frame on blocks, which needs the frame velocity of a pair from the particles' velocities (a design
question: `w_ij = ½ (w_i + w_j)` from the reduction instead of the interface states' velocities).

## API

```@docs
num_blocks
block_flux
block_max_eigenvalue
block_admissible
block_velocity
block_is_admissible
block_nc_jump
block_fluid_velocity
wall_reduction_type
block_wall_reduce
block_wall_apply
reorder!
HyperCloudCore.adapt_particles!(::ParticleGrid{D, W, T}, ::ParticleAdaptivity{T}, ::Int, ::BlockState{W, T}) where {D, W, T}
BlockOf
BlockState
block_view
set_block_states!
BlockTimeStepper
AbstractBlockSource
AbstractExplicitBlockSource
AbstractImplicitBlockSource
BlockSourceTerm
reduction_type
block_reduce
block_source
block_implicit_solve
reduction_partial_type
block_partial
finalize_reduction
HyperCloudCore.get_time_step(::ParticleGrid{D, W, T}, ::HyperbolicPDE{D, W, T}, ::Any, ::BlockState{W, T}) where {D, W, T}
```
