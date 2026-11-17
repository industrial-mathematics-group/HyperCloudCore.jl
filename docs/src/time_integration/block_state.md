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
(`mood_flags`, `mood_inadmissible`) are the same. `MOODu2` is not available yet: its curvature test reads the
scheme's gradient cache, which holds the last block's gradients.

Scope of this version: static grids; the boundary conditions of the grid apply per block. Checked against the standard pipeline: a one-block Burgers run is
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

What comes next, in this order: moving frames (the frame velocity from the reduction), kinetic walls that read other blocks
of the same ghost, adaptivity and reordering of all blocks together, and the velocity decomposition across
devices (the only exchange per stage is the reduction).

## API

```@docs
num_blocks
block_flux
block_max_eigenvalue
block_admissible
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
HyperCloudCore.get_time_step(::ParticleGrid{D, W, T}, ::HyperbolicPDE{D, W, T}, ::Any, ::BlockState{W, T}) where {D, W, T}
```
