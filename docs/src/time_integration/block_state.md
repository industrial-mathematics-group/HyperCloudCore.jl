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

Scope of this first version: explicit tableaus, static grids, no MOOD, no source terms; the boundary
conditions of the grid apply per block. Checked against the standard pipeline: a one-block Burgers run is
bit-identical, a discrete transport with four blocks of two velocities reproduces the eight-component run
to round-off (bit for bit with `MinmodLimiter`; the Venkatakrishnan limiter's tolerance depends on the
state vector as a whole) and converges to the exact solution at second order, in 1-D and 2-D, with inflow
and outflow conditions, on the Polyester and the KernelAbstractions CPU backends.

What comes next, in this order: source terms in two passes (a reduction over the blocks per particle, e.g.
the moments and the parameters of the Maxwellians, then the update per block), the MOOD admissibility check
as a reduction, moving frames (the frame velocity from the reduction), kinetic walls that read other blocks
of the same ghost, adaptivity and reordering of all blocks together, and the velocity decomposition across
devices (the only exchange per stage is the reduction).

## API

```@docs
num_blocks
block_flux
block_max_eigenvalue
BlockOf
BlockState
block_view
set_block_states!
BlockTimeStepper
HyperCloudCore.get_time_step(::ParticleGrid{D, W, T}, ::HyperbolicPDE{D, W, T}, ::Any, ::BlockState{W, T}) where {D, W, T}
```
