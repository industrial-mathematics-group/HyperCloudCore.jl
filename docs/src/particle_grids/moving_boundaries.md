# Moving Boundaries

A piece of the domain's shape can move: `Moving(shape, motion)` turns a shape into a rigid body whose pose at time $t$ is `motion(t)::RigidMotion` (translation $c$, rotation $R$, velocity $v$ and angular velocity $\Omega$; `translating(v)` and `rotating(ω; center)` build the common motions, any function returning a `RigidMotion` works). The region is still described by one shape, for example a channel with a rotating plate:

```julia
plate = Moving(Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4), rotating(1.0))
geom  = GeometricDomain(Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate;
                        is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                        bc_map = Dict(3 => wall, 4 => wall, 5 => wall, 6 => wall, 7 => wall, 8 => wall))
```

`at_time(shape, t)` is the shape with every moving piece at its pose at `t` (a static shape); `wall_velocity(shape, x, t)` the velocity of the piece nearest to `x`. The moving pieces are looked up by **tag**: every tag of a moving piece must belong to that piece alone (give it its own tags with `Tagged`), and the grid records the piece of each tag when the particles are generated.

## How the cloud follows

The interior cloud stays where it is (Eulerian); only the ghost particles of a moving piece move, and the interior particles make way:

1. **Ghosts.** The ghost particles generated inside a piece at $t = 0$ are placed at the piece's pose at every stage time and at the end of every step (`place_ghosts!`): an exact rigid displacement of their positions and normals from the step start, so they never drift from the body. The time stepper runs its moving-grid path for this (position stages, a neighbour search per step) with grid velocity zero for the interior particles; a `GridMover` can move the interior as well.
2. **Boundary conditions** see the ghosts at the time of the state they build. The stored positions and normals belong to `pg.meta.boundary_time`; the fit-based conditions displace them to the stage time themselves (`tag_motion`), and the `WallPoint` their operator receives carries the wall velocity at the wall point. An operator for a moving wall reflects the velocity relative to the wall, `u_rel = u - w.velocity`.
3. **Sweep** (`sweep_boundaries!`, after every step), in two steps. First the **region**: on a lattice with two points per spacing, the points a piece has covered since the last sweep leave the interior particles that shared them, volume and content alike (the content there is the limited linear reconstruction of the nearest interior particle), and the points it has uncovered join them with the reconstructed state. The particle volumes follow the region exactly up to the quadrature (the interior volumes of a channel with a translating plate stay at the region's volume to 6 digits over 200 steps), the content of what was covered is gone and what was uncovered appears with the local state: a **consistent interpolation**, which is what a collocation scheme needs at a moving wall. The transport of the swept fluid is the equations' business, through the wall velocity in the conditions; enforcing the old content on the new region instead (a conservative remap) flattened the solution around every insertion in a growing region. Then the **topology**: interior particles inside a moving piece or closer than a quarter spacing to it are removed, lattice points of the canvas at least half a spacing inside the region and half a spacing from every remaining particle are inserted, each through the conservative remap of the [adaptivity](@ref "Particle Volumes and Adaptivity") on the new region (a removed particle's remaining share and content go to its neighbours, an inserted one takes its share from them). `sweep_boundaries!` returns the counts and the content change of the call.

Keep the step small enough that a piece moves less than the clearance per step, $|w|\,\Delta t \le \Delta x/4$ (the sweep warns once when a particle was covered before it was removed). The time step estimate does not know the wall speed. Points that the point generator would snap onto a moving piece become ghosts half a spacing outside instead (a particle on a moving wall would be covered as soon as it moves).

## What was measured

Scalar advection with velocity $(1, 0)$ in a channel periodic in $x$ and a plate $0.4 \times 0.1$ translating at $(1, 0)$: the fluid is at rest relative to the plate and the exact solution translates. With `MirrorBC(degree = 1)` on the plate and the walls the maximum error at $t = 0.5$ is 0.22 / 0.065 / 0.011 for $N = 20 / 40 / 80$ (order 1.7–2.5; away from the plate 0.081 / 0.018 / 0.005), the solution stays within the initial range, the interior volumes sum to the region's volume (1.96) to six digits, the content changes by 0.03 / 0.004 / 0.0007, and the plate's ghosts sit on the exactly translated lattice. The same in 1-D (a piston receding at the flow speed, the fluid at rest relative to it): 0.19 / 0.060 / 0.014 for $N = 50 / 100 / 200$, the interior volumes sum to the tube's length exactly, and the content grows by the inflow (0.51 / 0.502 / 0.501 against 0.5). A piston advancing into a fluid at rest removes the covered particles with their content, keeps the first particle between a quarter and one spacing from the face and the volumes at the tube's length. A plate rotating in a fluid at rest keeps the scalar within its initial range and the volumes constant (the sweep only interpolates; the peak the plate sweeps through erodes). Ghost positions after 200 steps of a rotation agree with the exact rotation to $2 \cdot 10^{-15}$.

## Limits

- Rigid motions prescribed in time; nothing is coupled back (no fluid–structure interaction).
- The ghost band of a piece is the one generated at $t = 0$ inside it: a thin piece has few ghost rows, and a piece that approaches a wall or another piece keeps its own normals and wall distances (the composite geometry is not re-evaluated for the ghosts).
- The interior particles' stored wall distances and normals are not updated as a piece moves; only the conditions on the ghosts read them.
- The volume of the region at $t = 0$ (`domain_volume`) is what the volume estimators renormalise to; the stepper does not re-estimate volumes while only pieces move (`volume_estimator` acts on moving interiors), so the sweep's volumes stand.
- `sweep_boundaries!` is a host pass (a device grid is swept on a host copy), like the adaptivity.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["MovingBoundaries.jl"]
Private = false
```
