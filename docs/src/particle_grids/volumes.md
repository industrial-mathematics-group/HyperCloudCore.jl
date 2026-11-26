# Particle Volumes and Adaptivity

Collocation particles carry no mass. To define a discrete conserved quantity, and to make inserting and merging particles exactly conservative, every interior particle gets a positive weight $V_i$ (stored in `pg.core.volumes`; the ghost particles outside $\Omega$ get none) with

```math
\sum_i V_i = |\Omega|, \qquad \sum_i V_i\, f(x_i) \approx \int_\Omega f\,\mathrm dx .
```

No Voronoi tessellation is built or stored. $V_i = \int_\Omega \phi_i\,\mathrm dx$ for shape functions $\phi_i$ that form a partition of unity ($\sum_i \phi_i \equiv 1$), evaluated with a quadrature on the point cloud itself or on a throwaway lattice. Only the scalar $V_i$ is kept.

## Volume estimators

| Estimator | Shape functions | Evaluation | Quadrature order | Cost |
|---|---|---|---|---|
| `ShepardVolumes(; sweeps = 1)` | $\phi_i(x) = w(\lvert x - x_i\rvert) / \sum_k w(\lvert x - x_k\rvert)$ | particle-based, no lattice | exact ($\Delta x^D$) on a uniform periodic lattice; not convergent on jittered or stretched clouds; 1 on bounded boxes | one neighbour sweep per `sweep` (kernel) |
| `MLSVolumes(P; samples_per_dx = 2)` | MLS reproducing-kernel functions of degree $P$ | background lattice over $\Omega$ | about $P + 1$ on periodic clouds for $P \ge 1$ (also jittered); on bounded boxes 2 for every $P$ in 1-D and 2-D (the lattice's midpoint rule) | lattice points × neighbours (serial, host) |

The orders are those of $\sum_i V_i f(x_i) \to \int_\Omega f$ over the interior particles, measured with $R = 2.5\Delta x$ and a smooth, non-harmonic $f$ (1-D: $N = 40 \dots 320$; 2-D: $N = 20 \dots 160$; for a harmonic $f$ such as $e^x \cos y$ the leading term of the midpoint rule, proportional to $\int \Delta f$, vanishes and the errors look one to two orders smaller). On a periodic cloud there are no ghosts. On a bounded region only the interior particles carry volume: the final renormalisation makes them sum to $\lvert\Omega\rvert$ exactly, the ghosts get $V = 0$, and MLS forms its partition of unity from the interior particles only, so its reproduction holds up to the wall; what is left is the lattice's midpoint rule, second order for every $P$. Shepard's particle-based density gives a particle on the wall a whole cell, hence its first order there (relative errors of about $0.1$–$1\,h$, depending on $f$ and the domain: $1.1\,h$ for $x^2$ on $[-1, 1]$).

Relative errors on the unit interval / square ($R = 2.5\Delta x$, $N = 40$ in 1-D, $N = 20 \to 40 \to 80$ in 2-D):

| | Shepard | MLS0 | MLS1 | MLS2 |
|---|---|---|---|---|
| 1-D, $x^2$ / $e^x$ | $1.4\cdot 10^{-2}$ / $2.2\cdot 10^{-3}$ | – | – | $3.9\cdot 10^{-5}$ / $6.5\cdot 10^{-6}$ |
| 2-D, $e^{x + y}$ | $8.7 \to 4.4 \to 2.2 \cdot 10^{-3}$ | $6.3 \to 1.6 \to 0.38 \cdot 10^{-4}$ | $3.0 \to 0.79 \to 0.20 \cdot 10^{-3}$ | $5.1 \to 1.3 \to 0.33 \cdot 10^{-5}$ |

(1-D MLS0/MLS1 not measured.) (Up to 0.8 the ghosts got volumes too and the renormalisation ran over all particles, so the interior volumes summed to less than $\lvert\Omega\rvert$ by the ghosts' share: 4 % in 1-D at $h = 1/160$, 30–40 % on a 2-D box at $N = 20$, and the discrete mass included the ghost content.) Shepard on a jittered periodic cloud: error about $2\cdot 10^{-3}$ at 30 % jitter for every $N$; MLS$(P \ge 1)$ converges there at order 2–4.

**Shepard** starts from the kernel-density volume $V_i \propto 1/S_i$, $S_i = w(0) + \sum_j w_{ij}$, applies `sweeps` partition-of-unity sweeps $V_i \leftarrow \sum_j (w_{ij}/S_j)\,V_j$ (which preserve $\sum V$), zeroes the ghosts and renormalises the interior to $|\Omega|$. On a uniform periodic cloud it reproduces $V_i = \Delta x^D$ exactly. It needs a kernel that is bounded at zero distance; with `InverseWeightFunction` the volumes fall back to $|\Omega|/N$ with a warning.

$S_i$ is a lattice sum over a fixed number of neighbours, which is not proportional to the local particle density when the spacing varies. On a moving (stretched) cloud the Shepard quadrature therefore has a bias that depends on the local stretch and does **not** shrink with the resolution: for a smooth profile on a 1-D grid moved by a sinusoidal velocity of amplitude 0.5, $\sum_i V_i u(x_i)$ of the *exact* solution is off by $-2\cdot 10^{-3}$ at every $N$ from 200 to 1600, while MLS volumes of degree 2 converge ($-2\cdot 10^{-5} \to 5\cdot 10^{-9}$). Mass errors measured with Shepard volumes on a moving grid contain that bias.

**MLS** evaluates, at every lattice point $y$, the shape functions $\phi_i(y) = w_i\, p_i^{\mathsf T} a$ with $\big(\sum_k w_k p_k p_k^{\mathsf T}\big) a = e_1$ and the full basis $p_k = (1, \text{monomials}(x_k - y))$. Because $\sum_i \phi_i(y)\, q(x_i) = q(y)$ for polynomials $q$ of degree $\le P$, the volumes integrate such polynomials exactly (up to the lattice's midpoint error). `P = 0` is the lattice-integrated Shepard partition. Shape functions of degree $\ge 1$ can be negative; a volume that comes out negative (rare without ghosts in the partition) is clamped before the final renormalisation. The lattice sweep is a serial host loop: as the per-step re-estimation of a moving grid it costs about 3 (1-D, 1600 particles) to 21 (2-D, $141^2$) explicit MUSCL2/RK2 steps on 8 threads, so it is not the default. On a device grid it runs on a host copy (only the volumes are copied back).

```julia
pg = ParticleGrid(geom, dx, wf, M; volume_estimator = MLSVolumes(2))   # at construction
update_volumes!(pg, ShepardVolumes())                                   # any later time
total_mass(pg)                                                          # Σ V_i U_i
domain_volume(pg)                                                       # |Ω|
```

`domain_volume` is analytic for the built-in rectangular and spherical domains; define a method for a custom geometry label.

## Volumes on moving grids

After every step of a moving grid, the time stepper recomputes the volumes from the new geometry with its `volume_estimator` (default `ShepardVolumes()`; `nothing` leaves them untouched). With the config-driven builders the estimator of the `Grid` namespace (`:volumes`) is used for the initial volumes *and* for every re-estimation ([`build_particle_grid`](@ref) stores it in `context[:VolumeEstimator]`, which [`build_timestepper`](@ref) and [`build_adaptivity`](@ref) read). Recomputation keeps the volumes consistent with the current cloud at the cost of one neighbour sweep, and never drifts. The content form of the Lagrangian movers with MUSCL ([moving grids](../time_integration/moving_grids.md)) does not re-estimate: its volumes are integrated from the scheme's own volume rate $\dot V_i = V_i\,(\nabla\cdot w)_i$ (the geometric conservation law, smooth in time, with $\sum V_i U_i$ exactly conserved), and $\sum_i V_i$ drifts from $\lvert\Omega\rvert$ by the discretisation error of the divergence; the events of the particle adaptivity re-estimate the volumes they touch.

The collocation scheme itself is not exactly conservative, with or without motion: $\sum_i V_i U_i$ changes by the scheme's truncation error per step. What the volumes buy is a *well-defined* discrete mass, so that topology changes can preserve it exactly.

## Particle insertion and merging

On Lagrangian grids particles cluster in compressions and thin out in expansions. `ParticleAdaptivity` repairs that between steps:

```julia
ad = ParticleAdaptivity(Float64; max_gap = 1.6, min_gap = 0.5, clearance = 0.7,   # (the defaults in 1-D and 2-D)
                        samples_per_dx = 2, exclude_reduced_order = true,
                        max_events = typemax(Int), max_rounds = 8,
                        volume_estimator = ts.volume_estimator)   # moving grid: the stepper's estimator
report = adapt_particles!(pg, ad, _extract_order(interp))   # between time steps
report.inserted, report.removed, report.mass_residual         # residual at round-off
report.escaped, report.held, report.seeded, report.refused    # open boundaries, events left out (below)
```

### Detection

Lengths are in units of $\Delta x = \min_d \Delta x_d$.

- **Insertion:** a neighbour pair farther apart than `max_gap` whose midpoint has no particle within `clearance` gets a new particle at the midpoint. The clearance test is what prevents insertions on a regular lattice, where second neighbours are $2\Delta x$ apart but the midpoint is occupied. The defaults depend on the dimension: `max_gap` 1.6 and `clearance` 0.7 in 1-D and 2-D, $1.6\sqrt{D/2}$ and $0.7\sqrt{D/2}$ above (1.96 and 0.86 in 3-D), because the longest diagonal of a lattice cell, $\sqrt D\,\Delta x$, must stay below `max_gap` or its midpoint (the cell centre, $\tfrac{\sqrt D}{2}\Delta x$ from the corners) within the clearance: with the 2-D values a 3-D lattice got a particle at every cell centre.
- **Merge:** a neighbour pair closer than `min_gap` is replaced by one particle at its midpoint.
- At most one event per particle neighbourhood and per round, and at most `max_events` per round. Event centres are kept $(R + \Delta x/2)\sqrt D + \tfrac32 R + \tfrac32 \Delta x$ apart, so that no event's quadrature sees a particle that another event of the round adds or removes: the events of a round are remapped one after the other on the old topology. Their local sets (the particles a remap writes) may overlap: every remap is exactly conservative on its own set, runs on the values the earlier ones wrote, and knows the particles the others remove, which are no part of its balance. (Up to 0.9.9 the whole local sets were kept apart, $R\,(2 + 2\sqrt D) + \Delta x$: 13 instead of 9.5 spacings in 2-D at $R = 2.5\,\Delta x$, i.e. half the events per round. With the still older spacing of $4R$ and no knowledge of the other events' removals, calls with several events lost up to $10^{-7}$ of the mass.)
- **Rounds:** a call repeats detection, remaps, topology change and neighbour search until no pair violates the rules, at most `max_rounds` times (default 8; the volumes are re-estimated once, at the end). Between the rounds the neighbour lists are repaired, not searched again: a particle that neither lost nor gained a neighbour keeps its list (renumbered), and the lists of the particles next to the events are written again from the rebuilt cell list, entry for entry those of a full search; one full search at the end of the call restores the compact slot layout. A round is therefore cheap, and a smaller limit saves little while it can starve an expanding region: on a 3-D blast at 64³, 3 instead of 8 rounds took a call from 0.32 to 0.26 time steps and the fewest neighbours of a particle from 37 to 7. The spacing above allows a few events per $(9.5\,\Delta x)^2$ and round in 2-D at $R = 2.5\,\Delta x$, and a pair is only found while it is a neighbour pair: a gap between `max_gap` and $R$ that is not served in time grows beyond $R$ and is never filled. With one round per step (up to 0.9.9) the centre of a Sedov blast at $R = 2.5\,\Delta x$ lost its stencils (2 neighbours) and the run blew up at $t = 0.1$; with rounds it keeps 14 and completes.
- **Not outside the region:** no event is centred outside the region (the midpoint of a pair can lie there: it would be an interior particle without a volume, with the state 0/0), and an event that cannot be remapped is left out (`report.refused`): one next to a particle without a usable volume, or whose new particle gets no share of the region. Such events used to stop the run with "Remap produced no volume".
- **Shock band:** with `exclude_reduced_order`, no event happens within $4R$ of a particle whose MOOD order was lowered during the last step. Interpolation of any order is unreliable across a discontinuity, and a Lagrangian grid in a shock region is better left alone (or driven towards Eulerian motion by the mover).

### Open boundaries

On a moving cloud (a grid mover) the call first keeps the cloud inside the region and up to its boundary:

- **Escaped particles.** An interior particle that has left the region is removed with its content, the outflow (`report.escaped`, `report.escaped_content`). Up to 0.9.9 such particles stayed outside (37 % of the cloud of the 2-D Riemann problem C6 by $t = 0.3$), kept interacting with the ghosts and with each other, and got no volume from an estimator. A particle that crossed a wall is put back onto the wall instead (`report.held`): a `MirrorBC` of a shape-based domain, or any condition type that declares [`is_wall`](@ref). Only particles with a ghost among their neighbours (or with none) are tested, so the test costs one kernel on a cloud that stays inside.
- **Seeds.** Where an interior particle and a ghost among its neighbours are farther apart than `max_gap`, with the midpoint inside the region and free within `clearance`, a new interior particle is put at the midpoint (`report.seeded`): the cloud receded from the boundary, an inflow. Its state is the mean of the pair's (the ghost's is the boundary datum), its volume the estimator's (the nominal $\prod_d \Delta x_d$ without one). A seed is not a remap event: the gap is volume that no particle holds, and its content enters the region (`report.seeded_content`). A whole row is seeded in one call.

What leaves and what the seeds bring is not part of `mass_residual`: the interior content changes by `seeded_content - escaped_content + mass_residual` over the call, with the residual at round-off.

### The conservative remap

A topology change replaces the partition of unity $\{\phi_i^{\text{old}}\}$ by $\{\phi_i^{\text{new}}\}$. Each particle's volume and content change by the *swept* amounts

```math
\delta V_i = \int \big(\phi_i^{\text{new}} - \phi_i^{\text{old}}\big)\,\mathrm dx, \qquad
\delta m_i = \int \big(\phi_i^{\text{new}} - \phi_i^{\text{old}}\big)\,\tilde U\,\mathrm dx ,
```

where $\tilde U$ is the limited (bound-clipped) linear reconstruction of the *old* solution around the nearest old particle. New particles have $\phi^{\text{old}} = 0$, removed particles $\phi^{\text{new}} = 0$. Since both families sum to one pointwise, $\sum_i \delta V_i = 0$ and $\sum_i \delta m_i = 0$ for *any* quadrature: conservation is exact by construction, which is the particle analogue of swept-region remapping in ALE finite volumes. The integrals are evaluated on a lattice of spacing $\Delta x/\texttt{samples\_per\_dx}$ over the box of half-width $2R$ around the event, using only the old neighbour structure. The new state is $U_i = (V_i U_i + \delta m_i)/(V_i + \delta V_i)$.

Two corrections follow:

A survivor's volume changes by its swept share relative to the larger of its volume and its share of the quadrature box: $V_i \leftarrow V_i + \delta V_i \min(1, V_i / \int_{\text{box}} \phi_i^{\text{old}})$, and its swept content is scaled alike. For volumes that are the partition's this is $V_i + \delta V_i$. Volumes that are not (an estimator of higher degree, the scheme's own of the content form) can be smaller than the share an event takes away; the former $\max(V_i + \delta V_i,\ 10^{-12} V_i)$ then left volumes of $10^{-12} V_i$ with a negative content, and repeated events volumes of $10^{-148}$ (Sedov). The residual step keeps both sums exact.

1. **Residual.** The stored content of a removed particle, $V_r U_r$, is not exactly $\int \phi_r^{\text{old}} \tilde U$. The difference is spread over the affected particles in proportion to their new volumes, so $\sum V U$ is preserved to round-off.
2. **Repair.** Each affected particle is clipped to the bounds of its old neighbourhood, and the clipped content is redistributed to affected particles that still have room. This enforces the discrete maximum principle while keeping the sum exact; if no room is left, the remainder is spread by volume and the bounds are violated by that remainder.

**Boundary particles** (the ghosts of a non-periodic box) are not remapped: their states are boundary data (`FixedDirichlet` restores them from `pg.rhos`, `OutflowBC` overwrites them at the next stage), so an event next to a wall leaves their states and volumes as they are. Their shape functions still change near the event; that swept share is left out of the balance, so the residual step puts it onto the interior particles and $\sum V U$ over the interior (the whole content, since ghosts carry no volume) is preserved exactly; `mass_residual` reports the change of the interior content. (Up to version 0.7.9 the ghosts were remapped like interior particles, which changed Dirichlet data permanently.) Events only start at interior particles, and insertions need two interior particles.

Then the particle arrays are compacted/extended, the neighbour lists are rebuilt, and new particles start at the scheme's maximum order. The remap is second-order accurate in smooth regions (linear reconstruction, averaged over the new shape function's support).

### Volumes after the call

The remap's volumes $V_i + \delta V_i$ are integrals of the Shepard partition. A moving grid re-estimates the volumes after the next step with its own estimator, which defines volumes differently (per particle, the two differ by several percent near an event), and that re-estimation would change $\sum V U$ by about as much as an event moves. With `volume_estimator` set (the builders pass the stepper's estimator on moving grids), `adapt_particles!` therefore re-estimates the volumes right after the topology change and puts the resulting change of the interior $\sum V U$ back onto the particles the remap changed (not with `own_volumes = true`, for a stepper that keeps its volumes between the steps, [`keeps_own_volumes`](@ref): there the remap's volumes stay, $\sum V U$ is exact without a correction, and only the seeds are given the estimator's volume; replacing all volumes, as up to 0.9.9, put the mismatch between the scheme's and the estimator's volumes of the whole cloud onto the changed particles), with the bounded, exactly conservative redistribution of the repair above. The call is then exact for the volumes the following steps use, and the states differ from the remap's only by that small correction. Ghost particles carry no volume, so the interior content is the whole content and `total_mass` stays exact as well. (Up to version 0.7.9 the ghosts had volumes and their content change was put onto the interior particles, i.e. moved across the wall: about $10^{-3}$ of the mass per event next to a wall; in 0.8 it stayed with the ghosts.) (Keeping each particle's content $V_i U_i$ across the re-estimation would also be exact, but it puts the per-particle volume differences, up to 18 % next to a merge, into the states.)

### Cost

A call on a well-spaced cloud inside its region costs two kernels (the interior particles next to the boundary, the pairs that violate the rules). Otherwise the serial passes visit the flagged particles only, the candidates of later rounds follow from those of the first (the particles left, the neighbours of what was removed, the new particles and their neighbours), the remap's scratch is one neighbour list, and with `own_volumes` no estimator runs over the cloud. The clearance tests use the stored distance vectors of the neighbour lists, and the lists are repaired between the rounds. Measured with $R = 2.5\,\Delta x$, 8 threads, as the cost of a call in time steps of the scheme (`RK2MUSCL2` with the VK limiter unless stated), with a full search per round (0.9.10) and with the repair:

| case | particles | full search per round | repair |
|:--|:--|:--|:--|
| 3-D blast, 48³ | 114 k | 0.91 | 0.37 |
| 3-D blast, 64³ | 268 k | 0.89 | 0.32 |
| 3-D blast, 100³ | 1.02 M | 0.82 | 0.29 |
| 3-D vortex, 64³ | 273 k | 0.49 | 0.29 |
| 3-D vortex, 100³ | 1.03 M | 0.33 | 0.20 |
| 2-D Riemann C6, 800², `RK2MUSCL1` | 703 k | 0.71 | 0.39 |
| 2-D Riemann C6, 800² | 702 k | 0.48 | 0.27 |
| 2-D Kelvin–Helmholtz, 512² | 298 k | 0.27 | 0.20 |
| 2-D Sedov, 400², `RK2MUSCL1` | 179 k | 0.74 | 0.47 |

The management is then 16 to 32 % of the run time (21 to 48 % before). One full neighbour search costs about a tenth of a time step; where the active region is smaller than the event spacing (the blast) a round serves one or two events, and a call ran to the limit of eight rounds, each with a search. (Before the measures of 0.9.10 a call cost about 10 time steps on C6 at 100²: 42 % the estimator's sweep over the region, 20 % the allocation of a scratch over all neighbour slots of the grid, 20 % detection over all particles.)

On a device grid the events still run on the host, on a copy of the particle arrays (positions, states, volumes, wall geometry) that is kept between the calls. The neighbour slots stay on the device: the host searches the lists the events read, those of the candidates the device pre-check flagged, of the escaped particles and of the local sets of the remaps (a few hundred to a few thousand particles of a million), and repairs them between the rounds as on the host. The particle arrays are copied back and the device searches once. Up to 0.9.10 a call copied the whole grid with its neighbour slots to the host (in the slot-major layout of the device, converted in a serial loop), allocated that copy anew every time and searched all lists on the host. Measured in single precision (`RK2MUSCL2` with the VK limiter unless stated, $R = 2.5\,\Delta x$, 16 host threads), per call of the management:

| case | GPU | particles | time step | call, 0.9.10 | call | in time steps |
|:--|:--|:--|:--|:--|:--|:--|
| 2-D Riemann C6, 800² | V100 | 702 k | 64 ms | 1950 ms | 227 ms | 30 → 3.5 |
| 2-D Kelvin–Helmholtz, 512² | V100 | 290 k | 32 ms | 656 ms | 42 ms | 18 → 1.3 |
| 2-D Sedov, 400², `RK2MUSCL1` | V100 | 179 k | 16 ms | 484 ms | 63 ms | 35 → 3.9 |
| 3-D blast, 64³ | V100 | 266 k | 194 ms | 1705 ms | 212 ms | 8.8 → 1.1 |
| 3-D blast, 100³ | A40 | 1.01 M | 610 ms | 4277 ms | 557 ms | 7.0 → 0.91 |
| 3-D vortex, 100³ | A40 | 1.02 M | 592 ms | 3215 ms | 470 ms | 5.4 → 0.79 |

The copies between the device and the host are then 10 to 30 ms of a call. What is left in 3-D is the quadrature of the remaps (about 200 ms at 100³, serial, the same code as on the host), the device search at the end (96 ms) and the pre-check (43 ms); in 2-D the passes over the whole cloud that every round makes on the host (the compaction, the list repair, the index maps), five rounds per call on C6. `HyperCloudCore._ADAPT_TIMING[] = true` collects the wall time of the sections of the calls in `HyperCloudCore._ADAPT_TIMES`.

### What the remap does not do

- It does not change states outside the event's $3R$–$4R$ neighbourhood, so far-field solutions are untouched (with `volume_estimator`, far-field *volumes* change by the estimator's global renormalisation, typically $O(1/N)$; not with `own_volumes`).
- It does not conserve across the time step itself; see above.
- It does not move particles for mesh quality. A repulsion or smoothing term belongs in the `GridMover`, and keeps the particle set well-spaced *before* adaptivity has to act.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Volumes.jl", "Adaptivity.jl"]
Private = false
```
