# Moving Grids (ALE)

Particles can move with an arbitrary grid velocity $w(x, U, t)$: fixed (Eulerian), with the flow (Lagrangian), or anything in between. The motion is part of the same Runge–Kutta step as the state, so no operator splitting is involved, and the coupled scheme has the order of the tableau.

## Formulation

For a particle moving with velocity $w_i$, the rate of change of $U$ along its path is

```math
\frac{\mathrm d U_i}{\mathrm d t} = \partial_t U + w_i \cdot \nabla U = -\nabla\cdot F(U) + w_i\cdot\nabla U + S .
```

With $w_i$ frozen at the particle (a constant within the stencil), $w_i\cdot\nabla U = \nabla\cdot(w_i U)$, so

```math
\frac{\mathrm d x_i}{\mathrm d t} = w_i, \qquad
\frac{\mathrm d U_i}{\mathrm d t} = -\nabla\cdot\big(F(U) - w_i\, U\big)\Big|_{x_i} + S(U_i).
```

The second equation is the original PDE seen from a frame moving with $w_i$. HyperCloudCore implements it by wrapping the PDE in an **`ALEFrame(eq, w_i)`** before calling the divergence interpolator:

| Method | `ALEFrame(eq, w)` |
|---|---|
| `flux(a, U)` | $F_d(U) - w_d\, U$ |
| `velocity(a, U, d)` | $\partial F_d/\partial U - w_d I$ |
| `max_eigenvalue(a, U, d)` | `max_eigenvalue(eq, U, d, w_d)` = $\max \lvert\lambda_d - w_d\rvert$ |
| `evaluate_nc_jump`, `prim2cons`, `cons2prim` | forwarded to `eq` |

Every scheme therefore reconstructs, upwinds and dissipates with respect to the *relative* velocity, without knowing that the grid moves. For $w = u$ in linear advection the relative flux vanishes identically and the state is carried exactly; the test suite checks this for every scheme. This frame is used for the `PrescribedMover` (a smooth velocity) and for every scheme other than MUSCL.

### The Lagrangian frame (conservative form)

Where $w$ jumps, as at a shock on a Lagrangian cloud, $w_i\cdot\nabla U$ is a non-conservative product: the frozen-velocity form gained mass at the shock at a rate independent of the resolution, and the shock did not converge to its position (Sod on Lagrangian particles with Euler and BGK, VK and EPD2: +0.1 to 0.5 % of the mass by $t = 0.2$, the shock 0.011 to 0.017 too far at every $N$; `study/bgk-lagrangian-contact` of the simulations). For the `LagrangianMover` and the `ScaledLagrangianMover` with the MUSCL scheme the stepper therefore uses the **`LagrangianFrame(eq, α)`**, whose velocity is a function of the state, $w(U) = \alpha\,$`fluid_velocity(eq, U)`, with two additions to the scheme:

1. **The volume term.** The particle's update is the conservative form in point values,
   ```math
   \frac{\mathrm d U_i}{\mathrm d t} = -\nabla\cdot\big(F(U) - w(U)\, U\big)\Big|_{x_i} - U_i\, \nabla\cdot w\big|_{x_i} ,
   ```
   which is $\mathrm d(V U)/\mathrm dt = -V \nabla\cdot(F - wU)$ with $\mathrm dV/\mathrm dt = V\nabla\cdot w$ through one and the same discrete divergence. MUSCL adds the term $(w_{ij} - w_i)\otimes U_i$ to every pair's flux difference ([`ale_volume_term`](@ref HyperCloudCore.ale_volume_term)), with the interface velocity $w_{ij} = \tfrac12(w(U_{ij}) + w(U_{ji}))$ of the reconstructed states.
2. **Pair-antisymmetric dissipation.** Through a shock the Lagrangian particles compress by the density ratio, and the MLS divergence weights of the pairs across it are systematically non-antisymmetric ($V_i c_{ij} + V_j c_{ji} \neq 0$; on a jittered static cloud the asymmetry is random and averages out). In the Lagrangian frame the mass flux $\rho(u - w)$ vanishes, so the Rusanov dissipation of the density is the only pair term of the mass equation, and its asymmetry was the mass source. MUSCL therefore contracts the dissipative part $F^*_{ij} - \tfrac12(F_L + F_R)$ of every pair with the antisymmetrised face vector $\tfrac12(V_i c_{ij} - V_j c_{ji})$ (the partner's weight is its cached full-order one, found through the reverse neighbour slot; the central part keeps the consistent weight, so the order is unchanged). Pairs the aligned flux orients (either particle below full order) keep their LED weights, wall pairs and one-sided pairs their plain ones.

3. **Content form.** The stages of the Runge–Kutta step carry the content $V_i U_i$ of an interior particle, with the stage volumes $V_i$ integrated from the scheme's own volume rate $K_V = V\,\nabla\cdot w$ (the pair loop writes it beside the divergence; the volume term of item 1 then drops out of the right-hand side). At the end of the step the scheme's volumes become the grid's volumes (the stepper's `volume_estimator` is not run in this form). The mass equation's right-hand side sums to exactly $-\sum_i U_i K_{V,i}$ against these volumes (its only pair term, the dissipation, is antisymmetric, and the Lagrangian mass flux $\rho(u - w)$ vanishes), so $\sum_i V_i \rho_i$ is conserved to round-off on a periodic domain and up to the wall pairs' dissipation on a closed one. This holds for a density equation, whose Lagrangian flux $\rho(u - w)$ vanishes; momentum and energy, and a scalar equation whose flux is not $u\,U$ (Burgers), keep the pair defect of their central flux parts. The point value $U_i = (V_i U_i)/V_i$ evolves as in the point-value form (the same accuracy), and $\sum_i V_i$ drifts from $\lvert\Omega\rvert$ by the discretisation error of $\nabla\cdot w$: a measure of the cloud's quality (0.61 on the collapsed blast cloud below, 0.99999 with particle management, whose events re-estimate the volumes they touch). A stage volume below a tenth of the step's start (a collapsing pair) is floored.

   Versions 0.9.6 to 0.9.8 put the content onto the re-estimated volumes every step instead, $U_i \leftarrow U_i\, V_i^{\mathrm{scheme}} / V_i^{\mathrm{est}}$. That made $\sum V_i^{\mathrm{est}} \rho_i$ exact but fed the estimate's per-particle noise into the density (MLS volumes of degree 2 on a jittered 1-D cloud differ from the trapezoid cell by 3–5 % in the median, by factors where the cloud is compressed below the lattice spacing): on the Lagrangian BGK Sod ($N = 800$, VK) the density error grew from 3.96e-3 to 7.26e-3, the mass measured with the geometric cells got worse (2.5e-5 to 2.5e-4), and with small stencils ($R = 2.5\,\Delta x$, or $3.5\,\Delta x$ at $N \geq 1600$) a pair at the shock collapsed and the run blew up where the point-value form completed. Putting a fraction of the ratio on (an exponent $\beta$ on the volume ratio) degrades the accuracy monotonically in $\beta$ (4.19e-3 at 0.1, 4.99e-3 at 0.3), so the remap is gone.

Measured on the Euler Sod of HyperCloud (`euler_1d_sod`, jitter 0.2, $R = 5.5$, ARS222 MUSCL2 with the VK limiter, MLS volumes of degree 2, $t = 0.2$; the shock is at 0.3504):

| cloud | shock, $N = 800 / 3200$ | mass change $\Sigma h_i \rho_i$, $N = 800 / 3200$ |
|---|---|---|
| static | 0.3523 / 0.3509 | +1.3e-4 / +1.5e-5 |
| Lagrangian, frozen velocity (`ALEFrame`) | 0.3628 / 0.3616 | +1.0e-3 / +1.0e-3 (linear in time) |
| Lagrangian, `LagrangianFrame` without the pair term | 0.3602 / 0.3591 | +7.7e-4 / +7.2e-4 |
| Lagrangian, `LagrangianFrame`, point-value form | 0.3538 / 0.3522 | +1.7e-4 / −4.2e-5 |
| Lagrangian, `LagrangianFrame`, content form | 0.3524 / 0.3511 | +2.5e-4 / +1.7e-4 (an open tube: inflow and outflow ghosts) |
| static, EPD2 with the aligned flux | 0.3519 / 0.3508 | +1.2e-4 / +1.4e-5 |
| Lagrangian, `LagrangianFrame`, EPD2 with the aligned flux | 0.3515 / 0.3504 | +6.6e-5 / −2.0e-4 |

The stress case is the Woodward–Colella blast (`lag_blast_1d` of the simulations: pressures 1000 / 0.01 / 100, density ratios of 6 behind the shocks, slip walls, $N = 400$, RK2 MUSCL2 VK with the aligned flux, $R = 3.5$, MLS volumes, $t = 0.038$), where a Lagrangian cloud without particle management reported +78 % of mass: within 300 steps pairs had collapsed to 0.004 spacings and crossed (the MLS weight of a nearly coincident neighbour vanishes with the distance, so the pair feels no dissipation), the rarefaction had opened gaps of 70 spacings, and the MLS volumes ranged from 0.1 to 27 spacings (the stepper now warns once about such a cloud, `_check_cloud_quality`). With the adaptivity keeping the cloud regular (spacing 0.5–1.6, no crossings):

| blast, Lagrangian + adaptivity, $N = 400$ | mass change | energy change |
|---|---|---|
| static reference | −7.7e-4 | −5.7e-4 |
| point-value form (`LagrangianFrame`, items 1 and 2) | +6.4e-2 (+9.0e-2 at $N = 1600$) | +4.0e-2 |
| content form (item 3), slip walls, mass by the geometric cells $\sum h_i \rho_i$ / by the volumes $\sum V_i \rho_i$ | −1.3e-3 / −6e-5 | −7.0e-3 / +7e-4 |
| content form with the remap of 0.9.6–0.9.8, slip walls | +2.7e-4 | +8.8e-3 |
| content form with the remap, periodic domain (pressure 1000 in the middle, same shocks) | +6e-15 | +1.5e-2 |

(The adaptivity's events change $\sum V_i \rho_i$ by their residuals; without particle management the cloud collapses (pairs at 0.007 spacings, holes of 70) and $\sum V_i$ ends at 0.61: the geometric mass is then +12 % in both forms, the content exact.)

With a scaled motion ($\alpha = 0.5$) the mass flux of the frame is $\tfrac12\rho u \neq 0$ and its central part leaks through the pair defect: +3.2 % on the periodic blast (6.2 % before item 3); the fix is complete only for $\alpha = 1$.

With both parts the Lagrangian Sod shock converges like the static one and the mass drift shrinks with $N$ (what is left at $N = 800$ is mostly the start-up transient of the discontinuous data that the static run shows as well). The scalar Burgers test of the suite (a cloud moving at $0.2 u$ through the shock of $\tfrac12 + \tfrac12\sin\pi x$, MLS volumes) goes from +1.0e-2 / +8.8e-3 to +3.1e-3 / +2.8e-3 at $N = 100 / 200$ by $t = 1.2$; in the smooth phase the drift converges (3.7e-4 / 1.4e-6 at $t = 0.6$). The remaining defect is the asymmetry of the central part and of the pairs the aligned flux orients.

Positions use only the explicit part of the tableau: $X_k = x^n + \Delta t \sum_{j<k} \tilde a_{kj} K_{X,j}$ with $K_{X,j} = w(X_j, Y_j)$.

## Movers

| Mover | Grid velocity | Requires |
|---|---|---|
| `NoGridMover()` | $0$ (static grid; the moving-grid code paths are compiled away) | – |
| `LagrangianMover()` | `fluid_velocity(eq, U)` | `fluid_velocity` for the PDE |
| `ScaledLagrangianMover(α)` | $\alpha\,$`fluid_velocity(eq, U)` | `fluid_velocity` |
| `PrescribedMover(f)` | `f(x::Space, t)` | – |

Pass the mover when building the grid: `ParticleGrid(geom, dx, wf, M; mover = LagrangianMover())`. Boundary (ghost) particles do not move.

A custom mover subtypes `GridMover` and implements

```julia
grid_velocity(::MyMover, eq, p, U, pg, t) -> Space{D, T}
```

evaluated at the current stage positions (`pg.core.positions`) and stage state `U`. Mesh-quality corrections, such as a repulsion term that keeps Lagrangian particles from clustering, belong here.

## Geometry during a step

Within a step the neighbour **topology is frozen**: each particle keeps the neighbour list from the start of the step. At every stage, `set_positions!` writes the stage positions and `refresh_geometry!` recomputes distances (minimum image) and weights. Pairs that move beyond the cutoff get weight zero; pairs that come within the cutoff during the step are picked up by the full neighbour search at the end of the step. This is the usual neighbour-skin assumption, and it holds when particles move only a fraction of the cutoff per step. With `full_search_per_stage = true` the stepper instead runs a full search at every stage.

The MOOD checkpoint for stage $k$ recomputes $K_{E,k-1}$, so the geometry is switched to $X_k$ only after that checkpoint has finished.

## PDE API for moving grids

```julia
fluid_velocity(eq, U) -> Space{D, T}                       # Lagrangian movers only
max_eigenvalue(eq, U, d, shift) -> T                       # optional, recommended
```

The default of the four-argument `max_eigenvalue` is the safe bound $\max|\lambda_d| + |w_d|$. That keeps the scheme stable but gives it more dissipation than the relative speed calls for. Override it with the exact $\max_k |\lambda_{d,k} - w_d|$ to obtain the full benefit of a Lagrangian frame; this matters for Rusanov fluxes in particular.

## Limitations

- **Time step.** `get_time_step` uses the absolute wave speeds $|\lambda|$, not $|\lambda - w|$. For co-moving grids this is conservative. If the grid moves *against* the waves, the relative speed is larger, and the CFL number has to be reduced accordingly. The particle displacement $|w|\,\Delta t$ should also stay well below the cutoff radius.
- **Volumes** in `pg.core.volumes` are recomputed after every step from the new geometry with the stepper's `volume_estimator`; see [Particle Volumes and Adaptivity](@ref).
- **Conservation.** Like the static collocation scheme, the ALE scheme is not exactly conservative in general; mass errors converge with the scheme's order. On Lagrangian clouds (`LagrangianMover` with MUSCL, see above) the mass is conserved to round-off on closed or periodic domains and to the wall pairs' dissipation otherwise; momentum and energy, and the mass of a `ScaledLagrangianMover`, keep the pair defect of the central flux parts; the other schemes keep the frozen-velocity frame and its resolution-independent gain at shocks. Pure Lagrangian motion through strong shocks needs particle management (`ParticleAdaptivity`): without it the particles collapse and cross, and the stepper warns. Particle insertion and merging between steps (`adapt_particles!`) *are* exactly conservative with respect to `total_mass`, also for calls with several events and, with the adaptivity's `volume_estimator` set to the stepper's (the builders do this), across the following volume re-estimation. Mass on a moving grid is only a convergent quantity with `MLSVolumes`; `ShepardVolumes` have a resolution-independent bias on stretched clouds.
- **Particle distribution.** Lagrangian motion clusters particles in compressions. Use `ParticleAdaptivity` between steps, and/or a mover that adds a smoothing velocity.
- **Reordering** has to happen between steps, as for static grids.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["GridMovers.jl"]
Private = false
```

The hooks of the Lagrangian frame that a scheme uses (not exported):

```@docs
HyperCloudCore.ale_volume_term
HyperCloudCore.frame_velocity
```
