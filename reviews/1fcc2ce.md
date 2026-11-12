# Review: HyperCloudCore.jl — moving boundaries (PR #41, 44bb41d..1fcc2ce)
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation

Independent review of the merged PR #41 (commits 2cfc6cb "Moving boundaries: rigid pieces, placed ghosts, swept
interior" and 8e8b732 "no interior particles on a moving face, flush faces refused"; WIP 9bb581f squashed into them),
in the read-only worktree `HyperCloudCore.jl/.claude/worktrees/review-mb` (= origin/main_dev, 1fcc2ce). Probes and
logs are in `/tmp/brev3/r2/probes/` and `/tmp/brev3/r2/full_suite.log`. No file in the repository was edited and no
git state was changed. One probe (`seq_sweep.jl`) redefines `sweep_boundaries!` **at run time inside the probe
process only** (one event per remap, neighbour search between events) to test a hypothesis about H1; the repository
is untouched.

## Scope

- `src/Grid/Shapes.jl`: `RigidMotion`, `translating`, `rotating` (2-D/3-D), `Moving`, `Posed`, `at_time`,
  `moving_pieces`, `wall_velocity`; `_probe` returns the piece.
- `src/Grid/Domains.jl`: `pieces`/`piece_of_tag`, `_ShapeInside`/`_ShapeTag` with a time, `_assign_pieces!` (shared
  tags and flush faces refused), `GhostMotion`, `tag_motion(s)`, `_push_out` for points on a moving face,
  `wall_geometry(geom, x, t)`.
- `src/Grid/MovingBoundaries.jl` (new): `place_ghosts!`/`_place_ghost!`, `_update_region!`, `sweep_boundaries!`.
- `src/Grid/Adaptivity.jl`: `_remap_event!` with `removed_mask`/`is_other`, ghosts out of the partition, region at
  `boundary_time`; `_apply_topology!` wall data at `boundary_time`.
- `src/Grid/BoundaryConditions.jl`: ghost fit moves ghosts by `tag_motion(geom, tag, boundary_time, t)`, wall velocity
  in `WallPoint`, argument tuple packed to ≤ 20.
- `src/TimeStepping/TimeSteppers.jl`: `_is_moving` with moving pieces, `place_ghosts!` per stage and at step end,
  no volume re-estimation for `NoGridMover`, `sweep_boundaries!` after each step.
- `ParticleGrids.jl`/`DeviceAdapt.jl` (`boundary_time`, `sweep_time`), docs (new page `moving_boundaries.md`,
  domains/BC/stepper/roadmap), `CLAUDE.md` (20-argument rule), tests (`test/unit_schemes.jl:869-1040`).

Read completely: the diff, `MovingBoundaries.jl`, the moving-piece parts of `Shapes.jl`/`Domains.jl`, `_remap_event!`
and `_apply_topology!`, the stepper's step/stage/BC flow (`TimeSteppers.jl:300-595`), `update_volumes!`/
`_renormalise_volumes!`, the docs page.

## Tests run

- Full suite under the lock: `flock /tmp/hcc_test.lock timeout 60m julia -t 8 --project=test test/runtests.jl`
  → **exit 0**: core 25011/25011, schemes 428/428, pipeline 135/135, backends-a 25/25, backends-b 71/71 (wall
  ≈ 2 min per group in parallel).
- Probes (`julia -t 2 --project=<worktree>/test /tmp/brev3/r2/probes/<file>`; `common.jl` holds the test suite's
  `LinearAdvection`). Scalar advection, MUSCL3 unlimited + Rusanov, RK3-SSP, `MOOD()` as in the new tests unless noted.

| probe | what | result |
|:--|:--|:--|
| `p1_shapes.jl` | FD check of `body_velocity` vs `d/dt body_to_world`, R orthogonality, fixed centre (2-D, 3-D arbitrary axis); Float32 constructors; `Moving` under `Rotated`/`Translated` | velocities 8e-11 / 2.6e-11, ‖RᵀR−I‖ 1.6e-16, det 1, centre fixed. Float32 / wrappers: see L2, M1 |
| `p2a_translated.jl` | ghosts of `Translated(Moving(…rotating…))` after `place_ghosts!(0 → 0.5)` | off the piece by **0.146** (untranslated control 3e-17), M1 |
| `p2_grid.jl` | one stepper step starting at t₀ = 0.3 | ghosts moved by dt only, region at pose(0.305), M2 |
| `p3_plate.jl` | commit's plate case (N = 20/40/80, t = 0.5) and the plate crossing the periodic seam (t → 1.4) | reproduction below; seam: M3 |
| `p4_piston.jl`, `p5_piston2.jl` | 1-D receding (commit case), advancing against the flow, oscillating, fast pistons | table below |
| `p6_piston2d.jl`, `p7_dups.jl` | 2-D plane piston in a channel periodic in y | duplicate particles and +5 % region volume, H2 |
| `p8_rest.jl`, `p9_rest2d.jl`, `p10_onesweep.jl` | fluid at rest (velocity 0): only the sweep acts; one sweep that removes a row | 1-D fine; 2-D: states change by 0.2 up to 12 h from the face, H1 |
| `p11_plate_smooth.jl`, `p12_plate_seq.jl` | co-moving plate with structure at the plate; same with the probe-only sequential sweep | diverges 0.085/0.17/0.23; sequential 0.048/0.083/0.092, H1 |
| `p14_piston2d_cmp.jl`, `p15_profile.jl`, `p16_wallband.jl` | 2-D receding plane piston co-moving, walls y = 0, 0.5; error by distance to the face; wall ghosts behind the piston | non-convergent; no wall ghosts/wall-row particles in the uncovered part, H3 |
| `p17_perf.jl`, `p23_alloc.jl` | step cost moving vs static plate (N = 80), allocations | 4.5× slower, 15 MB/step, M5 |
| `p18_ka.jl` | KA CPU vs Polyester `MirrorBC(op; degree = 2)` on a rotating plate's tags with `t ≠ boundary_time`; `WallPoint.velocity` | parity **0.0** (bitwise), 0 B per launch; velocity at the ghost, L1 |
| `p19_misc.jl` | ball leaving the canvas through a static wall; two balls (tags 5, 6) closing to a 1 h gap; 3-D ball (h = 1/16); Float32 plate | all run; 3-D sweep drifts out of the DMP range, H1 |
| `p20_3d.jl`, `p21_3d_seq.jl` | 3-D ball, place + sweep only (no PDE), 20 sweeps; same with the sequential sweep | U leaves [1, 2] → [0.875, 2.125]; sequential [0.9953, 2.0047], H1 |
| `p22_lagr.jl` | 1-D receding piston with `LagrangianMover` | Σ V renormalised to the t = 0 volume, M4 |

Not run: device tests (no GPU), the docs build, HyperCloud.jl against this tree, the Euler-like vector system with a
slip operator (not attempted). A spinning ball (`p24_spinball.jl`, region invariant) was inconclusive (the scalar
problem past a ball is not consistent) and is not used.

### Accuracy: reproduction and harder cases

| case | N | max error | commit / notes |
|:--|:--|:--|:--|
| plate translating with the flow (commit case), t = 0.5 | 20 / 40 / 80 | 0.265 / 0.0652 / 0.00997 | commit 0.22 / 0.065 / 0.011. Σ V = 1.960000 at every N ✓; content change −0.0039 / −0.0025 / −0.00064 (commit "0.03 / 0.004 / 0.0007"); min U 0.9992 at N = 80 ✓ |
| same, smooth data with structure **at** the plate (`1 + 0.5 sin π(x−t) cos 2πy`) | 20 / 40 / 80 | **0.085 / 0.172 / 0.230** | diverges (H1, H3-like near-wall residual); in the commit case u − 1 ≤ 5e-7 at the plate |
| 1-D piston receding at the flow speed (commit case), t = 0.5 | 50 / 100 / 200 | 0.197 / 0.0596 / 0.0138 | commit 0.19 / 0.060 / 0.014 ✓; Σ V = 1.500000 ✓; inflow 0.5105 / 0.5012 / 0.5002 ✓ |
| 1-D piston advancing (w = −0.5) against the flow, pulse crossing the face, t = 0.4 | 50…400 | 0.189 / 0.061 / 0.018 / 0.0064 (Extrapolation), 0.189 / 0.072 / 0.024 / 0.0097 (Mirror) | static face at the same place: 0.17 / 0.049 / 0.011 / 0.0026. Converges, order ≈ 1.6 vs 2.1 |
| 1-D oscillating piston x_p = 1 + 0.05 sin 4πt (\|w\| ≤ 0.63 < a), t = 0.8 | 50…400 | 0.230 / 0.098 / 0.031 / 0.0104 | converges, order ≈ 1.5 |
| 1-D fluid at rest, piston advancing | 50…400 | 0.0106 / 0.0014 / 0.00036 / 0.00009 | order 2 |
| 2-D plane piston receding co-moving (walls y = 0, 0.5, MirrorBC), x > t + 0.2 | 20 / 40 / 80 | **0.233 / 0.254 / 0.190** | static face: 0.068 (N = 20), 0.017 (N = 40) over the same region; H3 (+H1) |
| 2-D plane piston advancing into fluid at rest | 20 / 40 | 0.206 / 0.203 after 4 steps (first row removed) | H1 |

## Findings

### H1 (high, CONFIRMED) — the sweep runs all removals/insertions of a step as simultaneous remap events on the old topology: in 2-D/3-D a row of events corrupts volumes and states up to ~12 spacings from the face
`src/Grid/MovingBoundaries.jl:258-267` builds one `_AdaptEvent` per removed particle and per inserted point and runs
`_remap_event!` for all of them on the same neighbour lists, with no event spacing. The adaptivity guarantees
independent events by `_event_spacing` (`Adaptivity.jl:300-315`: "two remaps are only independent if neither sees a
particle that the other removes or adds"); the sweep does not. The `removed_mask`/`is_other` fix
(`Adaptivity.jl:433-437, 540-542`) takes other events' removals out of `V_before` only: in the quadrature
(`Adaptivity.jl:486-494, 522-529`) `S_new` still omits *every* removed particle while `S_old` contains them, so each
event hands its survivors the shares of all removed particles in its box, then the "exact residual correction"
(`:561-566`) takes the excess back from the whole local set (radius `R(1+2√D)+dx` ≈ 10.6 h in 2-D, 12.2 h in 3-D) in
proportion to volume, and the DMP repair pushes states to their (ghost-inclusive) bounds. Insertions are not seen by
later events either (each inserted particle takes its share as if alone). 1-D is unaffected (one event per step).
Evidence:
- `p10_onesweep.jl`: one sweep, 2-D plane piston at h = 1/40 removing one row (20 particles), smooth data, no PDE:
  surviving particles change by **|ΔU| up to 0.19–0.21 at 4–12 h** from the wall, volumes by **×0.52 … ×2.8**;
  a ball sweep with 2 removals + 4 insertions changes them by ≤ 0.045 (≤ 0.003 beyond 4 h).
- `p9_rest2d.jl`: plane piston advancing into a fluid at rest (velocity 0, the solution must stay u0): at the first
  row removal (step 4) a particle 7 h from the face jumps to U = 1.4997 (u0 = 1.294, V/h² = 0.57); max error 0.206
  (N = 20) and 0.203 (N = 40): non-convergent. 1-D (`p8_rest.jl`): order 2.
- `p11_plate_smooth.jl`: commit's co-moving plate with structure at the plate: 0.085 / 0.172 / 0.230 (N = 20/40/80),
  errors beyond 12 h 0.064 / 0.017 / 0.046. With the probe-only sequential sweep (`p12_plate_seq.jl`):
  0.048 / 0.083 / 0.092 and beyond 12 h 0.027 / 0.0075 / 0.0023 (converging there).
- `p20_3d.jl`: 3-D ball (r = 0.2, h = 1/16) moving at 0.5 through U = 1 + y, place + sweep only: U drifts to
  **[0.875, 2.125]** after 20 sweeps (worst particles on the y = 0/1 walls 5–8 h from the ball); sequential
  (`p21_3d_seq.jl`) [0.9953, 2.0047].
The commit's plate test cannot see this: its data are u − 1 ≤ exp(−14.5) at the plate.
- Fix: make the events independent: process the sweep in rounds (select a subset with the adaptivity's
  `_event_spacing`, remap, apply the topology, neighbour search, repeat until no candidates remain; insertions then
  also see each other), or remap each connected cluster of events as one event (all its removed and added particles
  in one quadrature and one balance). Add a 2-D test with non-constant data at a moving face (e.g. the advancing
  plane piston at rest: error vs u0 away from the face).

### H2 (high, CONFIRMED) — a sweep box spanning a periodic axis duplicates the seam: +1/N of the swept volume and coincident inserted particles
- Region pass: `MovingBoundaries.jl:95-105` iterates `ceil(lo/hq):floor(hi/hq)` and wraps; when the (clipped) box
  spans a periodic axis, index 0 and index n wrap to the same point. The `visited` set that would catch it exists only
  for more than one box (`:95`).
- Insertion: `:230-240` has the same index range on the spacing lattice, and a candidate is checked against existing
  particles only (`_collect_near!`, `:237`), not against the points already in `added`.
`p7_dups.jl`/`p6_piston2d.jl`, 2-D plane piston (HalfSpace, unbounded → the box is the whole canvas) in `[0,2]×[0,0.5]`
periodic in y, N = 20 (10 rows): every insertion adds **11** particles, the minimum pair distance is **0** from the
second step on; Σ V grows by 0.013125 per uncovered column instead of 0.0125 (21/20); at t = 0.5 Σ V = 0.7625 vs
|Ω| = 0.75 (N = 40: 0.75625). Errors 0.53 / 0.59 / 0.30 (N = 20/40/80). Any piece whose box (bounding box + R + 2dx)
reaches both ends of a periodic axis hits this (a piston or long body in a channel periodic across it — HyperCloud's
planned 2-D Euler piston).
- Fix: on periodic axes iterate `0:n-1` with `n = round(L/hq)` (resp. `L/dx`) once and clip the box to one period, or
  always deduplicate by wrapped key; check insertion candidates against `added` as well (H1's rounds do this).

### H3 (high, CONFIRMED) — the uncovered region has no static-wall ghosts and no wall-row particles
The ghost band of every static wall is generated once at t = 0 and filtered to the t = 0 interior
(`Domains.jl:339-341`, `_narrow_band`); the sweep inserts only lattice points with `signed_distance ≤ −h/2`
(`MovingBoundaries.jl:236`), i.e. never the wall-row particles that `get_points` snaps onto walls. Behind a receding
piece next to a static wall the cloud therefore has neither (`p16_wallband.jl`, 2-D piston in `[0,2]×[0,0.5]`,
h = 1/40, w = 1, t = 0.5): bottom-wall ghosts (tag 3) only for x ≤ 1.025 while the face is at 1.5; 342 interior
particles beyond; **0** interior particles on y = 0/0.5 for x > 1 (82 for x ≤ 1); the particle (1.475, 0.025) has
5 ghost neighbours of 12, all the piston's. The wall condition is simply absent there: co-moving receding piston,
error by distance to the face (N = 40, `p15_profile.jl`) 0.33 / 0.33 / 0.32 / 0.23 / 0.16 for 0–4 h against
0.009–0.013 for a static face; max error 0.233 / 0.254 / 0.190 for N = 20/40/80 (`p14_piston2d_cmp.jl`, beyond 3 h of
the face 0.14 / 0.15 / 0.10). The documented limit (`moving_boundaries.md`, "the ghost band of a piece is the one
generated at t = 0 inside it") covers the piece's own band, not the static walls'.
- Fix: generate the static walls' ghost band (and the wall-snapped rows) over everything a piece can uncover (e.g. keep
  the ghost candidates within the cutoff of the canvas inside the union of the pieces' boxes over the motion, or add
  static ghosts in the sweep from the canvas lattice with `get_tag`, wall normals and distances), and insert wall-row
  points with the same snapping rule as `get_points`. At least document and refuse a receding piece adjacent to a
  static wall until then.

### M1 (medium, CONFIRMED) — `Moving` under `Translated`/`Rotated`: ghosts and wall velocities use the piece's local-frame pose
`at_time` handles the wrappers (`Shapes.jl:599-600`), so the shape is right, but `_probe` returns the bare `Moving`
node and `pose(m, t)` is in the wrapper's local frame: `tag_motion`/`GhostMotion` (`Domains.jl:428-432`) and
`wall_velocity` (`Shapes.jl:617-620`) apply it to world coordinates. `p2a_translated.jl`:
`Translated(Moving(plate, rotating(1.0)), (0.3, 0))` — after `place_ghosts!(0 → 0.5)` the ghosts' signed distance to
the piece is off by up to **0.146** (≈ 6 h; untranslated control 3e-17), i.e. they rotate about the world origin.
`p1_shapes.jl`: `wall_velocity` of `complement(Rotated(Moving(ball, translating((1,0))), π/2))` is (1, 0) instead of
(0, 1); of `Translated(Moving(ball, rotating(1)), (1, 0))` at (1, 0.35), t = π/2: (−0.35, 1.0) instead of
(−0.35, 0). Silent; nothing refuses it.
- Fix: compose the wrappers' transforms into the motion (have `moving_pieces`/`_probe` return a world-frame motion:
  for `Translated` `c' = c + o − R…`, for `Rotated` conjugate by the rotation), or throw in `GeometricDomain` when a
  `Moving` sits under `Translated`/`Rotated` and tell the user to put the transform inside `Moving`. Add a test.

### M2 (medium, CONFIRMED) — a run that does not start at t = 0 (or resumes) places the ghosts wrongly
Ghost positions in the stage buffers are moved from the step start `time` (`TimeSteppers.jl:564, 582`,
`place_ghosts!(…, X, t_n, …)`), but the generated ghosts are at pose(0) and `boundary_time = sweep_time = 0`
(`ParticleGrids.jl:267`). `p2_grid.jl`: one step of the translating plate from t₀ = 0.3, dt = 0.005: the ghosts move by
**0.005** while the region the sweep uses is the plate at 0.305: 48 interior particles found inside the plate (the
warning fires), Σ V = 1.9541 vs 1.96, and the plate's ghosts stay 0.3 behind the body for the rest of the run
(normals are rotated from `boundary_time`, positions from `t_n`: inconsistent too). Any `tspan` not starting at 0, and
a restart from a checkpoint without `boundary_time`/`sweep_time`, hits this.
- Fix: move positions from `pg.meta.boundary_time` as the normals are (in the stepper: at step start, if
  `boundary_time ≠ time`, place `pg.core.positions` from `boundary_time` to `time` and sweep from `sweep_time`), or
  provide `set_boundary_time!(pg, t0)` and throw when `boundary_time ≠ time` at step start.

### M3 (medium, CONFIRMED) — a piece crossing a periodic seam is lost
`Posed` evaluates the shape without periodic images (`Shapes.jl` `signed_distance(p::Posed…)`, `_probe`), while the
ghosts are wrapped by the neighbour search. `p3_plate.jl`, commit's plate case run to t = 1.4 (the plate reaches
x = 1 at t = 0.8): Σ V 1.960000 → 1.960625 (t = 0.8) → 2.000625 (t = 1.2, the whole channel), interior count
3195 → 3280 (every lattice point, including those inside the plate's wrapped part, while its ghosts are still there:
interior particles and ghosts overlap); error 0.065 (t = 0.5) → 0.095 → 0.113 (t = 1.0) → 0.144 (t = 1.4).
- Fix: evaluate moving pieces with the nearest periodic image (min over images in `signed_distance`/`_probe`/
  `bounding_box`, as the cell lists do), or warn/throw when a piece's box crosses a periodic boundary; document.

### M4 (medium, CONFIRMED) — the volume estimators use the t = 0 region and renormalise to the t = 0 volume
`update_volumes!` integrates `geom.is_interior(y)` (one argument, t = 0; `Volumes.jl:288`) and
`_renormalise_volumes!` scales to `domain_volume(pg)` (t = 0; `:255`). The stepper still calls it for a grid mover
with moving pieces (`TimeSteppers.jl:587`), and the adaptivity's `_reestimate_volumes!` (`Adaptivity.jl:219`) for any
grid. The docs say "a `GridMover` can move the interior as well". `p22_lagr.jl`, 1-D receding piston with
`LagrangianMover`: **Σ V = 1.010 vs |Ω(t)| = 1.496** at t = 0.496 (NoGridMover: 1.500): a third of the content is lost
from `interior_mass`.
- Fix: integrate the region at `boundary_time` and renormalise to its current volume (`shape_volume` of `at_time`
  clipped to the box, or the volume the sweep maintains), or refuse a `GridMover`/adaptivity estimator together with
  moving pieces and say so in the docs.

### M5 (medium, CONFIRMED) — a step with a moving body costs 4.5× a static one and allocates 15 MB
`p17_perf.jl`, plate channel N = 80 (13.8k particles), 2 threads: static 5.65 ms/step and 41 kB, moving
**25.2 ms/step and 15.0 MB**. Components: sweep 3.0 ms and 4.3 MB per call (0.49 ms and 2.9 MB when nothing moved:
the region pass and its per-call `Vector`s/`Set`), neighbour search 0.93 ms, `set_positions!` 0.49 ms per stage;
the rest is the moving-grid path recomputing every particle's geometry caches each stage although only ghosts moved.
`place_ghosts!` allocates 4.3 kB per call (`tag_motions` builds a runtime-length `ntuple`, so the launch is
dynamically dispatched; `p23_alloc.jl`). The stated performance check of the brief is not in the PR.
- Fix: refresh only the neighbour data/caches of rows that contain a moving ghost (frozen interior–interior geometry);
  build the motion table with a static length (`Val(K)` stored in the domain); preallocate the sweep's scratch.
  At least state the cost in the docs.

### L1 (low, CONFIRMED) — `WallPoint.velocity` is the body velocity at the ghost, not at the wall point
`BoundaryConditions.jl:378`: `velocity(gm, x_i)` with `x_i` the ghost; the docs/docstring say "at the wall point"
(`moving_boundaries.md` §2, `WallPoint` docstring). The difference is `Ω φ n`: tangential only (Ω is skew), so a slip
wall is unaffected, but a no-slip or diffuse (BGK) wall — the planned next step — gets a tangential wall velocity off
by `|ω| φ` (φ up to R + dx ≈ 3.5 h): `p18_ka.jl`, rotating plate ω = 1: 0.04 = the largest ghost distance.
- Fix: `velocity(gm, x_i - φ_i * n_i)`.

### L2 (low, CONFIRMED) — Float32 motions need explicit Float32 centres and velocities
`Moving(Ball((0f0,0f0), 0.2f0), rotating(1f0))` throws "motion(t) must return a RigidMotion{2, Float32}, got
…Float64…" because the default `center = (0.0, 0.0)` promotes to Float64 (`Shapes.jl:502, 514`); the 3-D form with all
Float32 arguments fails the same way, and `translating((1.0, 0.0))` on a Float32 shape too (`p1_shapes.jl`). Only
`rotating(1f0; center = (0f0, 0f0))` and `translating((1f0, 0f0))` work. Once built, a Float32 run is fine
(`p19_misc.jl`: Σ V 1.96003, U stays 1).
- Fix: take `T` from the shape in `Moving` and convert the motion's output (`RigidMotion{D,T}(m(t))`), or default
  `center` to integer zeros so `ω` decides the type.

### L3 (low, CONFIRMED) — beyond |w|Δt ≤ Δx/4 the volumes leave the region, and the warning is once per session
`MovingBoundaries.jl:138`: a particle whose whole share is covered keeps its old volume and gives it away when it is
removed. `p5_piston2.jl`, oscillating piston |w| up to 2.5, CFL 0.4 (|w|Δt up to 1 Δx): Σ V 1.1105 vs face 1.0951 at
N = 50 (0.77 Δx; with |w|Δt ≤ Δx/4 0.245 Δx, the quadrature). The warning (`:221`) is a process-global flag
(`_WARNED_SWEEP`) and fires only when a particle is already inside the solid, not when |w|Δt > Δx/4. Documented limit.
- Fix: warn per grid (or per stepper) whenever `max |w| Δt > clearance·Δx`; zero the volume of a fully covered
  particle in the region pass so its removal hands out nothing.

## Correctness checks that passed

- **Motions**: 2-D `rotating` (R, `Ω = ωJ`, `v = −Ω R c₀`) and 3-D Rodrigues with an arbitrary axis: velocities equal
  the finite-difference derivative of `body_to_world` to 8e-11 / 2.6e-11, R orthogonal to 1.6e-16, the centre fixed;
  `GhostMotion` `c_to + R_to R_fromᵀ (x − c_from)` maps body points exactly; counter-clockwise sign as documented.
- **Placement**: exact (commit test, 2e-15 after 200 steps); the stepper resets the ghost rows of `X_stages`/`x_cand`
  to `x_n` in `_stage_predictor!` before each `place_ghosts!`, so MOOD re-forms do not double-move them; BCs in
  `form_candidate!` see the ghosts at stage i−1 and move them to `t_i` via `tag_motion`; ghosts get `K_X = 0` under a
  `LagrangianMover`, so a mover does not move them twice.
- **Kernels**: `_ghost_fit_body!` has 19 tuple entries; KA CPU vs Polyester ghost fit with a non-identity
  `GhostMotion` and an op reading the velocity: bitwise equal, 0 B per `MirrorBC` launch.
- **Mechanics**: a ball leaving the canvas through a static wall (no crash, Σ V 1.0011 vs 1); two balls with own tags
  closing to a 1 h gap (`piece_of_tag = [0,0,0,0,0,1,2]`, Σ V exact, no particle left in the gap); 3-D ball Σ V
  0.96648 vs 0.96649 constant; shared tags and flush faces refused; 1-D accuracy as claimed (table above).

## Test coverage

- The new testset cannot even load on the parent (new exports), so "fails on the parent" holds trivially; what matters
  is what it can detect.
- The translating-plate test has u − 1 ≤ 5e-7 at the plate: it cannot see H1 (nor M3, it stops at t = 0.25 before
  the seam). The advancing-piston test is 1-D and checks bounds/volume/content only; the rotating-plate test checks
  bounds only. There is no 2-D test with non-constant data at a moving face (H1), no box spanning a periodic axis
  (H2), no receding piece next to a static wall (H3), no `Translated`/`Rotated` wrapper (M1), no start at t ≠ 0 (M2),
  no grid mover with moving pieces (M4), no allocation/performance check of a moving step (M5).
- KA parity covers `_place_ghost!` only; the changed ghost-fit kernel (`gm` argument) is untested on KA (verified by
  probe: equal).

## Minor issues

- `src/Grid/Domains.jl:398` `_tag_motion` is dead code.
- `MovingBoundaries.jl:12-15` (header comment) says both steps go through the conservative remap "so the content of the
  region is preserved exactly" — contradicts the two-step design and the docstring.
- `sweep_boundaries!(pg, t_old, t_new, …)`: `t_old` is overwritten by `pg.meta.sweep_time` (`:181`); the docstring
  describes it as the start pose. Drop it or document that it is ignored.
- `piece_of_tag` lives in the (shared) `GeometricDomain` and is rewritten by every `ParticleGrid` construction
  (`Domains.jl:366-389`); two grids from one domain with different spacings can disagree about which tags exist.
- Commit/docs numbers: plate N = 20 error 0.265 (claimed 0.22) and content change −0.0039 / −0.0025 / −0.0006
  (claimed 0.03 / 0.004 / 0.0007) — probably measured on 2cfc6cb before 8e8b732 changed the face rows.
- Tag numbers are limited to 63 (`Domains.jl:438`); not documented on the docs page.

## Verdict

The rigid-motion algebra (2-D/3-D, signs, velocities), the exact ghost placement through the stages, the KA parity, the
≤ 20-argument packing and the 1-D results are correct, and the commit's 1-D and plate numbers reproduce (plate N = 20
somewhat worse than claimed). In 2-D/3-D the sweep is not yet usable for accuracy: simultaneous remap events
corrupt volumes and states up to ~12 h from a moving face, so the co-moving plate diverges as soon as the data have
structure at the plate and a 3-D sweep leaves the DMP range (H1); boxes spanning a periodic axis insert coincident
particles and over-count the region (H2); and a receding piece next to a static wall leaves a region without wall
ghosts or wall rows (H3). Further: wrappers around `Moving` silently mis-place ghosts (M1), runs not starting at 0
mis-place them (M2), pieces crossing a periodic seam vanish (M3), volume estimators/grid movers use the t = 0 region
(M4), and a moving step costs 4.5× a static one with 15 MB of allocation (M5). The tests pass because their data are
constant where the bugs act. Full suite green.
Findings: 3 high (H1–H3), 5 medium (M1–M5), 3 low (L1–L3); all confirmed.
