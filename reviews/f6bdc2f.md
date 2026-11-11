# Review: HyperCloudCore.jl PR #37, part b — commit f6bdc2f "Boundary conditions: wall geometry per particle, fit-based ghosts"
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation

Independent A2 review of one of the two commits of PR #37 (merged into main_dev as 2c64b8b). The shapes/domains commit
07db492 is reviewed separately; `wall_geometry` is taken as given here. Probes and outputs referenced below are in
`/tmp/brev_b/probes/` (`*.jl`, `*.out`) and `/tmp/brev_b/repro/` (standalone compiler reproducer).

## Scope

`git diff f6bdc2f~1 f6bdc2f` (11 files, +648/−16), read completely:
- `src/Grid/BoundaryConditions.jl` (ExtrapolationBC, MirrorBC, WallPoint, `requires_shape`, `_check_bc_shape`,
  `_launch_ghost_fit!`, `_WallOp`, `_nearest_interior`, `_support_count`, `_fit_support`, `_ghost_fit_body!`,
  `_fit_eval`, `_fit_degree_b`);
- `src/Grid/ParticleGrids.jl` (`pg.core.normals`, `pg.core.wall_distances`, constructor, `ensure_capacity!`),
  `src/Grid/Adaptivity.jl` (`_upload_particles!`, `_apply_topology!`), `src/Grid/Reordering.jl`,
  `src/DeviceAdapt.jl`, `src/MemoryBudget.jl`;
- `docs/src/particle_grids/boundary_conditions.md`, `overview.md`, `docs/src/gpu_roadmap.md`, the matching tests in
  `test/unit_core.jl` (memory-budget loop, "Ghost states from one-sided fits", the wall-geometry part of "Shape domains").

Context read: commit messages of f6bdc2f and 07db492, repo `CLAUDE.md`, `reviews/d92911e.md` (format),
`src/Grid/Domains.jl` (`get_points`, `wall_geometry`), `src/Interpolation/Interpolators.jl` (`_cholesky_moment`,
`_pivots_ok`, `_intercept_basis`, `basis_count`), `src/Grid/MLSWeightFunctions.jl`, the step driver in
`src/TimeStepping/TimeSteppers.jl` (`particle_orders` filled with `max_order` every step), StaticArrays 1.9.24
`src/cholesky.jl`.

## Tests run

- Full suite under the lock: `flock /tmp/hcc_test.lock timeout 60m julia -t 8 --project=test test/runtests.jl` →
  **exit 0, 24 344 pass, 0 fail, 0 error** (core 23 743 in 88 s, schemes 370, pipeline 135, backends-a 25,
  backends-b 71; wall ≈ 2 min). Log: `/tmp/brev_b/fullsuite.log`.
- Toolchain: Julia 1.13.3, LLVM 20.1.8, CPU znver5 (AVX-512), StaticArrays 1.9.24. Probes with `julia -t 2 --project=test`.

| probe | what | result |
|:--|:--|:--|
| `exact.jl`, `diag.jl` | exactness for random polynomials of the degree at every ghost (ExtrapolationBC at the ghost, MirrorBC `bounded=false` at the mirror point); D = 1/2/3, F64/F32, q = 1–5, R = 2.5–5.5Δx, jitter 0.2/0.45, periodic x, InverseWeightFunction with mirror points on particles | **F64: exact everywhere the degree is reached** (≤ 2.3e-10 rel; jitter 0.45, periodic channels, corners, outermost layer). **F32**: MirrorBC fine (≤ 4e-5); ExtrapolationBC 2-D q=3 up to 0.1 rel (F4), 3-D q=2 up to **2.0 rel** (F2). 1-D: requested degree 4 gives degree 1 at R = 3.5Δx (F3) |
| `f32.jl`, `f32b/c.jl`, `repro/repro6.jl` | the F32 3-D failure isolated | the kernel's moment matrix comes out non-symmetric; reproduced standalone (StaticArrays only); correct at `-O1` and `-C x86-64` (F2) |
| `num1d.jl` | 1-D bump leaving through the right wall, MUSCL5, RK4-SSP, t = 0.6, N = 50…400, R = 3.5/4.5Δx, lattice/jittered | OutflowBC 1st order, ExtrapolationBC(1) 2nd, (3) 4th; (4) = (1) at R = 3.5Δx, 4.9 at R = 4.5Δx (F3) |
| `channel.jl stab` | the commit's channel (Gaussian on the wall, periodic x, jitter 0.2, R = 2.5Δx, unlimited MUSCL3/5, t = 2) N = 20/40 | claims reproduce: unbounded MUSCL5 N=40 ×2.24 (deg 1), ×6.92 (deg 2); OutflowBC MUSCL5 N=20 ×3.68; every bounded case max ≤ 0.91 |
| `conv2.jl`, `grow*.jl`, `chan_diag.jl` | accuracy/convergence of the same channel, with an exact-ghost reference | **MirrorBC does not converge on jittered clouds and its error grows in time** (F1) |
| `outflow2d.jl` | 2-D outflow, velocity (1, 0.3), t = 2 | ExtrapolationBC(1) identical to OutflowBC; (2) MUSCL5 1.8e10 (N=20), 3.8e2 (N=40), MUSCL3 N=40 0.33: matches the docs |
| `ka3.jl`, `ka4.jl`, `ka5.jl`, `kaalloc.jl` | KA CPU vs Polyester vs Threads, D = 1–3, F64/F32, mixed tags incl. OutflowBC; ball domains; allocation; cost; tag-order independence | KA vs Polyester ≤ 2.6e-13 (F64), 0 (F32 1-D/2-D); F32 3-D: **KA is right, Polyester/Threads wrong** (F2); balls exact at the mirror point (3e-14 / 8e-14), all mirror points inside; 0 B on Polyester, ~1 KB/launch on KA (same as OutflowBC); tag order irrelevant (bitwise); 3-D degree-3 KA compile 30 min (F5) |
| `adapt.jl` | adaptivity insertion and reordering of the new arrays (1-D/2-D, shape box) | inserted particles get `wall_geometry` of their position, reordering keeps every particle's geometry with it |

## Findings

### F1 (high, CONFIRMED) — MirrorBC is inaccurate and non-convergent on jittered clouds; the error grows in time and exceeds OutflowBC at N = 80
`src/Grid/BoundaryConditions.jl:293-365` (`_ghost_fit_body!`) in combination with the near-wall ghosts of a jittered
cloud. The commit's own channel case (velocity (1, 0), Gaussian centred on the wall y = 0 so the even extension is the
exact solution; periodic x; R = 2.5Δx; unlimited MUSCL3, RK3-SSP; dt = ½ CFL), max interior error:

| t = 0.25, jitter 0.2 | N = 20 | 40 | 80 |
|:--|--:|--:|--:|
| exact solution in the ghosts | 8.1e-2 | 4.4e-2 | 3.4e-2 |
| OutflowBC | 4.3e-1 | 2.0e-1 | 9.2e-2 |
| MirrorBC(degree 1) | 2.1e-1 | 1.7e-1 | **1.7e-1** |
| MirrorBC(degree 2) | 2.3e-1 | 1.9e-1 | **2.5e-1** |
| MirrorBC(degree 2, bounded = false) | 2.7e-1 | 2.0e-1 | **2.8e-1** |

Over time at N = 80 (jitter 0.2): MirrorBC(2) 0.25 → 0.35 → **0.60** at t = 0.25/0.5/1.0 (OutflowBC 0.09/0.23/0.15,
exact ghosts 0.03/0.03/0.03); the peak stays ≤ 1, so the commit's boundedness criterion is met while half of the
amplitude near the wall is wrong. The mean error grows from N = 40 to 80 as well (1.4e-3 → 2.4e-3). Without jitter the
mirror is fine (N = 20/40/80: 0.18/0.036/0.019, close to the exact-ghost 0.12/0.024/0.013; 0.038 at t = 1).
Jitter 0.05 is as bad as 0.2 (0.41 at t = 0.5).
- Localisation (`grow3.jl`): the culprits are the ghosts with φ < 0.5Δx, i.e. the jittered wall row (the lattice row on
  y = 0 is jittered and about half of it lands just outside the wall, φ down to 0.001Δx; 48 such ghosts at N = 80).
  MirrorBC everywhere but the exact value for φ < 0.5Δx → **3.1e-2** (exact-ghost level); exact everywhere but MirrorBC
  for φ < 0.5Δx → **2.0e-1**. MirrorBC(0) gives 0.10. The error sits in the first rows (y < 1.5Δx: 0.3; y > 6Δx: 0.016).
- So the fit is exact (t = 0 ghost error 5.6e-3, polynomials exact to round-off) but the ghosts that sit essentially
  *on* the wall, slaved to their interior neighbours every stage, feed back into the near-wall stencils in a way that
  accumulates (linear growth in time, no resolution dependence). OutflowBC has a milder version of the same problem
  (0.23 at N = 80, t = 0.5), so part of this is the jittered wall row of the grid generator, but MirrorBC — documented
  as *the* wall condition and as beating the donor copy — is worse than OutflowBC at N = 80 and does not converge.
- The unit test (`test/unit_core.jl:870-897`) only runs N = 20, where e_mirror < e_copy holds; the stability study of
  the commit message measured boundedness, not accuracy.
- Fix (to investigate): (a) do not let jitter push wall-row points across the wall (jitter tangentially only for
  points on a wall, or reflect them back inside), so every ghost is ≥ ~Δx/2 from the wall; (b) or treat ghosts with
  φ < cΔx differently (e.g. take the value of the fit at the wall point with the interior data, which is what a
  zero-normal-derivative condition means there, or exclude them from the interior stencils); (c) add a convergence test
  of the jittered channel (N = 20/40/80 at t = 0.25 with exact ghosts as reference, ~30 s) — it fails today.

### F2 (medium, CONFIRMED) — Float32 3-D ExtrapolationBC(2): O(1) wrong ghost values on the CPU backends (compiler miscompile hit by the fit kernel)
`src/Grid/BoundaryConditions.jl:394-418` (`_fit_degree_b`) with `_cholesky_moment` (`Interpolators.jl:46`).
- Scenario (`ka5.jl`): unit cube, Δx = 0.1, jitter 0.2, R = 2.5Δx, quadratic data, `ExtrapolationBC(2)` on all faces.
  Max ghost error **7.09** (319 of 3079 ghosts > 1e-2) on Polyester, 7.09 (640 ghosts) on Threads, **1.5e-2 on KA CPU**
  (round-off). F64: 4e-12. Ghost 713: kernel value −3.70, exact 3.38; the Float32 normal equations solved on the host
  give 3.3827.
- Root cause: the moment matrix accumulated in the same function as the Cholesky solve comes out non-symmetric (row 1,
  columns 5–7 off by 0.30) although every term `w * (p * p')` is symmetric. `/tmp/brev_b/repro/repro6.jl` reproduces it
  with StaticArrays and LinearAlgebra only (15 data points, 10×10 Float32 system): value 0.7385 instead of 3.3828 at
  `-O2` (native znver5 and `-C x86-64-v3`), correct at `-O1` and `-C x86-64`. It depends on the code shape (a closure
  basis, a plain `cholesky` or dropping `_min_image` make it disappear; Threads and Polyester give different wrong
  values), so it is a Julia 1.13.3 / LLVM 20 optimisation bug, not a defect of the formula; but this kernel triggers
  it in a supported configuration and no test covers Float32 or 3-D. Julia 1.12 (the minimum, CI's other version)
  is not installed here, so whether it is 1.13-only is unknown.
- MirrorBC with the same data is fine (1.3e-4) in this case; the existing MLS paths (`_mls_divergence`, Conservation)
  use the same accumulate-then-`_cholesky_moment` shape — worth a Float32 3-D check.
- Fix: report upstream with `repro6.jl`; add a Float32 3-D exactness test of the ghost fit (it fails today on 1.13.3
  and would guard against a regression either way); if a workaround is wanted, a shape that does not trigger it
  (e.g. symmetrise `N_s` before the factorisation, or accumulate in a separate `@noinline` function) — verify with the
  test, since the trigger is fragile.

### F3 (medium, CONFIRMED) — the degree fallback is not monotone: a higher requested degree can give a lower one (1-D: degree 4 → 1, degree 3 → 3)
`src/Grid/BoundaryConditions.jl:322-341`. The inward donor move is attempted only with the support needed for the
*requested* degree Q; if no interior neighbour carries `_fit_support(D, Q)`, the original donor is kept and the degree
falls to what that donor supports, which at a wall is far below what a neighbour would carry.
- 1-D, R = 3.5Δx: the wall-most donor has 3 interior neighbours (n_sup = 4 → q = 1); its neighbours have at most 7 < 8
  (needed for q = 4), so ExtrapolationBC(4)/(5) run at **degree 1**, while ExtrapolationBC(3) finds a donor with 6 and
  runs at degree 3 (`diag.out`). MUSCL5, bump leaving the domain (`num1d.out`, lattice, N = 50…400):
  Extrap(3) 1.1e-2 … 1.13e-6 (EOC 4.3), **Extrap(4) 2.9e-2 … 3.3e-4 (EOC 2.0, identical to Extrap(1))**; at
  R = 4.5Δx Extrap(4) reaches EOC 4.9 (1.77e-6 vs OutflowBC 8.8e-3 at N = 400). MirrorBC(4) at R = 3.5Δx: degrees 1–3
  mixed. The commit's "in 1-D degree 4 reduces the error 10–100×" therefore holds for R ≥ 4.5Δx only (R is not stated).
- Fix: loop q = Q, Q−1, …: take the nearest donor (the original or an interior neighbour) with
  `_support_count ≥ _fit_support(D, q)` and stop at the first q that has one; state in the docs that degree q needs
  R ≳ (q + 0.5)Δx in 1-D.

### F4 (low/medium, CONFIRMED effect, SUSPECTED cause) — high-degree extrapolation loses accuracy in Float32
`src/Grid/BoundaryConditions.jl:248` (`invL = 1 / minimum(dx)`), `:400` (basis of `r * invL`), `:414`.
Independent of F2 (measured at `-O1`, `exact_f32_O1.out`): 2-D ExtrapolationBC(3), R = 3.5Δx, jitter 0.2: max
relative error **9.1e-2** on cubic data, 186 of 408 ghosts above 1e-3, growing with the wall distance (φ < Δx: 7e-4,
φ > 3Δx: 0.18); 3-D degree 2: 2.9e-3 (56 ghosts > 1e-3). F64 2.3e-10, MirrorBC (interpolation) ≤ 4e-5.
- Suspected cause: the data lie up to |x_t − x_j| + R ≈ 7Δx from the target, so with a basis scaled by 1/Δx the
  degree-3 columns are ~(7)³ larger than the constant one; the pivot test (`_pivot_tol(Float32) = 1.2e-6`) fails and the
  regularisation `3.5e-4 · tr/B` biases the fit.
- Fix: scale the basis by the stencil size seen from the target (`1 / (|x_t − x_j| + R)`), which equilibrates the
  moments; or document that ExtrapolationBC degree ≥ 3 needs Float64. A Float32 exactness test would show it.

### F5 (low/medium, CONFIRMED) — degrees 4–5 in 3-D take an allocating, host-only Cholesky; degree 3 in 3-D takes 30 min to compile on KA
`src/Grid/BoundaryConditions.jl:191-197, 228` accept degree 0–5 for every D, and `_fit_eval` (`:369-379`) compiles
every degree from Q down. In 3-D degree 4/5 have 35/56 unknowns; StaticArrays' `_cholesky` hands sizes > 24 to
`_cholesky_large` (`cholesky.jl:27-63`), which converts to a `Matrix` and calls LAPACK: an allocation in the kernel
body on the CPU and not compilable on a GPU (kernel rule 1). The rest of the package stops at degree 3 in 3-D
(`MUSCL.jl:10`). Degree 3 in 3-D (20 unknowns) compiles in 12.9 s on Polyester, the KA CPU kernel took
**1798 s (30 min)** to compile on the first launch (`ka3d3.out`; the 2-D/3-D degree ≤ 2 kernels compile in seconds); not measured on a GPU, where
a kernel of that size is likely just as costly.
- Fix: cap the degree per dimension (e.g. `Val(min(Q, D == 3 ? 2 or 3 : 5))` at the launch, or reject Q > 3 for D = 3 in
  the constructor/at grid construction) and document it.

### F6 (low, CONFIRMED by construction) — corner ghosts get a diagonal normal and one operator application
`src/Grid/BoundaryConditions.jl:300, 358`; normals from `wall_geometry` (`Domains.jl:304`). A ghost outside a box
corner has the corner as its nearest wall point, so `n = (±1, ±1)/√2` (test "outside a corner: Euclidean"), the mirror
point is the point reflection, and `op` is called once with the diagonal normal. For the even reflection (no op) and
for scalar odd ops that is right; for the slip wall the docs give as the example (`reflect_velocity(eq, U, w.normal)`)
it maps (u, v) → (−v, −u) instead of the double reflection (−u, −v). The ghost also belongs to one face's tag only, so
with different conditions on the two faces only one of them acts. These ghosts are in the stencils of the corner
particles only.
- Fix: document the corner behaviour in the MirrorBC docstring (ops must handle a corner normal), or give the corner
  ghosts two reflections (e.g. a `WallPoint` with the per-face normals when the nearest point is an edge/corner).

### F7 (low, docs, CONFIRMED)
- `boundary_conditions.md:13`: "Degree 0 is a donor copy like `OutflowBC`" — the docstring (`BoundaryConditions.jl:177`)
  and the test (`e0 > 0.05`, convex combination) say it is the weighted average of the donor and its interior
  neighbours.
- `gpu_roadmap.md:30`: "Taylor fit" — the kernel is a target-centred weighted MLS (the docs page explicitly says the
  donor-centred Taylor fit was abandoned).
- `boundary_conditions.md:32/34`: "MUSCL3 and MUSCL5, degrees 1–3" and "degree 2 and 3 … with MUSCL3": under MUSCL3 the
  donor-order cap (`particle_orders` is filled with `max_order = 3` every step, `TimeSteppers.jl:545`) makes degree 3
  run as degree 2, so those MUSCL3 degree-3 rows are degree-2 results.
- `boundary_conditions.md:24`: "0.2–0.5 ms per application on a 200² grid" holds for degree 1–2; measured on 2 threads
  (`ka3_cost.out`, 42.7k particles, 2.7k ghosts): OutflowBC 0.16 ms, ExtrapolationBC(1) 0.34, MirrorBC(1) 0.53,
  MirrorBC(2) 0.86, and at R = 3.5Δx ExtrapolationBC(3) 1.9 ms, MirrorBC(3) **2.9 ms** (×15 OutflowBC, three times per
  RK3 step).
- The 1-D degree-4 statement needs its radius (F3).

## Test coverage

Covered well: degree-2 exactness at every 2-D ghost incl. corners and the outermost layer (MirrorBC with/without an
operator, ExtrapolationBC), degree 1 vs 0, the bound, the MOOD-order cap, the no-donor fallback, KA vs Polyester and
Threads (2-D, F64), zero allocation per functor, the shape requirement, the constructor checks, a short channel run.
All of it is new API, so every test fails on the code before the change (and the memory-budget loop includes the two
new arrays).

Missing / weak:
- Only 2-D Float64. No 1-D or 3-D, no Float32 (would catch F2 and F4), no curved wall (the probe shows balls are exact
  at the mirror point — cheap to add), no periodic axis with a mirror.
- The channel test runs N = 20 only and checks boundedness and one comparison with OutflowBC; a three-resolution
  jittered run with exact ghosts as reference would catch F1.
- No test that a requested degree is actually reached (F3), e.g. 1-D R = 3.5Δx ExtrapolationBC(4) on a quartic.
- No test that adaptivity-inserted particles get their wall geometry, or that the reordering permutes it with the
  particle (`adapt.jl`: both correct today, ~5 s to add).
- `e5 < 1.0` (`test/unit_core.jl:896`) is very loose for an error against an O(1) solution.

## Minor issues

- `requires_shape` is checked only in the `ParticleGrid` constructor; a `MirrorBC` added to `bc_map` later on a
  shape-less domain gets NaN mirror points, finds no donor and silently keeps the ghost states.
- On moving grids `normals`/`wall_distances` of moved interior particles go stale (never refreshed; harmless because only
  ghosts, which do not move, are read — but the field is documented as "per particle").
- Thin regions: a ghost deeper than half the local width has its mirror point beyond the opposite wall (the fit then
  extrapolates); the donor's neighbour list can also reach across a thin obstacle (as for every MLS stencil).
- Threads and Polyester are not bitwise identical in 3-D F64 (1.2e-13 for ExtrapolationBC(2)); the 2-D test asserts
  `==` and passes; a 3-D version of that test would need `≈`.
- The two arrays cost (D + 1)·sizeof(T) per particle (32 B in 3-D F64) and are uploaded to devices even when no
  condition reads them; registered in `particle_buffers`, so `memory_estimate`/`reserve!` are right.

Checked and fine: no ghost reads another ghost (the fit, the bound and the donor only read `!isb` particles, so the
order of tags and the `rhos_buffer`/stage/MOOD-candidate buffers are irrelevant; bitwise identical for three tag
orders with Mirror/Extrapolation/Outflow/Dirichlet); kernel arguments are bits (`_WallOp`, `Val`s, views); no Polyester
capture problems (all values passed through the argument tuple); `_min_image` with `L = Li = 0` on walled axes;
periodic channels exact; mirror points of the step-corner ghosts and the ball ghosts lie inside; InverseWeightFunction
with a mirror point on a particle exact (F64 5e-16); adaptivity/reordering/device adapt/`ensure_capacity!` carry the
new arrays; the 2-D outflow and channel stability numbers of the commit message reproduce.

## Verdict

The fit itself is sound: target-centred, interior-only, exact in Float64 for every dimension, jitter level, periodic
axis and curved wall probed; the kernel is allocation-free, backend-portable (KA parity at round-off) and race-free;
the bookkeeping of the two new per-particle arrays (constructor, adaptivity, reordering, device adapt, memory budget) is
complete; and the stability numbers of the commit message reproduce. But the main wall condition does not deliver
accuracy on jittered clouds: on the commit's own channel MirrorBC neither converges nor stays accurate in time and is
worse than OutflowBC at N = 80, because of the ghosts of the jittered wall row (F1). Float32 3-D ExtrapolationBC hits
a compiler miscompile with O(1) errors (F2, standalone reproducer provided); the degree fallback can drop a requested
degree 4 to 1 (F3); high-degree Float32 extrapolation is inaccurate (F4); 3-D degrees 4–5 allocate and 3-D degree 3
is impractical on KA (F5); corner ghosts get one reflection with a diagonal normal (F6); several docs statements need
precision (F7). Full suite 24 344/24 344.
Findings: 1 high (F1), 2 medium (F2, F3), 4 low (F4 low/medium, F5 low/medium, F6, F7 docs); all confirmed (F4 cause suspected).
