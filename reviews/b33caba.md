# Re-review: HyperCloudCore.jl fixes for the review of PRs #39/#40 (`1fcc2ce..b33caba`, branch `claude/volumes-review`)
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation (A2)

A re-review of five commits on `claude/volumes-review` that answer the A2 review of #39/#40
(`/tmp/brev3/r1/review.md`: F1 medium, F2-F4 low, minors). The commits are 1aed5e8 (`conservation_defect`
counts interior pairs only), 087cb73 (volume accuracy claims and a test with a non-harmonic function), 0135b4c
(docs on gathered ghosts, MOODu2 wall test, remeasured mirror errors), 14cce12 (note in `reviews/0a10453.md`) and
b33caba (`total_mass` change == `mass_residual` test). I read the code in the read-only worktree
`.claude/worktrees/rr-volumes` and changed nothing there and no git state. Probe runs used `git archive` copies
with the test manifest copied in: `/tmp/brev3/rr2/trees/new` (b33caba), `/tmp/brev3/rr2/trees/par` (1fcc2ce) and
`/tmp/brev3/rr2/trees/mask` (b33caba plus a probe-only ghost mask in `_rhs_gather!`, `probes/mask.diff`). I also
reused the earlier copies `/tmp/brev3/r1/old` (fbb54d1, before #39) and `/tmp/brev3/r1/mid` (a56944c, before #40).
Probes and outputs are in `/tmp/brev3/rr2/probes/`.

## Scope

- `src/Interpolation/Conservation.jl` (the `bnd` skips, the `wall` field, the docstring) and
  `docs/src/spatial/mls_interpolation.md`. I also looked for users of `conservation_defect` in HyperCloud.jl
  (all branches, `git grep`) and in `/workspace/julia/Simulations` (working tree, worktrees, all branches).
- `src/Grid/Volumes.jl` (header, `ShepardVolumes`/`MLSVolumes` docstrings), `docs/src/particle_grids/volumes.md`.
- `docs/src/particle_grids/boundary_conditions.md` (item 0 numbers, the OutflowBC/ExtrapolationBC sentence, the
  new paragraph on gathered ghosts), `docs/src/time_integration/mood.md`, `docs/src/time_integration/time_stepper.md`.
- Tests: `test/unit_core.jl:452-460` (bounded lattice in the defect testset), `:995-1018` (MOODu2 wall test),
  `test/unit_schemes.jl:1089-1111` (2-D quadrature), `:1290-1316` (adaptivity identity).
- `reviews/0a10453.md` (follow-up note).

## Tests run

- Full suite under the lock in the worktree (`flock /tmp/hcc_test.lock timeout 60m julia -t 8 --project=test
  test/runtests.jl`, at b33caba): **exit 0**. core 25016, schemes 434, pipeline 135, backends-a 25, backends-b 71; wall
  time 120 s (`probes/full_suite.log`). The worktree stayed clean (`git status` empty before and after). Against the
  earlier review run at 1fcc2ce: core +5 tests, 105.2 → 106.7 s; schemes +6 tests, 116.7 s both times.
- Probes (`julia -t 2 --project=<tree>/test`; `cost.jl` under the lock with `-t 8`):

| probe | what | result |
|:--|:--|:--|
| `vol.jl` (new) | 1-D [0,1] and [-1,1] x², eˣ, N = 40/80/160; 2-D unit square e^{x+y}, N = 20/40/80; Shepard, MLS0-2; Float64 and Float32 | every number of the new volumes table, the docstrings and the test comments reproduced (below) |
| `t2d.jl` (new, mid) | the new 2-D quadrature test as printed checks, Float64 and Float32 | passes on b33caba in both precisions; every check fails on a56944c |
| `cdef.jl` (par, new) | `conservation_defect`: bounded lattice (the test case), lattice with drops at a wall, jittered box, periodic jittered, channel, disc (lattice, jittered), 1-D | see F4 |
| `wallsum.jl` (new) | face vectors of the skipped interior→ghost slots, summed per wall tag; `Σ_j S_ij` per interior particle | see F4 |
| `moodt.jl` (old, new) | the new MOODu2 test verbatim, plus MOODu1(1e-4), MOODu2(1e-5), exact ghosts, N = 80, drops on both walls | see F3 |
| `cost.jl` + `cost_run.sh` (old, new, mask; lock, `-t 8`, each tree twice) | ms per step, OutflowBC boxes | see F2 |
| `chan.jl` (new, old) | Stability item 0: jittered channel (0.3Δx, seeds 7/1/2), off-lattice wall (0.1Δx), mirror vs exact ghosts, N = 20/40/80 | the new numbers are reproduced exactly with seed 7 (below) |
| `adapt.jl` (mid, new) | the b33caba adaptivity loop as printed checks | passes on b33caba; the new check fails on a56944c with an estimator (1.2e-3, 2.1e-3) |
| `mdtable.jl`, `mdlist.jl` | `volumes.md` / `boundary_conditions.md` parsed with Julia's `Markdown` stdlib (the parser Documenter uses) | see N1 |

Not run: the docs build (the inventory download may be blocked; I parsed the two changed pages with the stdlib
`Markdown` instead), GPU (no device), the HyperCloud.jl suite (no user of the changed function there).

## Earlier findings

### F1 (medium): harmonic test function, wrong orders in the docs — **FIXED**

`vol_new.out` reproduces every number now written down (R = 2.5h, relative error of Σ_int V f):

| where | claim | measured |
|:--|:--|:--|
| `volumes.md` table, 1-D unit interval, N = 40 | Shepard 1.4e-2 / 2.2e-3 (x² / eˣ), MLS2 3.9e-5 / 6.5e-6 | 1.37e-2 / 2.24e-3, 3.91e-5 / 6.48e-6 |
| table, 2-D e^{x+y}, N = 20/40/80 | Shepard 8.7/4.4/2.2e-3, MLS0 6.3/1.6/0.38e-4, MLS1 3.0/0.79/0.20e-3, MLS2 5.1/1.3/0.33e-5 | 8.74/4.35/2.17e-3, 6.30/1.55/0.383e-4, 3.02/0.786/0.200e-3, 5.12/1.30/0.325e-5 |
| `MLSVolumes` docstring, `Volumes.jl` header | order 2 for every P in 1-D and 2-D; P = 1 the least accurate on a box | ratios 3.9-4.1 per halving for MLS0-2 in 1-D and 2-D; MLS1 is 4.8× MLS0 and 59× MLS2 at N = 20 in 2-D (also the worst MLS in 1-D) |
| `ShepardVolumes` docstring | first order, 0.1-0.5 h relative, 1.4e-2 (x², [0,1]) and 4.4e-3 (e^{x+y}) at N = 40 | first order (ratio 2.00); 0.55 h (x², [0,1]), 0.09 h (eˣ), 0.17 h (2-D); see M1 |
| test comments | Shepard 8.7e-3 (was 0.43), MLS1 3.0e-3, MLS2 5.1e-5 → 1.3e-5 | 8.740e-3 (0.432 on a56944c), 3.020e-3, 5.123e-5 → 1.296e-5 |

The new test (`unit_schemes.jl:1089-1111`, `t2d_*.out`): Shepard 8.74e-3 < 1.2e-2 (margin 1.37), MLS1 3.02e-3 <
4e-3 (1.32), MLS2 5.12e-5 < 1e-4 (1.95), the order check 0.253 < 0.3. In Float32 the same values come out (Shepard
8.74e-3, MLS1 3.02e-3, MLS2 5.14e-5, ratio 0.263) and every check passes, so the bounds are not at round-off level.
On a56944c (ghosts in the partition) every one of these checks fails: Σ_int V = 0.563, Shepard 0.432, MLS1 5.2e-2,
MLS2 7.2e-3, ratio 0.496, and `total_mass` with ghost state 100 gives 3.73, not 3. The ghost-state check is the
discriminating version the earlier review asked for. On a regression of the order itself, the ratio check
catches anything below order 1.74. The wording "Up to 0.8" is now consistent across the changed files. One
regression in the rewritten `volumes.md` paragraph is N1 below.

### F2 (low): gather of every ghost, cost undocumented — **FIXED (documented; the decision holds)**

The fix documents the cost and says that skipping the ghosts no interior particle reads "saves nothing
measurable". I measured that claim with a probe-only mask (`mask.diff`: a ghost without an interior neighbour
returns before `update_content!`). It gives bitwise the same interior states as b33caba (checksums equal in all
five cases), so it is a valid alternative. `cost.out`, lock, `-t 8`, min ms/step, each tree run twice (another
agent's `-t 2` job was running on the machine, so ±3 % is noise):

| case (ghost share, unread ghosts) | fbb54d1 (no ghost gather) | b33caba | mask |
|:--|:--|:--|:--|
| 2-D N = 160 MUSCL3 (7 %, 656) | 2.79 / 2.65 | 2.66 / 2.77 | 3.09 / 2.74 |
| 2-D N = 80 MUSCL5 (13 %, 336) | 3.18 / 3.07 | 3.25 / 3.19 | 3.22 / 3.21 |
| 3-D N = 20 MUSCL3 (51 %, 3458) | 9.60 / 9.22 | 9.92 / 9.95 | 9.77 / 9.83 |
| 3-D N = 32 MUSCL3 (38 %, 7778) | 38.0 / 38.1 | 40.7 / 39.0 | 39.1 / 38.8 |
| 3-D N = 16 MUSCL5, R = 3h (59 %, 896) | 29.5 / 29.5 | 35.3 / 34.7 | 34.3 / 33.8 |

The documented "29 → 35 ms, up to about 20 %" in 3-D and "a few percent" in 2-D are right (2-D: 0-5 %). The mask
saves 1-3 % in 3-D (N = 16 MUSCL5: 35.0 → 34.1 ms, N = 32: 39.9 → 38.9, N = 20: 9.93 → 9.80), at the edge of the noise, so "saves nothing measurable" is fair. The parenthesis "(a third of the
band)" is not right for the very case quoted: with R = 3h only 13 % of the ghosts (896 of 6970) are unread, so
the mask has little to skip there (M2).

### F3 (low): MOODu2 at walls changed, undocumented and untested — **FIXED**

`mood.md` and `boundary_conditions.md` now say that ghosts carry gradients, curvatures and possibly lowered orders,
and that MOODu2 acted as MOODu1 on the wall rows before. The test discriminates (`moodt.out`, the test's
`run_lattice` verbatim):

| criterion, condition, N | fbb54d1: err, drops on the bottom wall rows | b33caba |
|:--|:--|:--|
| MOODu2(1e-4), Mirror2, 40 (the test) | 0.209, 600 (4.17/step) → **test fails** | 0.0683, 0 → passes |
| MOODu2(1e-4), Mirror2, 80 | 0.0743, 1679 (5.83/step) | 0.0103, 0 |
| MOODu2(1e-4), exact, 40 / 80 | 0.0489, 81 / 0.0141, 230 | 0.0474, 0 / 0.0087, 0 |
| MOODu1(1e-4), Mirror2, 40 | 0.209, 601 | **0.213, 564 → the same test would fail** |

So the test does not only check the gather. It checks that MOODu2 differs from MOODu1 at the wall, which is what
the docs claim. Its numbers (4.2 drops/step, error 0.21 → 0.068) and the docs' "4-6 times per step (N = 40/80)" and
"5.8 order drops per step ... 7.4e-2 before, no drops and 1.0e-2 now" are reproduced. The top wall rows never drop
in this setup (the Gaussian sits on the bottom wall), so counting only `y < 2.5h` loses nothing.

### F4 (low): `conservation_defect` next to walls — **FIXED**

The function now skips ghost rows and interior→ghost slots and counts the latter in a new `wall = (; pairs)`
field. `cdef.out`, par → new:

| grid | total.rel | mixed pairs | max particle defect | wall slots |
|:--|:--|:--|:--|:--|
| bounded 20² lattice (the test) | 0.172 → **4.67e-3** | 764 → 0 | 0.661 → **0.0755** | 904 |
| the same, order 1 on 6 wall particles | 0.219 → 0.0625 | 1334 → 221 | 1.14 → 1.17 | 904 |
| jittered box 0.3 | 0.619 → 0.499 | 753 → 0 | 0.965 → 0.908 | 800 |
| periodic jittered 0.3 | 0.4998 → 0.4998 (identical) | 0 → 0 | 0.832 → 0.832 | 0 |
| channel (periodic x) | 0.0872 → 2.18e-3 | 320 → 0 | 0.394 → 0.0173 | 440 |
| disc n = 40 / jittered | 0.112 → 0.0217 / 0.552 → 0.486 | 1060 → 0 / 1162 → 0 | 0.601 → 0.193 / 0.914 → 0.908 | 1432 / 1315 |
| 1-D [0,1], N = 40 | 0.0804 → 2.51e-3 | 4 → 0 | 0.524 → 0.0472 | 6 |

The commit's numbers (0.17 → 4.7e-3, 0.66 → 0.075) are reproduced. The old "mixed" pairs on lattices were
interior–ghost pairs (a ghost has fewer neighbours, hence another degree), which is gone. Periodic grids are
unchanged (no boundary particles). A mixed-degree drop next to a wall is still reported (the second row). No NaN,
and every ghost reports 0.

Correctness of the skip: the face vectors satisfy `Σ_j S_ij = 0` per particle to 3e-13 relative (`wallsum.out`),
so `Σ_wall slots S = −Σ_interior pairs (S_ij + S_ji)`. The skipped slots are therefore exactly the boundary part
of the budget, and nothing is lost. Their per-wall sums approximate the outward `∫ n dA` (lattice N = 80:
(−1.023, 0), (1.019, 0.003), (0.003, −1.019), (0, 1.016)), with deviations of the size of the near-wall pair
defects. Jittered walls (the jittered box and disc) and moving pieces behave the same: the piece ghosts are
`is_boundary` particles (`MovingBoundaries.jl:33,59,191`) with zero volume, so they count as `wall`. I checked the
moving pieces by reading only.

API: `conservation_defect` is not exported (it is called as `HyperCloudCore.conservation_defect` and is in the
docs through `@docs`). The NamedTuple gains a trailing field, so field access and `(; total) = r` keep working, but
`total`/`equal`/`particle` change meaning on bounded grids (now interior pairs only). The docstring says so. There
are no users outside Core: HyperCloud.jl has none on any branch, and the Simulations hits (`defect_stats.jl`,
`run_defect.jl`, `mass_table.jl`) are an unrelated `conservation_defect!` statistic, the time-integrated mass
budget.

### Earlier minor issues

- Version labels: **fixed** ("up to 0.8"/"in 0.8" everywhere in the changed text; `Project.toml` 0.8.0).
- Stability item 0 pre-#39 numbers: **fixed**. `chan.out` reproduces the new numbers exactly with jitter 0.3 and
  seed 7: mirror 0.273/0.0631/0.00999, exact 0.154/0.0429/0.00836; off the lattice at t = 1 0.378/0.115/0.0195,
  exact 0.230/0.0744/0.0158. Seeds 1 and 2 agree within 10 %. The "before" values in the parenthesis
  (0.28/0.062/0.023) differ slightly from fbb54d1 with the same seed (0.279/0.060/**0.029**; 0.032 and 0.023 for
  seeds 1/2). They were probably measured at an older commit, and the off-lattice "before" values match
  (0.364/0.0953/0.0245). Not worth a change.
- `muscl.md` note on ghosts in the pre-gather: **not fixed**, but `boundary_conditions.md` now covers it (starved
  ghosts and EPD1). Acceptable.
- D1 paragraph inside the numbered list: **not fixed**, and now two paragraphs sit between items 0 and 1. It
  renders (stdlib parse: a list starting at 0, then the paragraphs, then a list starting at 1), but reads as part
  of item 0.
- `reviews/0a10453.md` D1 note: **fixed** (the numbers match the earlier review's table).
- Missing tests named in the earlier coverage section: the ghost-state `total_mass` check and the adaptivity
  identity are **added**. `adapt.out`: the new check passes on b33caba (|Δtotal − residual| = 0 in all four cases)
  and fails on a56944c with an estimator (1.17e-3 in 1-D, 2.12e-3 in 2-D). `total_mass == interior_mass` is exact
  equality, which is robust: both are `mapreduce` over the same `1:N`, so the blocking is the same, and the ghost
  terms are ±0.

## Findings (new)

### N1 (low, CONFIRMED): `volumes.md` loses a paragraph in the rendered docs — the text after the new table is glued onto its last row
`docs/src/particle_grids/volumes.md:25`. 087cb73 inserted the table in the middle of the old paragraph, and the
rest of that paragraph is on the same line as the last table row: `| 2-D, $e^{x + y}$ | … | $5.1 \to 1.3 \to
0.33 \cdot 10^{-5}$ | (Up to 0.8 the ghosts got volumes too … MLS$(P \ge 1)$ converges there at order 2–4.` Markdown
reads it as a sixth cell of a five-column table and drops it. With Julia's `Markdown` stdlib (`mdtable.out`) the
last row has 5 cells, and neither "ghosts got volumes too" nor "Shepard on a jittered periodic cloud" appears
anywhere in the parsed page. Both are present in the parsed parent page. So the rendered docs lose the
history note and the only statement of Shepard's and MLS's accuracy on jittered periodic clouds.

- Fix: put a blank line after the table row and start the parenthesis as its own paragraph.

## Test coverage

- The new tests fail where they should: the 2-D quadrature and ghost-state checks on a56944c, the MOODu2 test on
  fbb54d1 (and with MOODu1 on b33caba), the adaptivity identity on a56944c, and the bounded-lattice defect test on
  1fcc2ce (no `wall` field; total.rel 0.17 > 1e-2). They pass with margins of 1.3-2× and, for the volumes, in
  Float32 too.
- Not covered: `wall.pairs` is only checked to be > 0. A check against the count of interior→ghost slots would be
  stricter, but the value is a plain counter. `conservation_defect` on a jittered bounded cloud and with a MOOD
  drop at a wall (the second row of the F4 table) are untested, and they behave sensibly.
- The MOODu2 test is the first 2-D `MOOD{EPD1, MOODu2}` stepper in the `core` group, so it compiles one new stepper
  (the other MOODu2 steppers are 1-D in `backends-a`, or in `pipeline`). The measured cost is small: the core group
  takes 105.2 → 106.7 s, and the suite wall time is unchanged at 120 s. Acceptable under the compile budget of
  CLAUDE.md.

## Minor issues

- M1: "relative errors of about 0.1-0.5 h" for Shepard (volumes.md, `ShepardVolumes` docstring, `Volumes.jl`
  header) depends on f and on the domain, since a relative error is not scale-free. The unit interval gives 0.55 h
  for x², and [-1, 1], the domain of the 1-D test in the same file, gives 1.1 h for x² (5.5e-2 at N = 40). "About
  0.1-1 h depending on f" or "of order h" would be safer.
- M2: `boundary_conditions.md` (new paragraph): "(a third of the band)" holds for R = 2.5h (34-37 %), not for the
  quoted 3-D MUSCL5 case with R = 3h (13 %). The mask saves about 3 % there.
- M3: the 1-D row of the volumes table leaves MLS0/MLS1 empty. Measured at N = 40 on [0,1]: 4.7e-4 / 7.9e-5 and
  2.7e-3 / 4.5e-4. They would show that MLS1 is the weakest MLS in 1-D as well, as the docstring says.

## Verdict

All four earlier findings are fixed, with numbers I could reproduce to the stated digits. F1: a non-harmonic
function, corrected orders and docs, a test that fails on the old partition and passes in Float32. F2: the cost is
documented correctly, and the mask alternative really saves only 1-3 %. F3: documented, and tested with a check
that separates MOODu2 from MOODu1. F4: interior pairs only, which is the right split since `Σ_j S_ij = 0`, the
periodic case is unchanged, and there are no users outside Core. The new tests discriminate and are not fragile.
One regression of the docs edit: a paragraph of `volumes.md` disappears from the rendered page (N1).

Earlier findings: 4 FIXED (F1, F2, F3, F4), 0 partly fixed, 0 not fixed. Earlier minors: 4 fixed, 2 not fixed (cosmetic).
New findings: 0 high, 0 medium, 1 low (N1, confirmed); 3 minor issues (M1-M3).
