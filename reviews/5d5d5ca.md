# Third re-review: HyperCloudCore.jl — moving-boundary fixes after the second re-review, 8d41d25..5d5d5ca
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation (A2)

Independent re-review of commit 5d5d5ca ("Moving boundaries: insertions in one round, wall rows back", on top of
8d41d25) in the read-only worktree `HyperCloudCore.jl/.claude/worktrees/rr-moving3` (called *v3* below), against
8d41d25 (`rr-moving2`, *v2*), 405f797 (`rr-moving`, *prev*) and the original parent 1fcc2ce (`review-mb`, *parent*).
Earlier reviews: `/tmp/brev3/r2/review.md` (H1–H3, M1–M5, L1–L3), `/tmp/brev3/rr1/review.md` (N1–N11),
`/tmp/brev3/rr4/review.md` (N12–N15 + minors). Probes and logs: `/tmp/brev3/rr5/probes/` (`logs/<probe>.v3.log`,
`.v2.log` where run on 8d41d25; `run_on.sh` shows the invocation, `julia -t 2 --project=<worktree>/test`), full-suite
log `/tmp/brev3/rr5/full_suite.log`. No file in any worktree was edited, no git state was changed.

Four probes redefine functions **inside the probe process only** to attribute effects:
`o1_variants.jl`/`o2_periodic.jl`/`o3_plate_variants.jl` with `VARIANT=nowall` (insertion additionally requires
`signed_distance(region, x) <= -gap`, i.e. v2's no-wall-row rule) and `VARIANT=rounds` (insertion events split into
rounds of `_spaced` events again, v2's scheme, wall rows kept); `o_chunks.jl` (`VARIANT=chunks`, `*_chunks.jl`):
insertions as joint events (all candidates within R of a seed in one `_AdaptEvent` with several `added`, so they
share `S_new` in the quadrature), the joint events in rounds.

## Scope

`git show 5d5d5ca` (+48/−38), read completely:
- `src/Grid/MovingBoundaries.jl`: `sweep_boundaries!` (drop/insert flow, `n_inside` count, one insertion round),
  `_covered_particles` (composite distance `> -clear` and a piece's own `< clear`), `_sweep_lattice` (tolerance
  `max(1e-6, 16 eps(T) |r|)`), `_uncovered_points` (`_is_interior`, `gap <= min_p |sd_p| <= reach`, dedup of candidates
  by wrapped distance).
- `src/Builders/domains.jl` (`Meta.parseall`, `_toplevel_to_block`, `sprint(showerror, …)`).
- `docs/src/particle_grids/moving_boundaries.md`, docstring of `sweep_boundaries!`, tests `test/unit_core.jl` (+1),
  `test/unit_schemes.jl` (1 changed).

Context read: the commit message, repo `CLAUDE.md`, the whole of `MovingBoundaries.jl`, `_remap_event!`,
`_dmp_repair!`, `_apply_topology!`, `_slots_after_events`, `_detect_events` (`Adaptivity.jl`), `get_points` and
`_wall_index` (`Domains.jl`).

## Tests run

- Full suite under the lock: `cd rr-moving3 && flock /tmp/hcc_test.lock timeout 60m julia -t 8 --project=test
  test/runtests.jl` → **exit 0** in 125 s: core 25015/25015, schemes 452/452, pipeline 135/135, backends-a 25/25,
  backends-b 71/71.
- All rr4 probes re-run on v3 (except the four that override functions of v2: `q15b`, `q16d`, `r8b`, `r8d`, `r8e`),
  compared line by line with rr4's v2 logs. New probes:

| probe | what | v3 | v2 (8d41d25) |
|:--|:--|:--|:--|
| `t1_insert_onesweep.jl` | one sweep behind a receding plane piston (fluid at rest, no PDE) uncovering 1/2/3 columns; survivors' ΔU, V/V0 by distance; inserted V/h² vs a cloud generated at the pose | see N16 | |
| `t2_recede_fast.jl` | receding piston 0.4…3 h per sweep, 12 h total | no NaN; volumes N16 | NaN at 3 h, periodic |
| `t3_f32_lattice.jl`, `t3b_lattice_missing.jl` | `_sweep_lattice` over the whole canvas vs the generated cloud, Float32/Float64 | last row missing at some N (N17) | same |
| `o1_variants.jl` | receding co-moving piston along walls, `CASE=p14` (r2's data), `r8` (consistent data), `r8c` (ExtrapolationBC face); base/nowall/rounds; N = 20…160 | table in N16 | |
| `o2_periodic.jl` | p6's channel periodic in y: error and where the extreme volumes are; base/rounds | N16 | |
| `o3_plate_variants.jl` | the docs' plate case 1 + ½cos 5π(x−t); base/rounds/nowall | N16 | |
| `*_chunks.jl` | the three above + t1 with joint insertion events in rounds (probe-only) | N16 (fix sketch) | |

Not run: device tests (no GPU), the docs build, HyperCloud.jl against this tree, an Euler system.

## Status of the earlier findings

### rr4 (N12–N15) and its minors

| finding | status | evidence (this review) |
|:--|:--|:--|
| N12 non-convergent error behind a receding co-moving face | **fixed in the measured case, by a change that breaks the volumes elsewhere (N16); root cause not established** | `r8_n2loc` [3,6) h band: 0.122 / 0.087 / 0.052 / **0.0141** (N = 20…160; v2 0.064 / 0.045 / 0.059 / 0.064, parent 0.098 / 0.087 / 0.053 / 0.013). `q15` uncovered part 0.122 / 0.087 / 0.036 = the docs' numbers. But the same case with correct inserted volumes (joint events, `r8 chunks`) is non-convergent again: 0.217 / 0.086 / 0.062 / **0.075**. The convergence comes with inserted volumes 0.69–1.68 h² (`r9_vols`, N16), so "the rounds caused N12" is not shown: what one round changes is that each insertion over-claims its share. |
| N13 corner error / lost wall rows / curved walls | **fixed** | `r8c` (ExtrapolationBC face, overall max): 0.243 / 0.138 / 0.058 / **0.0154** (v2 0.121 / 0.193 / 0.191 / 0.319). `q4b_osc`: 220 interior particles at the initial pose after every period (v2 208). `r3c`: 1 wall-row particle lost in the upper chamber (v2 9). `r4_curved`: 6 particles less than h/2 inside the arc, largest arc gap 1.08 h (v2 0 / 1.42 h; generated 0.65 h). Wall-row volumes: see N16. `nowall` changes the receding cases by < 0.02 (`o1`), so the wall rows are neither the cause of N12 nor of N16. |
| N14 removal where a piece's surface is no wall | **fixed** (small leftover) | `r3b`: no more remove/re-insert churn at the plane in the upper chamber (−2 +0 per crossing, was −19 +19). `r3`: error there 0.016 after 40 sweeps (v2 0.030, prev 0.0061); one top wall-row particle at (0.8, 1.0) is still removed and not re-inserted. |
| N15 static ghosts never removed | **not fixed, documented** | `r10`: 74 static ghosts inside the piston at the inner turning point (v2 62, more wall ghosts are added now), 62 not in a generated cloud, Δn up to 1.17. The new Limits line documents it. |
| minor: docstring stale | **partly fixed** | Insertion text updated; still "A piece whose box reaches across a periodic end is refused" (`MovingBoundaries.jl:190`) and in the docs' Limits; the summary paragraph (`:166-169`) omits "within reach of a moving piece". |
| minor/N6: Float32 lattice tolerance | **partly fixed** | the index range now has a relative tolerance, but the canvas filter after it drops the same row (N17). |
| minor: `_check_periodic_pieces` piece entirely past the seam | **not fixed** | code unchanged (`:301-305`). |
| minor: string shapes | **fixed** | `r7`: newline-separated statements, `using`, functions work in Float32/Float64; errors read `ArgumentError: … not valid Julia code: ParseError: # Error @ none:1:19 …`. Syntax errors are now `ArgumentError` instead of `ParseError` (behaviour change, harmless). |
| minor: warning count | **fixed** | `n_inside = n_dropped + inside among the first candidates` (`:235-239`). |
| minor: initial-cloud corner tie | **not fixed (pre-existing)** | `q12`: 2 particles inside before a sweep, both lattice points of the initial cloud on the face. |
| minor: refresh only when ghosts were added | unchanged | docs sentence unchanged. |

### Earlier reviews (r2, rr1): did the new insertion step bring anything back?

| finding | status on v3 | evidence |
|:--|:--|:--|
| H1 far-field changes of one sweep (removals) | **still fixed** | `p10` row removal: beyond 4 h \|ΔU\| ≤ 0.008, V/V0 0.98–1.01 (same as v2); `p20` 3-D [0.9953, 2.0047]; `p9` unchanged. |
| H1 for **insertions** ("insertions are not seen by later events either", r2) | **back** | N16. |
| H2 coincident particles at periodic seams, Σ V vs region | **still fixed** | `p7`: min pair distance 1.000 h every step, +10 per column; `t1` periodic: min pair 0.5 h; Σ V = region to 1e-6 at the documented step (`p6`, `p14`, `q9`: 0.749999 / 0.75); the unit test passes. |
| N1 NaN for fast pieces | **still fixed** | `q2`, `q2b`: no NaN at 0.5–3 h per sweep. `t2` (receding 3 h per sweep, periodic): no NaN on v3, **NaN on v2**. Particles with V ≈ 1e-241 remain at ≥ 1.5 h per sweep on both (minors). |
| N2 wall-row error | **error gone, volume defect back and larger** | `o1 r8` converges with and without wall rows; inserted wall-row volumes alternate 0.457 / 0.752 h² (`q16_wallrow_vol`; generated 0.97, rr1's complaint was 0.59). |
| N5 wall-row particles next to an advancing face | **still fixed** | `q12`: closest 0.40 h. |
| N11 insertions near the piece only | **still fixed** | `q14`: 13 insertions 1–4 h from the face at 45 % jitter. |
| N3, N7–N10 | unchanged | `q13`, `q4a` (swept and generated region now identical sets, 436/436), `q6`, `q7` (8/64/256 sub-steps), KA parity bitwise (`p18`). |

## Findings

### N16 (high, CONFIRMED, regression of 5d5d5ca) — all insertions of a sweep in one round: overlapping remaps over-claim, inserted particles get up to 2.7 h², survivors are drained to 0, and several cases fall back to the parent's errors
`MovingBoundaries.jl:259-261` runs every candidate of a sweep as one `_AdaptEvent` each in a single `_sweep_round!`,
with candidates only `min_gap` (0.5 h) apart, not the event spacing. `_remap_event!` processes the events one after
the other on the same topology: each reads the volumes the earlier ones wrote, but its quadrature (`Adaptivity.jl:
486-536`) puts only its own new particle into `S_new`. Every inserted particle therefore takes its share as if it
were alone (including the share its neighbours in the same column also take), and the old particles lose that share
once per overlapping event; the residual correction keeps the local sum, so Σ V stays exact while the distribution
is wrong. This is r2's H1 remark "insertions are not seen by later events either", which the rounds of 405f797 had
fixed.
- **One sweep** (`t1_insert_onesweep`, h = 1/40, fluid at rest, smooth data, no PDE):
  - 1 column (0.6 h): inserted V/h² 1.21 (periodic) / 0.91–1.21 (walls) vs 1.015 / 0.97 generated; survivors 0–2 h
    from the face V/V0 0.68 (v2 0.90).
  - 2 columns (1.6 h): inserted **1.69–2.69 h²**; survivors 2–4 h V/V0 down to **0.000**, \|ΔU\| **0.17–0.27 up to
    12 h** from the face (v2: inserted 0.79–1.13, survivors ≥ 0.76, \|ΔU\| ≤ 0.03, nothing beyond 6 h). Two columns per
    sweep need |w|dt > h, beyond the documented limit, but one column already shows the bias, and it compounds:
- **Over many sweeps at the documented step** (0.4 h per step, one column every 2–3 sweeps), each new column drains
  the previous one: a persistent alternating pattern across the whole uncovered part.
  - `o2_periodic` (p6's channel periodic in y, receding co-moving, Gaussian): V/h² in **[0.23, 1.68]**, every odd
    column < 0.8, every even > 1.2, at N = 20/40/80 alike; v2 / rounds [0.75, 1.24].
  - `r9_vols` (walled channel, N12's case): V/h² **0.69–1.32** everywhere in the uncovered part, 1.53 / 1.68 in the
    two columns next to the face (v2 1.000–1.085, 1.16 / 1.24). `q16_wallrow_vol`: inserted wall rows 0.457 / 0.752
    alternating (generated 0.97).
  - `t2_recede_fast` at 0.4 h per step: off-wall V/h² [0.18, 1.70] (v2 [0.70, 1.55]).
- **Accuracy** (MUSCL3 + Rusanov, RK3-SSP, MOOD, t = 0.5):

| case | v3 | `VARIANT=rounds` (in-process) | v2 | parent |
|:--|:--|:--|:--|:--|
| p6: periodic in y, receding co-moving, Gaussian, N = 20/40/80 | **0.505 / 0.588 / 0.298** | 0.137 / 0.293 / 0.189 | 0.137 / 0.293 / 0.189 | 0.53 / 0.59 / 0.30 (r2, with the seam duplicates) |
| p14: walls, r2's data, N = 20/40/80/160 | **0.234 / 0.264 / 0.191 / 0.111** | 0.087 / 0.127 / 0.127 / 0.087 | 0.076 / 0.118 / 0.125 / – | 0.233 / 0.254 / 0.190 |
| plate, 1 + ½cos 5π(x−t), within 12 h of the plate (docs case `q10`) | **0.345 / 0.336 / 0.304** | 0.338 / 0.200 / 0.146 | 0.338 / 0.200 / 0.146 | 0.358 / 0.459 / 1.475 |
| plate with structure at the plate (`p11`), 3–12 h | 0.027 / 0.051 / **0.082** | – | 0.027 / 0.031 / 0.049 | – |
| N12 case `r8`, [3,6) h band | 0.122 / 0.087 / 0.052 / 0.014 | 0.072 / 0.048 / 0.061 / **0.063** | 0.064 / 0.045 / 0.059 / 0.064 | 0.098 / 0.087 / 0.053 / 0.013 |

  `VARIANT=nowall` reproduces v3 within 0.02 in every row (`o1`, `o3`): the wall rows are not involved. The docs'
  plate line "next to the plate 0.34 / 0.20 / 0.15" no longer reproduces (0.345 / 0.336 / 0.304: non-convergent next
  to the plate).
- **What a fix needs.** Joint insertion events (`*_chunks.jl`: all candidates within R of a seed in one event, the
  joint events in rounds) give correct volumes and v2's accuracy in the cases above (p6 V/h² [0.75, 1.24], errors
  0.18 / 0.32 / 0.21; plate 0.337 / 0.208 / 0.162; p14 0.089 / 0.127 / 0.132 / 0.088; t1 2 columns: inserted
  0.73–1.08 h², survivors ≥ 0.76, \|ΔU\| ≤ 0.03), **but** N12's case is non-convergent again with them (0.217 / 0.086
  / 0.062 / 0.075, ExtrapolationBC face 0.094 / 0.110 / 0.150). So N12 is not caused by the rounds as such; the one
  round only hides it there through the over-claimed volumes. 5d5d5ca trades N12's case for H1-type volume errors in
  the periodic channel, the plate and r2's data.
- Fix: revert to correctly coupled insertions (rounds, or joint events that share `S_new` as sketched in
  `o_chunks.jl`), and look for N12's cause elsewhere (rr4 located the error at y ≈ 0.3, 3–6 h from the face and
  positive; candidates: the reconstruction/DMP repair of the remap when the donor column is the one the region pass
  just enlarged, or the column-by-column insertion order relative to the face). Add a test that fails on v3: e.g.
  the periodic receding piston of `test/unit_schemes.jl:996` with `0.5 < V/h² < 1.5` for all interior particles after
  20 sweeps (v3: 0.23–1.68), or the inserted volumes of one sweep against 0.8–1.2 h². Correct the docs' sentence "(in
  rounds as well, the error behind a receding piston stopped converging)" and the plate numbers.

### N17 (low, CONFIRMED, pre-existing; N6 only partly fixed) — the canvas filter of `_sweep_lattice` drops the last lattice row when `canvas_min + n·dx` exceeds `canvas_max` by one ulp
The new tolerance keeps the last index in the range (`MovingBoundaries.jl:364-368`), but the point is then built as
`canvas_mins + j*dx` and rejected by `all(cd.canvas_mins .<= x .<= cd.canvas_maxs)` (`:376`), while `get_points`
generates it without such a filter (`Domains.jl:296-321`). `t3b_lattice_missing.jl` (channel [0,2]×[0,0.5] with a
piston, full-canvas box): the top canvas row is missing for **Float32 N = 52** (y = 0.55769235, canvas max
0.5576923, 55 generated particles there), **Float32 N = 100** (103 particles) and **Float64 N = 400** (y =
0.5075000000000001, 403 particles); `canvas_max − (canvas_min + n dx)` = −5.96e-8 / −1.1e-16. Same on v2. The row is
the outermost ghost row, 3 h from the wall: `_add_static_ghosts!` cannot re-add it behind a receding piece (it lies
within the band `R + dx` = 3.5 h). Interior stencils (R = 2.5 h) do not reach it, so the effect is small.
- Fix: iterate the generated index range `0:n` (n = `round((max − min)/dx)`) and skip the coordinate filter for
  in-range indices, or snap `j == n` to `canvas_maxs` as the wall indices are snapped.

## Accuracy against the parent (rr4's table on v3)

| case | v3 | v2 | parent | note |
|:--|:--|:--|:--|:--|
| plate translating with the flow (`p3`), N = 20/40/80 | 0.2645 / 0.06521 / 0.00997, Σ V 1.960000, content −0.00149 / −0.00100 / −0.00035 | 0.2642 / 0.06521 / 0.00997 | same | docs reproduce |
| plate, 1 + ½cos 5π(x−t) (`q10`) | within 12 h 0.345 / **0.336 / 0.304**; beyond 12 h 0.331 / 0.0995 / 0.0274 | 0.338 / 0.200 / 0.146 | 0.358 / 0.459 / 1.475 | docs' "0.34 / 0.20 / 0.15" no longer reproduces (N16) |
| plate, structure at the plate (`p11`) | 0.048 / 0.083 / 0.101; beyond 12 h 0.027 / 0.0075 / 0.0023 | 0.048 / 0.083 / 0.092 | 0.085 / 0.172 / 0.230 | 3–12 h worse (N16) |
| 2-D plane piston advancing into fluid at rest (`p6`) | 0.031 / 0.027 / 0.014 | same | 0.206 / 0.203 | docs reproduce |
| 2-D receding co-moving, periodic in y (`p6`) | **0.505 / 0.588 / 0.298** | 0.137 / 0.293 / 0.189 | 0.53 / 0.59 / 0.30 | N16 |
| 2-D receding along walls, consistent data, uncovered part (`q15`, `r8`) | 0.122 / 0.087 / 0.036 / 0.014 | 0.064 / 0.045 / 0.056 / 0.064 | 0.098 / 0.087 / 0.037 / 0.013 | docs reproduce; see N12 status |
| same, ExtrapolationBC on the face (`r8c`) | 0.243 / 0.138 / 0.058 / 0.015 | 0.121 / 0.193 / 0.191 / 0.319 | 0.252 / 0.165 / 0.070 / 0.024 | N13 fixed |
| same with r2's data (`p14`) | **0.234 / 0.264 / 0.191** (0.111 at 160) | 0.076 / 0.118 / 0.125 | 0.233 / 0.254 / 0.190 | back to the parent (N16) |
| 1-D receding piston (`p4`) | 0.1974 / 0.05956 / 0.01377, inflow 0.5105 / 0.5012 / 0.5002 | same | same | docs reproduce |
| 1-D fast oscillating piston (`p4`, `p5`) | Σ V 1.100199 / 1.097136 / 1.088276 vs 1.095106 | 1.100000 / 1.075571 / 1.073220 | – | L3 smaller |
| 3-D ball, place + sweep (`p20`) | [0.9953, 2.0047] | same | [0.875, 2.125] | docs reproduce |
| cost (`p17`, `q5`) | 25.6 vs 5.5 ms, 14.4 MB/step; sweep with motion 1.73 ms, 0.23 ms without; 3-D N = 24 15.5 s, 3.05 GB | 26.3 / 5.51 ms, 12.75 MB, 3.36 ms; 15.1 s | 26.8 / 5.85 ms; 10.9 s | docs' 3.4 ms / 13 MB stale (minors) |

## Test coverage

- The two changed tests detect what they target on v2: the multi-line string (`r7`: ParseError on v2) and wall-row
  particles at `y == 0` behind the receding piston (`q11`: 0 ≥ 5 fails on v2).
- Nothing tests the inserted volumes or the accuracy behind a receding face, so N16 passes the suite (the periodic
  test only checks the pair distance and Σ V). A bound on `V/h²` after the existing 20 periodic sweeps would catch it.
- Not covered: N13's ExtrapolationBC corner and oscillating-piston wall rows, N14's composite case, N17, the
  receding fast piece (v2's NaN at 3 h per sweep periodic, `t2`).
- Compile budget: the new asserts reuse existing grids and types.

## Minor issues

- Docs/docstring: "A piece whose box reaches across a periodic end is refused" (`MovingBoundaries.jl:190`, docs
  Limits) is still the old rule (now: a piece with points on a periodic end); the summary (`:166-169`) omits "within
  reach of a moving piece"; the docs' cost line (sweep 3.4 ms, 13 MB) no longer matches (1.73 ms, 14.4 MB, `p17`).
- Receding pieces faster than the documented limit (≥ 1.5 h per sweep) leave interior particles with V ≈ 1e-241
  (`t2`, both trees) and Σ V short of the region by 0.025–0.075 (region-pass truncation, L3). They carry no content;
  a positive floor or dropping them as `_drop_empty!` does would be cleaner.
- `_uncovered_points`' dedup is O(n_out²) per sweep (fine in 2-D; a 3-D piston plane of 10⁴ points costs ~10⁸
  distance checks).
- `first` is used as a local variable in `sweep_boundaries!` (`:236`), shadowing `Base.first` in that scope.
- Syntax errors in string shapes are now `ArgumentError` (was `ParseError`); the docstring of the builder does not say.

## Verdict

5d5d5ca fixes what rr4 asked for in the cases rr4 measured: the receding piston along walls converges again (0.014 at
N = 160, the docs' numbers reproduce), the ExtrapolationBC corner converges, an oscillating piston keeps its wall
rows, the composite case no longer churns particles, string shapes take newline-separated statements, and the suite
is green. The way N12 was fixed is wrong, however: inserting all candidates in one round lets every inserted particle
take its share as if alone, so inserted particles get up to 2.7 h², survivors are drained to zero volume, and an
alternating volume pattern (0.23 / 1.68 h²) fills every uncovered region. The receding piston in a channel periodic
across its face, the docs' plate case next to the plate and r2's receding case fall back to the parent's errors
(0.59, 0.30–0.34, 0.26). Joint insertion events restore the volumes and those cases but leave N12's case
non-convergent, so N12's cause is still open (N16). The canvas-row drop of N6 persists through the new canvas filter
(N17).

Earlier findings (rr4): N13, N14 fixed; N12 fixed in its case only through N16; N15 documented, not fixed; string
and warning-count minors fixed, docstring and Float32-lattice minors partly fixed. Earlier reviews: H1 (removals),
H2, N1, N5, N11 still fixed; H1 for insertions back (N16); N2's volume defect back and larger.
New findings: 1 high (N16, regression of 5d5d5ca), 1 low (N17, pre-existing); both confirmed.
