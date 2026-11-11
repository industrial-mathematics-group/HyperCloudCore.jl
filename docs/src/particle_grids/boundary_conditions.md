# Boundary Conditions

Boundary conditions act on the boundary (ghost) particles of the narrow band around the physical region. They are registered per boundary tag in the domain's `bc_map` and applied by `apply_boundary_conditions!`, which iterates over the map and calls each condition's functor.

The time stepper applies them to **every state it builds**: each stage state $Y_i$ (after the implicit solve) and the final state. MOOD re-forms a candidate after an order drop, and the boundary conditions are re-applied to that candidate as well. Interior particles are never touched by a boundary condition, and boundary particles are never updated by the spatial scheme; their right-hand side is zero.

On moving grids, boundary particles keep their positions (their grid velocity is zero).

## Built-in conditions

- **`FixedDirichlet`** sets every boundary particle with the given tag back to its value in `pg.rhos`. Because boundary particles are never advanced, this is their initial state.
- **`OutflowBC`** is a zero-gradient (transmissive) condition using a nearest-donor sweep. Interior particles act as donors; in up to five passes, each unresolved boundary particle with the tag copies the state of its nearest already-resolved neighbour. A boundary particle that finds no donor keeps its initial state.
- **`ExtrapolationBC(degree)`** is the transmissive condition of polynomial degree `degree` (0–5): the boundary particle takes the value *at its own position* of a one-sided MLS fit through the interior (see below). Degree 0 is the weighted average of the data (the donor and its interior neighbours), close to the donor copy of `OutflowBC`; degree `q` keeps the first `q` derivatives of the interior solution continuous across the boundary. It needs no wall geometry.
- **`MirrorBC(op; degree, bounded)`** is the wall condition built from the mirror image: the one-sided fit is evaluated at the mirror point $x - 2\varphi n$ behind the wall ($\varphi$ the wall distance, $n$ the outward normal of the particle, from the domain's [shape](@ref Shapes)), and the boundary particle stores `op(eq, U, wall::WallPoint)` of the value `U` found there. Without `op` the mirrored value itself is stored, an even reflection of every component (a symmetry plane, zero normal derivative). Physics packages supply the operators of their walls: a slip wall reflects the normal velocity, `op(eq, U, w) = reflect_velocity(eq, U, w.normal)`; a diffuse kinetic wall builds a Maxwellian from the wall state. `op` must be a bits-type callable (a plain function or struct without captured arrays). With `bounded` (the default) the mirrored value is clamped to the range of the data of its fit before `op` is applied, see the stability notes. On a planar wall the mirror image is exact and the condition as accurate as the fit; on a curved wall it is second-order accurate. Requires a shape-based domain.

### The fit

`ExtrapolationBC` and `MirrorBC` share one kernel. For a boundary particle it

1. finds the **donor**: the interior particle nearest to the target point (the particle itself, or its mirror image), searched in the particle's neighbour list, then in its neighbours' lists (the outermost ghosts of the $R + \Delta x$ band have no interior neighbour), and refined by walking the neighbour graph towards the target;
2. chooses the **degree** $q$: starting from the requested one, it is lowered one step at a time until the donor and its interior neighbours number at least $1.5\times$ its unknowns, or, failing that, an interior neighbour of the donor does (the one nearest to the target, which then becomes the donor: a corner particle has 3–4 interior neighbours for the 6 unknowns of a 2-D quadratic, its neighbours further inside more, and a well-supported fit a little farther away is still exact for polynomials while a lowered degree is not); then at most the MOOD order of that donor minus one; in 3-D at most 3;
3. fits a polynomial of degree $q$ **centred at the target point** through the donor and its **interior** neighbours only (ghost states are being built and must not enter), weighted by their distance to the target with the grid's weight function, with the basis scaled by the largest data distance (which balances the moment matrix, important in Float32) and with the conditional regularisation of every MLS fit; the constant term is the value at the target. The degree is lowered further if the moment matrix does not factorise; degree 0 is the weighted average of the data.

A boundary particle without any interior particle within two hops keeps its state. The cost is one small fit per boundary particle per stage, a few percent of a flux evaluation (on a 200² grid with 2 700 ghosts, 2 threads: 0.34 ms for `ExtrapolationBC(1)`, 0.53/0.86 ms for `MirrorBC` degree 1/2, against 0.16 ms for `OutflowBC` and 7.5 ms for a step; degree 3 with R = 3.5Δx costs 2–3 ms).

### Stability

The ghost states feed back into the interior scheme through the stencils of the near-wall particles, so a condition that amplifies perturbations of the interior data grows without bound. Three measures keep the fit-based conditions stable; they were found on a 2-D channel (periodic in x, walls at y = 0 and 1, jittered cloud, R = 2.5Δx) with a Gaussian centred on the wall advected along it for two periods at half the scheme's own time step, with unlimited MUSCL3 and MUSCL5 (the worst case: no limiter, no MOOD):

0. **No ghosts close to the wall.** A ghost much less than Δx outside the wall, with the first interior particle up to Δx inside, has its mirror point in the gap between them, so the fit extrapolates, and the ghost is slaved to the interior next to it every stage. Such ghosts arise from a jittered wall row (about half of it crosses the wall) and from every wall off the lattice (curved walls, obstacles, box extents that are not a multiple of Δx). The point generator therefore reflects jittered interior points back inside and snaps points less than Δx/2 outside a wall onto it, as interior particles ([Point generation](@ref)). With those ghosts `MirrorBC` did not converge: on this channel 0.32/0.32/0.35 at N = 20/40/80 (t = 0.5, MUSCL3), on the same channel with the wall 0.1Δx off the lattice row and no jitter 0.41/0.67/0.63 at t = 1; the unbounded mirror grew with MUSCL5, and `OutflowBC` lost accuracy. Now the mirror converges with about the error of exact ghost values (0.28/0.062/0.023, exact ghosts 0.16/0.040/0.014; off the lattice 0.36/0.095/0.025 at t = 1, exact 0.21/0.062/0.013), and MUSCL5 with degree 1 or 2 stays bounded over two periods with and without the bound of item 3.

**Open: long runs along a wall at fine resolution.** At N = 160 the error of the mirror grows roughly linearly in time along a wall, for every offset of the wall from the lattice rows including a wall on the lattice, and the order from N = 80 to 160 is only 0.2–0.5 (a wall 0.5Δx off the rows: 0.125 → 0.106 at t = 1; there already 0.33/0.14/0.13 for N = 20/40/80), while the same scheme with exact ghost values converges and stays flat in time. The ghost fit itself is accurate (with exact interior data its error is about 10⁻⁵ and converges at 4th order), so the loss comes from the feedback of the mirrored states into the near-wall stencils of the interior scheme; it is not understood yet. Check long wall-bounded runs against a finer resolution.
1. **The fit is centred at the target point** and weighted from it. A Taylor fit centred at the donor and evaluated at the target amplified perturbations on jittered clouds (the mirror point often lies between the donor and the wall, outside the one-sided data cloud) and grew without bound for every degree ≥ 1.
2. **The degree margin** (1.5× the unknowns): an exactly determined one-sided fit has large coefficients.
3. **The bound of `MirrorBC`**: the mirrored value is clamped to the range of its data. On clouds that still had ghosts on the wall (item 0) it was what kept every tested case below the initial maximum (MUSCL5 grew by ×2, ×7 and ×180 for degrees 1–3 without it); on the current clouds the bounded and the unbounded mirror give the same result in this test, and the bound stays the default as a safeguard. (Under MUSCL3 the donor-order cap runs a requested degree 3 as degree 2.) `OutflowBC` itself was not safe in that setting (MUSCL5 at N = 20 grew by ×3.7).

`ExtrapolationBC` has no bound (an extrapolation legitimately leaves the range of its data) and is an **outflow** condition: on a 2-D outflow (velocity (1, 0.3) leaving through two faces, bump gone by t ≈ 1, run to t = 2; measured before item 0) degree 1 is as stable as `OutflowBC` with MUSCL3 and MUSCL5, degree 2 and 3 grow with MUSCL5 (to 10⁹ and 10¹⁸) and degree 2 mildly with MUSCL3 at N = 40. Use degree 1 (the default) unless a run shows otherwise, and never use it at a wall with tangential flow (there even degree 1 grows: use `MirrorBC`). In 1-D (a bump leaving through the right wall, t = 0.6, MUSCL5) degree 3 converged at 4th order and reduced the error by 10–10⁴× against `OutflowBC` at R = 3.5Δx, and degree 4 at R ≥ 4.5Δx at 5th order (a degree `q` needs `R ≳ (q + 0.5)Δx` in 1-D), so the higher degrees are worth having where the flow is genuinely outgoing. In Float32 use degree ≤ 2 (degree 3 extrapolation amplifies the round-off to 10⁻¹), and in 3-D at most degree 3 is used.

## Writing your own

Subtype `AbstractBoundaryCondition` and implement the functor:

```julia
struct MyBC <: AbstractBoundaryCondition end

function (::MyBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)
    @inbounds for i in 1:pg.meta.N
        if pg.core.is_boundary[i] && pg.core.tags[i] == tag
            rhos_buffer[i] = ...   # write in place
        end
    end
    return nothing
end
```

| Argument | Meaning |
|---|---|
| `pg` | The grid; positions, tags, flags, neighbour lists and the wall geometry (`pg.core.normals`, `pg.core.wall_distances`) are available |
| `rhos_buffer` | The state vector being built (a stage buffer, the MOOD candidate, or the final state). **Modify this one**, not `pg.rhos`. |
| `tag` | The boundary tag this call is responsible for |
| `ts` | The time stepper (useful for type-based dispatch) |
| `eq` | The PDE |
| `t` | Time of the state being built (stage time, or `t + dt`) |

Register the condition with `get_rectangular_domain(...; bc_map = Dict(1 => MyBC()))`. Use `pg.shared` buffers for scratch space (as `OutflowBC` does with `pg.shared.int_buffer`) rather than allocating. A condition that reads the wall normals should return `true` from [`requires_shape`](@ref), so that a domain without a shape is refused when the grid is built instead of feeding `NaN`s to the kernel. Most wall conditions need no functor of their own: a `MirrorBC` with an operator on the mirrored state and the [`WallPoint`](@ref) covers reflections of any kind.

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["BoundaryConditions.jl"]
Private = false
```
