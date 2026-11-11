# Boundary Conditions

Boundary conditions act on the boundary (ghost) particles of the narrow band around the physical region. They are registered per boundary tag in the domain's `bc_map` and applied by `apply_boundary_conditions!`, which iterates over the map and calls each condition's functor.

The time stepper applies them to **every state it builds**: each stage state $Y_i$ (after the implicit solve) and the final state. MOOD re-forms a candidate after an order drop, and the boundary conditions are re-applied to that candidate as well. Interior particles are never touched by a boundary condition, and boundary particles are never updated by the spatial scheme; their right-hand side is zero.

On moving grids, boundary particles keep their positions (their grid velocity is zero).

## Built-in conditions

- **`FixedDirichlet`** sets every boundary particle with the given tag back to its value in `pg.rhos`. Because boundary particles are never advanced, this is their initial state.
- **`OutflowBC`** is a zero-gradient (transmissive) condition using a nearest-donor sweep. Interior particles act as donors; in up to five passes, each unresolved boundary particle with the tag copies the state of its nearest already-resolved neighbour. A boundary particle that finds no donor keeps its initial state.
- **`ExtrapolationBC(degree)`** is the transmissive condition of polynomial degree `degree` (0–5): the boundary particle takes the value *at its own position* of a one-sided MLS fit through the interior (see below). Degree 0 is a donor copy like `OutflowBC`; degree `q` keeps the first `q` derivatives of the interior solution continuous across the boundary. It needs no wall geometry.
- **`MirrorBC(op; degree, bounded)`** is the wall condition built from the mirror image: the one-sided fit is evaluated at the mirror point $x - 2\varphi n$ behind the wall ($\varphi$ the wall distance, $n$ the outward normal of the particle, from the domain's [shape](@ref Shapes)), and the boundary particle stores `op(eq, U, wall::WallPoint)` of the value `U` found there. Without `op` the mirrored value itself is stored, an even reflection of every component (a symmetry plane, zero normal derivative). Physics packages supply the operators of their walls: a slip wall reflects the normal velocity, `op(eq, U, w) = reflect_velocity(eq, U, w.normal)`; a diffuse kinetic wall builds a Maxwellian from the wall state. `op` must be a bits-type callable (a plain function or struct without captured arrays). With `bounded` (the default) the mirrored value is clamped to the range of the data of its fit before `op` is applied, see the stability notes. On a planar wall the mirror image is exact and the condition as accurate as the fit; on a curved wall it is second-order accurate. Requires a shape-based domain.

### The fit

`ExtrapolationBC` and `MirrorBC` share one kernel. For a boundary particle it

1. finds the **donor**: the interior particle nearest to the target point (the particle itself, or its mirror image), searched in the particle's neighbour list, then in its neighbours' lists (the outermost ghosts of the $R + \Delta x$ band have no interior neighbour), and refined by walking the neighbour graph towards the target;
2. moves the donor one particle inwards if it and its interior neighbours are fewer than $1.5\times$ the unknowns of the requested degree (a corner particle has 3–4 interior neighbours for the 6 unknowns of a 2-D quadratic): a well-supported fit a little farther away is still exact for polynomials, a lowered degree is not;
3. fits a polynomial of degree $q$ **centred at the target point** through the donor and its **interior** neighbours only (ghost states are being built and must not enter), weighted by their distance to the target with the grid's weight function and with the conditional regularisation of every MLS fit; the constant term is the value at the target. The degree is the requested one, at most the donor's MOOD order minus one, at most what the data count supports with the margin, and lowered further if the moment matrix does not factorise; degree 0 is the weighted average of the data.

A boundary particle without any interior particle within two hops keeps its state. The cost is one small fit per boundary particle per stage, a few percent of a flux evaluation (0.2–0.5 ms per application on a 200² grid with 2 700 ghosts, against 7.5 ms for a step).

### Stability

The ghost states feed back into the interior scheme through the stencils of the near-wall particles, so a condition that amplifies perturbations of the interior data grows without bound. Three measures keep the fit-based conditions stable; they were found on a 2-D channel (periodic in x, walls at y = 0 and 1, jittered cloud, R = 2.5Δx) with a Gaussian centred on the wall advected along it for two periods at half the scheme's own time step, with unlimited MUSCL3 and MUSCL5 (the worst case: no limiter, no MOOD):

1. **The fit is centred at the target point** and weighted from it. A Taylor fit centred at the donor and evaluated at the target amplified perturbations on jittered clouds (the mirror point often lies between the donor and the wall, outside the one-sided data cloud) and grew without bound for every degree ≥ 1.
2. **The degree margin** (1.5× the unknowns): an exactly determined one-sided fit has large coefficients.
3. **The bound of `MirrorBC`**: the mirrored value is clamped to the range of its data. With it every tested case stayed below the initial maximum (MUSCL3 and MUSCL5, degrees 1–3, both resolutions); without it MUSCL5 grew by ×2 (degree 1), ×7 (degree 2) and ×180 (degree 3) over two periods at N = 40, MOOD included, while MUSCL3 stayed bounded for degrees 1–3. `OutflowBC` itself is not safe in that setting (MUSCL5 at N = 20 grew by ×3.7), and the same scheme with the exact solution written into the ghosts is stable at every resolution, so the growth is a property of the feedback, not of the interior scheme.

`ExtrapolationBC` has no bound (an extrapolation legitimately leaves the range of its data) and is an **outflow** condition: on a 2-D outflow (velocity (1, 0.3) leaving through two faces, bump gone by t ≈ 1, run to t = 2) degree 1 is as stable as `OutflowBC` with MUSCL3 and MUSCL5, degree 2 and 3 grow with MUSCL5 (to 10⁹ and 10¹⁸) and degree 2 mildly with MUSCL3 at N = 40. Use degree 1 (the default) unless a run shows otherwise, and never use it at a wall with tangential flow (there even degree 1 grows: use `MirrorBC`). In 1-D (a bump leaving through the right wall, t = 0.6) degree 4 with MUSCL5 reduced the error by 10–100× against `OutflowBC` and converged at 4th order, so the higher degrees are worth having where the flow is genuinely outgoing.

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
