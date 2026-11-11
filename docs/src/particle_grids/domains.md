# Geometric Domains

A `GeometricDomain` describes the physical problem region and nothing else: its bounding box, which points are inside, how boundary points are tagged, which axes are periodic, and which boundary condition belongs to which tag. Numerical padding (ghost layers) and point generation are handled separately when the `ParticleGrid` is built.

## Fields

| Field | Meaning |
|---|---|
| `mins`, `maxs` | Bounding box of the physical region |
| `is_periodic::SVector{D,Bool}` | Periodicity per axis |
| `is_interior(x)` | `true` if `x` lies in the physical region |
| `get_tag(x)` | `0` for interior points, a positive integer boundary tag otherwise |
| `bc_map::Dict{Int,AbstractBoundaryCondition}` | Boundary condition per tag |
| `shape` | The [`AbstractShape`](@ref) of the region (`nothing` for a domain built from functions) |

## Shapes

The region is described by a **signed distance** $\varphi(x)$: negative inside, zero on the boundary, positive outside, $|\varphi|$ the distance to the nearest boundary point. Everything the boundary treatment needs follows from it: the outward unit normal $n = \nabla\varphi$, the nearest wall point $x - \varphi n$ and the mirror image $x - 2\varphi n$. The solver only ever evaluates the functions of the shape interface, so a region sampled from a CAD model (a level set on a grid, from a point-in-solid classifier and a distance query of the CAD kernel, or from a surface mesh) only needs a subtype that implements them.

| Shape | Region | Tags |
|:------|:-------|:-----|
| `Cuboid(mins, maxs; open)` | the axis-aligned box; `open` axes have no faces (unbounded, used for periodic axes) | face `2d - 1` (lower) / `2d` (upper) of axis `d` |
| `Ball(center, radius)` | interval, disc or ball | 1 |
| `HalfSpace(point, normal)` | `(x - point) ⋅ normal ≤ 0` (unbounded) | 1 |
| `union(a, b, ...)`, `intersect(a, b, ...)` | `min` / `max` of the signed distances | of the active piece |
| `complement(a)`, `setdiff(a, b, ...)`, `a - b` | the outside of `a`; `a` with `b` cut out | of the active piece (kept) |
| `Tagged(shape, tag)`, `Tagged(shape, f)` | the same region | `tag`, or `f(tag)` |
| `Translated(shape, offset)`, `Rotated(shape, R; center)` | the moved region | of the shape |

The **active piece** of a union is the one with the smallest signed distance (the nearest piece for an outside point), of an intersection the one with the largest (the piece the point is farthest outside of). Its normal and tag are used, so a ghost particle outside a channel with an obstacle is tagged by the wall or the obstacle face it actually lies behind. The signed distance of a cuboid is the Euclidean distance also outside its edges and corners, where the normal points to the corner and the mirror image is the point reflection through it.

```julia
# A 2-D channel with a step (the Mach 3 step): walls 1-4 from the canvas, the step faces 5 (front) and 8 (top)
shape = Cuboid((0.0, 0.0), (3.0, 1.0)) - Tagged(Cuboid((0.6, -1.0), (4.0, 0.2)), t -> t + 4)
geom  = GeometricDomain(shape; bc_map = Dict(1 => FixedDirichlet(), 2 => OutflowBC(),
                                             3 => wall, 4 => wall, 5 => wall, 8 => wall))

# A rotated plate in a periodic channel: the canvas is open in x, so its box must be given
plate = Rotated(Cuboid((-0.2, -0.02), (0.2, 0.02)), π / 6)
geom  = GeometricDomain(Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - Tagged(plate, 5);
                        is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = ...)
```

`GeometricDomain(shape; bc_map, is_periodic, mins, maxs, label, volume)` takes the bounding box from the shape; along an axis where the shape is unbounded (open cuboid axes, half spaces) `mins`/`maxs` must be given; on bounded axes they may enlarge the box (the lattice is anchored at it), but a box that would crop the shape is an error (intersect the shape with a `Cuboid` to cut it, so that the cut carries ghosts and a tag; the bounding boxes of rotated and combined shapes are conservative). A shape used on a periodic axis must be open along it: the ends of a periodic axis are no boundary. The volume of the region within the box is computed once, by the constructor: analytic for cuboids and balls (and their tagged, moved copies), a lattice integration of the signed distance otherwise ([`shape_volume`](@ref), 10–50 ms); pass `volume` if it is known. [`domain_volume`](@ref) returns it. Tags must be positive (`0` marks interior points); `Tagged` and the domain's tag function refuse others. A ball decides inside/outside on the squared distance, as the spherical domain did before shapes (the points within $\Delta x/2$ outside it are then snapped onto it, see [Point generation](@ref)).

A shape of your own subtypes `AbstractShape{D, T}` and implements [`signed_distance`](@ref) and [`bounding_box`](@ref); [`boundary_normal`](@ref) falls back to central differences of the signed distance, [`boundary_tag`](@ref) to `1`, [`shape_volume`](@ref) to the lattice integration. (Up to 0.8 a domain held only an interior test and a tag function; that constructor still exists, see below.)

### Wall geometry per particle

Every particle of a grid built on a shape-based domain stores the outward unit normal of its nearest wall and its signed wall distance ([`wall_geometry`](@ref); `pg.core.normals`, `pg.core.wall_distances`, positive for ghosts), filled when the grid is built and when adaptivity inserts particles. The conditions that build ghost states from the interior use them: [`MirrorBC`](@ref) evaluates the interior at the mirror image $x - 2\varphi n$, and its operator receives the wall point, normal and distance as a [`WallPoint`](@ref). A domain built from functions stores `NaN`s, and a condition that needs the wall geometry ([`requires_shape`](@ref)) is refused by the `ParticleGrid` constructor on such a domain. Behind a convex corner where two pieces meet (a step, a wedge, a ramp) the nearest wall point of a ghost is the corner itself, found by projection and a descent along the boundary (exact to round-off for corners of about 15° and more; below that the projections converge too slowly: 14 % off at 5°). Its mirror image is the point reflection through the corner, which lies inside the region for corners of 90° or more; behind sharper corners it can lie outside, where the fit of a `MirrorBC` extrapolates.

## Built-in shapes

### Rectangular

```julia
get_rectangular_domain(T, mins, maxs; bc_map = Dict(), is_periodic = false)
```

The [`Cuboid`](@ref) `mins ≤ x ≤ maxs`, open along the periodic axes. `is_periodic` is either one `Bool` for all axes or an `NTuple{D,Bool}`. Boundary points get the tag of the face they lie outside of: `2d - 1` for the lower and `2d` for the upper face of axis `d`; outside several faces (the edges and corners of the ghost band) the one they are farthest outside of, the lower tag on ties. The faces of a periodic axis are ignored. (Up to 0.7.9 the 2-D tag was that of the nearest face *plane*, also measured from inside the box, so ghosts outside one wall close to another plane carried the other wall's tag.)

| D | tags |
|:--|:-----|
| 1 | 1 left (`x = min`), 2 right |
| 2 | 1 left, 2 right, 3 bottom, 4 top (a channel periodic in x has only 3 and 4) |
| 3 | 1/2 at `x = min/max`, 3/4 at `y`, 5/6 at `z` |

Up to version 0.7.9, 1-D and 3-D boxes tagged every boundary point 1, so `Dict(1 => bc)` covered both walls of an interval (and every face of a box) and a tag-2 entry was dead. Such a map now covers only the left wall (the x-min face), and the constructor warns that the other tags have no condition: write `Dict(1 => bc, 2 => bc)` (`Dict(t => bc for t in 1:6)` in 3-D). For `FixedDirichlet` that is the same as before. For `OutflowBC` in 3-D it is not exactly the same at the edges and corners of the ghost band: each tag is resolved on its own, and a ghost takes its donor from the interior and from the already resolved ghosts of its own tag only (2–5 % of the ghosts of a cube get different states than with one tag for all faces).

Boundary particles whose tag has no entry in `bc_map` keep their state (like a Dirichlet condition), and an entry whose tag no particle carries does nothing. When a non-empty `bc_map` and the tags of the boundary particles do not match, the `ParticleGrid` constructor warns (once per combination and session). An empty `bc_map` freezes every boundary particle without a warning.

### Spherical

```julia
get_spherical_domain(T, center, radius; bc_map = Dict(), is_periodic = false)
```

A disc (2-D) or ball (3-D), the [`Ball`](@ref) of `center` and `radius`; every exterior point gets tag 1.

### From functions

```julia
GeometricDomain(label, mins, maxs, is_periodic, is_interior, get_tag, bc_map)
```

A domain from an interior test and a tag function, without a shape. It generates the same particles and tags as before 0.9, but no wall normals (`NaN`), so it serves [`FixedDirichlet`](@ref), [`OutflowBC`](@ref), [`ExtrapolationBC`](@ref) and user conditions that do not read them; a [`MirrorBC`](@ref) is refused. A `domain_volume` method for its label is needed as well.

## Point generation

When a `ParticleGrid` is constructed, the domain is wrapped in a `ComputationalDomain`, and points are generated as follows:

1. **Canvas.** On non-periodic axes the bounding box is extended by $\lceil R/\Delta x_d\rceil$ spacings of that axis on each side to host ghost particles, i.e. at least $R$ on every axis; periodic axes get no padding. (Up to version 0.7.9 every axis got $\lceil R/\Delta x_{\max}\rceil$ spacings, which left the band thinner than $R$ on the finer axis of an anisotropic grid.)
2. **Lattice.** A Cartesian lattice with spacing `nominal_dx` (plus optional jitter `randomness`) covers the canvas. A non-periodic axis gets `round(L/dx) + 1` points, endpoints included: if a wall lies on the lattice (its index `(w - canvas_min)/dx` is an integer up to the round-off of the wall, the canvas minimum and the division), that lattice point is put on the wall exactly before the jitter is added, so both walls are interior. A wall between two lattice points (an extent that is not a multiple of `dx`) gets no lattice index, however close; the point less than `dx/2` outside it is snapped onto it in step 3. (Up to version 0.7.9 a distance tolerance did the snapping; far from the origin in Float32 it grew to almost `dx` and could put two points on a wall.) A periodic axis gets `round(L/dx)` points, because the endpoint coincides with the start point after wrapping.
3. **No ghosts close to the wall.** A jittered point whose lattice position is interior but which the jitter carried outside (half of a jittered row on a wall) is reflected back inside, to its mirror image $x - 2\varphi n$ (its lattice position without a shape, or if the mirror image is not interior). On a shape-based domain every other point less than $\Delta x_{\min}/2$ outside a wall is snapped onto its nearest wall point and becomes interior, as the lattice row on a box face does: then every ghost is at least $\Delta x_{\min}/2$ outside and the interior reaches the wall. On a lattice a snapped point stays at least $\Delta x/2$ from every other lattice point; where two snapped points would come within $\Delta x/4$ (at a reflex corner) the later one is pushed out to $\Delta x/2$ as a ghost instead. Ghosts close to the wall spoil the fit-based conditions ([Stability](@ref)). Grids whose walls lie on the lattice and are not jittered are unchanged; jittered ones and walls off the lattice (balls, obstacles, extents that are not a multiple of `dx`) differ from earlier versions in the points that crossed a wall or lay within $\Delta x/2$ outside it (the random numbers are drawn in the same order).
4. **Narrow band.** Lattice points outside the physical region are kept only if they lie within $R + \Delta x_{\max}$ of an interior point. These become the boundary (ghost) particles.
5. **Tags and wall geometry.** `get_tag` sets `tags` and `is_boundary = (tag != 0)`; [`wall_geometry`](@ref) fills the normals and wall distances.

Boundary particles carry the boundary condition and are never updated by the spatial scheme; see [Boundary Conditions](@ref).

## Reference

```@autodocs
Modules = [HyperCloudCore]
Pages   = ["Domains.jl", "Shapes.jl"]
Private = false
```
