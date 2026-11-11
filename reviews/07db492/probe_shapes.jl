# Correctness probes for shapes and wall_geometry (current tree only).
using HyperCloudCore, StaticArrays, Random, LinearAlgebra, Statistics
const HC = HyperCloudCore
rng = MersenneTwister(1)

println("== 1. Cuboid vs brute-force distance (2-D/3-D, F64/F32, open axes)")
function brute_box(mins, maxs, open, x)
    D = length(x)
    inside = all(d -> open[d] || mins[d] <= x[d] <= maxs[d], 1:D)
    # nearest point of the box
    p = [open[d] ? x[d] : clamp(x[d], mins[d], maxs[d]) for d in 1:D]
    inside || return norm(x .- p)
    return -minimum(min(x[d] - mins[d], maxs[d] - x[d]) for d in 1:D if !open[d]; init = Inf)
end
for T in (Float64, Float32), D in (1, 2, 3)
    mins = ntuple(d -> T(-0.3d), D); maxs = ntuple(d -> T(0.7 + 0.2d), D)
    for open in (ntuple(_ -> false, D), ntuple(d -> d == 1, D))
        D == 1 && open[1] && continue
        b = Cuboid{D, T}(mins, maxs; open = open)
        err = 0.0; nerr = 0
        for _ in 1:20000
            x = SVector{D, T}(ntuple(_ -> T(4 * rand(rng) - 2), D))
            e = abs(Float64(signed_distance(b, x)) - brute_box(Float64.(mins), Float64.(maxs), open, Float64.(x)))
            err = max(err, e)
            # normal: finite-difference gradient of the brute force (away from ridges)
            n = boundary_normal(b, x)
            abs(norm(n) - 1) < 10eps(T) || (nerr += 1)
        end
        println("  T=$T D=$D open=$open: max |sd - exact| = $err, non-unit normals $nerr")
    end
end

println("== 2. Rotated: 2-D/3-D conventions")
b = Cuboid((0.0, 0.0, 0.0), (2.0, 1.0, 0.5))
r = Rotated(b, (0.0, 0.0, 1.0), π / 2)        # x -> y, y -> -x
println("  z-rot 90: point (-0.5, 1.5, 0.25) inside? ", signed_distance(r, SVector(-0.5, 1.5, 0.25)) < 0,
        "; (0.5, 1.5, 0.25) inside? ", signed_distance(r, SVector(0.5, 1.5, 0.25)) < 0)
println("  bbox ", bounding_box(r))
rx = Rotated(b, (1.0, 0.0, 0.0), π / 2)       # y -> z
println("  x-rot 90: (1, 0, 0.5) inside? ", signed_distance(rx, SVector(1.0, -0.25, 0.5)) < 0, " normal at (1,-0.25,1.2): ", boundary_normal(rx, SVector(1.0, -0.25, 1.2)), " tag ", boundary_tag(rx, SVector(1.0, -0.25, 1.2)))
# random rotation: sd invariance
k = normalize(SVector(0.3, -0.7, 0.4)); θ = 1.1
rr = Rotated(b, k, θ; center = (0.2, 0.1, -0.3))
R = rr.R
maxe = 0.0
for _ in 1:10000
    y = SVector(ntuple(_ -> 4rand(rng) - 2, 3))
    x = R * (y - rr.center) + rr.center
    global maxe = max(maxe, abs(signed_distance(rr, x) - signed_distance(b, y)), norm(boundary_normal(rr, x) - R * boundary_normal(b, y)))
end
println("  random rotation: max |sd(R y) - sd(y)|, |n(Ry) - R n(y)| = $maxe; det R = $(det(R)), R*k - k = $(norm(R * k - k))")
lo, hi = bounding_box(rr)
pts = [R * (SVector(2rand(rng), rand(rng), 0.5rand(rng)) - rr.center) + rr.center for _ in 1:100000]
println("  bbox contains rotated samples: ", all(p -> all(lo .- 1e-12 .<= p .<= hi .+ 1e-12), pts))
r32 = Rotated(Cuboid((0f0, 0f0, 0f0), (1f0, 1f0, 1f0)), (0f0, 0f0, 1f0), 0.3)
println("  Float32 3-D Rotated eltype ", eltype(r32), ", sd type ", typeof(signed_distance(r32, SVector(0.5f0, 0.5f0, 0.5f0))))
r2 = Rotated(Cuboid((0f0, 0f0), (1f0, 1f0)), π / 3)
println("  Float32 2-D Rotated ok: ", typeof(boundary_normal(r2, SVector(2f0, 0f0))))

println("== 3. wall_geometry behind convex corners of intersections (wedges): distance vs exact")
# region = triangle-like wedge: Cuboid ∩ HalfSpace with interior angle α at the corner on the bottom edge
function seg_dist(p, a, b)
    t = clamp(dot(p - a, b - a) / dot(b - a, b - a), 0, 1)
    return norm(p - (a + t * (b - a)))
end
for α in (15, 30, 45, 60, 90, 120, 150, 170)
    a = deg2rad(α)
    # half-plane through the origin whose boundary makes the angle α with the +x axis, region to the right of that line
    nrm = SVector(-sin(a), cos(a))                 # outward normal (points to the left of the line direction)
    shape = intersect(Cuboid((-5.0, 0.0), (5.0, 5.0)), HalfSpace((0.0, 0.0), Tuple(nrm)))
    geom = GeometricDomain(shape)
    dirline = SVector(cos(a), sin(a))
    # boundary polyline near the corner: segments (0,0)-(5,0) and (0,0)-(t*dir)
    worst = 0.0; worst_n = 0.0; wx = SVector(0.0, 0.0); nbad = 0
    for _ in 1:20000
        x = SVector(2rand(rng) - 1.0, 2rand(rng) - 1.0) * 0.3
        signed_distance(shape, x) > 0 || continue
        ex = min(seg_dist(x, SVector(0.0, 0.0), SVector(5.0, 0.0)), seg_dist(x, SVector(0.0, 0.0), 5dirline))
        n, d = wall_geometry(geom, x)
        e = abs(d - ex) / ex
        e > 1e-6 && (nbad += 1)
        e > worst && (worst = e; wx = x)
        # mirror point inside?
    end
    n, d = wall_geometry(geom, wx)
    xm = wx - 2d * n
    println("  α = $(α)°: max rel. error of the wall distance $(round(worst, sigdigits = 3)) ($nbad of the outside samples > 1e-6) at $wx; mirror sd $(round(signed_distance(shape, xm), sigdigits = 3))")
end

println("== 4. wall_geometry inside a union (interior particles): distance vs exact")
u = union(Cuboid((0.0, 0.0), (2.0, 1.0)), Cuboid((1.0, 0.0), (3.0, 1.0)))   # = box (0,0)-(3,1)
gu = GeometricDomain(u)
x = SVector(1.5, 0.5)
println("  union of two overlapping boxes = [0,3]x[0,1]; at (1.5, 0.5): sd $(signed_distance(u, x)), wall_geometry $(wall_geometry(gu, x)) (exact -0.5, normal ±y)")
x = SVector(1.9, 0.9)
println("  at (1.9, 0.9): sd $(signed_distance(u, x)), wall_geometry $(wall_geometry(gu, x)) (exact -0.1, (0,1))")
x = SVector(1.95, 0.5)
println("  at (1.95, 0.5): sd $(signed_distance(u, x)), wall_geometry $(wall_geometry(gu, x)) (exact -0.5, normal ±y), tag $(boundary_tag(u, x))")

println("== 5. L-shaped region from union; ghosts in the concave notch")
L = union(Cuboid((0.0, 0.0), (2.0, 1.0)), Cuboid((0.0, 0.0), (1.0, 2.0)))
gL = GeometricDomain(L)
for x in (SVector(1.2, 1.2), SVector(1.05, 1.5), SVector(1.5, 1.05))
    println("  $x: sd $(signed_distance(L, x)) wall $(wall_geometry(gL, x)) tag $(boundary_tag(L, x))")
end

println("== 6. Box minus disc (hole): ghosts inside the hole")
hole = Cuboid((0.0, 0.0), (1.0, 1.0)) - Tagged(Ball((0.5, 0.5), 0.2), 5)
gh = GeometricDomain(hole)
x = SVector(0.5, 0.45)
println("  $x: sd $(signed_distance(hole, x)), wall $(wall_geometry(gh, x)), tag $(gh.get_tag(x)) (exact 0.15, normal (0,1) towards the centre? outward of the fluid = into the hole: (0, 1)... centre-pointing)")
x = SVector(0.5, 0.5)
println("  centre: sd $(signed_distance(hole, x)), wall $(wall_geometry(gh, x))")

println("== 7. Tags: Tagged with 0 / negative")
z = Tagged(Cuboid((0.0, 0.0), (1.0, 1.0)), 0)
gz = GeometricDomain(z)
println("  Tagged(box, 0): get_tag of an outside point = $(gz.get_tag(SVector(-0.1, 0.5))) (0 = interior marker)")
pgz = ParticleGrid(gz, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
println("  grid: outside particles with is_boundary false: ", count(i -> signed_distance(z, pgz.core.positions[i]) > 0 && !pgz.core.is_boundary[i], 1:pgz.meta.N))

println("== 8. periodic axis with a closed shape (not open)")
gp = GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0)); is_periodic = (true, false))
pgp = ParticleGrid(gp, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
ints = [i for i in 1:pgp.meta.N if !pgp.core.is_boundary[i]]
println("  accepted silently; interior particle at x=0: wall distance ", minimum(pgp.core.wall_distances[i] for i in ints if pgp.core.positions[i][1] == 0.0), " normal ", first(pgp.core.normals[i] for i in ints if pgp.core.positions[i] == SVector(0.0, 0.5)))

println("== 9. shape_volume: accuracy and cost")
for (name, s, ex) in (("2-D box minus step", Cuboid((0.0, 0.0), (3.0, 1.0)) - Cuboid((0.6, -1.0), (4.0, 0.2)), 3.0 - 2.4 * 0.2),
                      ("2-D box minus disc", Cuboid((0.0, 0.0), (1.0, 1.0)) - Ball((0.5, 0.5), 0.2), 1 - π * 0.04),
                      ("2-D rotated box ∩ box", intersect(Cuboid((0.0, 0.0), (1.0, 1.0)), Rotated(Cuboid((0.0, 0.0), (1.0, 1.0)), π / 4; center = (0.5, 0.5))), 2 * (sqrt(2) - 1)),
                      ("2-D union of discs overlapping", union(Ball((0.0, 0.0), 1.0), Ball((1.0, 0.0), 1.0)), 2π - (2π / 3 - sqrt(3) / 2)),
                      ("3-D cube minus ball", Cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) - Ball((0.5, 0.5, 0.5), 0.3), 1 - 4π / 3 * 0.027),
                      ("3-D union of two balls", union(Ball((0.0, 0.0, 0.0), 1.0), Ball((1.0, 0.0, 0.0), 1.0)), 2 * 4π / 3 - 2 * π * (1 - 0.5)^2 * (2 + 0.5) / 3),
                      ("3-D rotated cube ∩ cube", intersect(Cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), Rotated(Cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (0.0, 0.0, 1.0), π / 4; center = (0.5, 0.5, 0.5))), 2 * (sqrt(2) - 1)),
                      ("F32 2-D box minus disc", Cuboid((0f0, 0f0), (1f0, 1f0)) - Ball((0.5f0, 0.5f0), 0.2f0), 1 - π * 0.04))
    shape_volume(s; samples = 1000)
    t = @elapsed v = shape_volume(s)
    a = @allocated shape_volume(s)
    println("  $name: rel err $(round(abs(v - ex) / ex, sigdigits = 3)), $(round(t, digits = 3)) s, alloc $a, type $(typeof(v))")
end

println("== 10. shape domain grid cost (2-D 400², step) and type stability")
step = Cuboid((0.0, 0.0), (3.0, 1.0)) - Tagged(Cuboid((0.6, -1.0), (4.0, 0.2)), t -> t + 4)
gs = GeometricDomain(step)
h = 1 / 400
cd = HC.ComputationalDomain(gs, (h, h), 2.5)
HC.get_points(cd, gs; nominal_dx = (h, h), interp_range_factor = 2.5)
t = @elapsed P = HC.get_points(cd, gs; nominal_dx = (h, h), interp_range_factor = 2.5)
println("  get_points step 3x1 at h=1/400: $(round(t, digits = 3)) s, N=$(length(P[1]))")
f(g, x) = (g.is_interior(x), g.get_tag(x), wall_geometry(g, x))
f(gs, SVector(0.7, -0.05))
println("  alloc of interior/tag/wall_geometry on a step corner ghost: ", @allocated f(gs, SVector(0.7, -0.05)))
println("  inferred: ", Base.return_types(f, (typeof(gs), SVector{2, Float64})))
