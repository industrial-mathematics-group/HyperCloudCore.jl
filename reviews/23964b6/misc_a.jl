using HyperCloudCore, StaticArrays, Random, LinearAlgebra
const HC = HyperCloudCore
wf = ExponentialWeightFunction(1.0, 0.125)
# interior mass: open-cuboid channel vs rectangular periodic channel
g3 = GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0); open = (true, false)); is_periodic = (true, false), mins = (0.0, 0.0), maxs = (1.0, 1.0))
gr = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false))
for (nm, g) in (("open cuboid", g3), ("rectangular", gr))
    pg = ParticleGrid(g, (0.05, 0.05), wf, 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    println(nm, ": domain_volume ", domain_volume(pg), " interior_mass ", interior_mass(pg), " sum V interior ", sum(pg.core.volumes[i] for i in 1:pg.meta.N if !pg.core.is_boundary[i]))
end
# crop check vs conservative bounding boxes: Rotated ball, the user box tight around the actual disc
rb = Rotated(Ball((0.0, 0.0), 1.0), π / 4; center = (0.0, 0.0))
println("Rotated ball bbox: ", bounding_box(rb))
try
    GeometricDomain(rb; mins = (-1.0, -1.0), maxs = (1.0, 1.0)); println("rotated ball, tight box: accepted")
catch e; println("rotated ball, tight box: ", typeof(e)); end
# loose box on a bounded axis: silently ignored (lattice anchor moves)
gl = GeometricDomain(Ball((0.5, 0.5), 0.4); mins = (0.0, 0.0), maxs = (1.0, 1.0))
println("loose box ignored: mins = ", gl.mins)
# periodic closed shape: still accepted?
try
    gpc = GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0)); is_periodic = (true, false)); println("closed shape on periodic axis: accepted, vol ", gpc.volume)
catch e; println("closed on periodic: ", typeof(e)); end
# volume keyword, Float32 volume, legacy volume
println("volume kw: ", GeometricDomain(Ball((0.0, 0.0), 1.0); volume = 3).volume, " F32 CSG vol ", GeometricDomain(Cuboid((0f0, 0f0), (1f0, 1f0)) - Ball((0.5f0, 0.5f0), 0.2f0)).volume, " exact ", 1 - π * 0.04)
# Translated/Tagged ball: analytic vol?
println("Translated tagged ball volume: ", GeometricDomain(Tagged(Translated(Ball((0.0, 0.0), 1.0), (1.0, 1.0)), 3)).volume)
# _inside of ball within union/difference: still via sqrt
u = union(Ball((0.0, 0.0), 1.0), Ball((5.0, 0.0), 1.0))
x = SVector(0.6000000000000001, -0.8)
println("ball sq inside ", HC._inside(Ball((0.0, 0.0), 1.0), x), "; in union ", HC._inside(u, x))
# Tagged with function returning 0 for inside? tag function checked; negative int tag; Tagged{...} direct
try; Tagged(Cuboid((0.0, 0.0), (1.0, 1.0)), -2); catch e; println("Tagged -2: ", typeof(e)); end
g0 = GeometricDomain(Tagged(Cuboid((0.0, 0.0), (1.0, 1.0)), t -> t - 1))
try; ParticleGrid(g0, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1); println("tag 0 grid built!"); catch e; println("tag-0 function grid: ", typeof(e)); end
# Tagged with a Float tag: _checked_tag(::Integer) method error?
try
    gf = GeometricDomain(Tagged(Cuboid((0.0, 0.0), (1.0, 1.0)), t -> 2.0)); println("float tag fn: ", gf.get_tag(SVector(-0.1, 0.5)))
catch e; println("float tag fn: ", typeof(e)); end
# wall_geometry cost: far points, union-difference-rotated combos, iterations
step = Cuboid((0.0, 0.0), (3.0, 1.0)) - Cuboid((0.6, -1.0), (4.0, 0.2))
rotw = Rotated(intersect(Cuboid((-1.0, 0.0), (1.0, 1.0)), HalfSpace((0.0, 0.0), (-sin(0.3), cos(0.3)))), 0.7; center = (0.2, 0.1))
for (nm, s) in (("step", step), ("rotated wedge 17°", rotw))
    g = GeometricDomain(s; mins = (-2.0, -2.0), maxs = (4.0, 4.0))
    rng = MersenneTwister(1)
    pts = [SVector(-3.0, -3.0) + 10 * rand(rng, SVector{2, Float64}) for _ in 1:20000]
    wall_geometry(g, pts[1])
    t = @elapsed for p in pts; wall_geometry(g, p); end
    # worst single call
    tw = maximum(p -> (@elapsed wall_geometry(g, p)), pts)
    println("$nm: wall_geometry mean $(round(t / length(pts) * 1e6, digits = 2)) us, worst $(round(tw * 1e6, digits = 1)) us")
end
