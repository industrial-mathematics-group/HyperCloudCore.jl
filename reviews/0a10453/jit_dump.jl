# Dumps jittered grids (positions, is_boundary, wall distances) for comparison between trees.
using HyperCloudCore, StaticArrays, Random, Serialization
out = ARGS[1]
res = Dict{String, Any}()
haswall = isdefined(HyperCloudCore, :wall_geometry)
function rec(name, pg)
    N = pg.meta.N
    res[name] = (pos = [Vector{Float64}(pg.core.positions[i]) for i in 1:N], isb = [pg.core.is_boundary[i] for i in 1:N],
                 wd = hasproperty(pg.core, :wall_distances) ? [Float64(pg.core.wall_distances[i]) for i in 1:N] : fill(NaN, N))
end
wf(T, R) = ExponentialWeightFunction(one(T), T(R))
for (T, D, n, j) in ((Float64, 1, 20, 0.45), (Float64, 2, 20, 0.2), (Float64, 2, 20, 0.45), (Float32, 2, 20, 0.3), (Float64, 3, 10, 0.3))
    dx = ntuple(_ -> T(1 / n), D)
    for per in (false, true)
        g = get_rectangular_domain(T, ntuple(_ -> T(0), D), ntuple(_ -> T(1), D); is_periodic = ntuple(d -> per && d == 1, D))
        rec("rect T=$T D=$D j=$j per=$per", ParticleGrid(g, dx, wf(T, 2.5 / n), 1; randomness = ntuple(_ -> T(j / n), D), rng = MersenneTwister(4)))
    end
end
# off-lattice wall (extent 1.03, h = 0.05)
g = get_rectangular_domain(Float64, (0.0, 0.0), (1.03, 1.0))
rec("rect offlattice", ParticleGrid(g, (0.05, 0.05), wf(Float64, 0.125), 1; randomness = (0.01, 0.01), rng = MersenneTwister(4)))
for D in (2, 3)
    g = get_spherical_domain(Float64, ntuple(_ -> 0.0, D), 1.0)
    rec("sphere D=$D", ParticleGrid(g, ntuple(_ -> 2 / (D == 2 ? 20 : 10), D), wf(Float64, 2.5 * 2 / (D == 2 ? 20 : 10)), 1;
                                    randomness = ntuple(_ -> 0.3 * 2 / (D == 2 ? 20 : 10), D), rng = MersenneTwister(4)))
end
if haswall
    step = Cuboid((0.0, 0.0), (3.0, 1.0)) - Tagged(Cuboid((0.6, -1.0), (4.0, 0.2)), 5)
    rec("step", ParticleGrid(GeometricDomain(step), (0.05, 0.05), wf(Float64, 0.125), 1; randomness = (0.015, 0.015), rng = MersenneTwister(4)))
    a = deg2rad(30)
    wedge = intersect(Cuboid((-1.0, 0.0), (1.0, 1.0)), HalfSpace((0.0, 0.0), (-sin(a), cos(a))))
    rec("wedge30", ParticleGrid(GeometricDomain(wedge), (0.05, 0.05), wf(Float64, 0.125), 1; randomness = (0.02, 0.02), rng = MersenneTwister(4)))
    plate = Rotated(Cuboid((-0.2, -0.02), (0.2, 0.02)), π / 6)
    # (volume of the open channel would fail on the old tree; build points only if possible)
    try
        gp = GeometricDomain(Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - Tagged(plate, 5);
                             is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5))
        rec("plate", ParticleGrid(gp, (0.025, 0.025), wf(Float64, 0.0625), 1; randomness = (0.008, 0.008), rng = MersenneTwister(4)))
    catch e
        println("plate: ", typeof(e))
    end
end
# legacy domain without a shape (function interior test), a disc
let
    isi(x) = sum(abs2, x) <= 1.0
    tg(x) = isi(x) ? 0 : 1
    g = GeometricDomain(Val(:disc_legacy), SVector(-1.0, -1.0), SVector(1.0, 1.0), SVector(false, false), isi, tg,
                        Dict{Int, AbstractBoundaryCondition}())
    HyperCloudCore.domain_volume(::GeometricDomain{Val{:disc_legacy}}) = π
    rec("legacy disc", ParticleGrid(g, (0.1, 0.1), wf(Float64, 0.25), 1; randomness = (0.03, 0.03), rng = MersenneTwister(4)))
end
serialize(out, res)
println("done ", length(res))
