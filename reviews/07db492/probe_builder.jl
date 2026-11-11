# Builder :shape with a shape of another element type; Float32 shape-domain grid end to end.
using HyperCloudCore, StaticArrays, Random
ctx = Dict{Symbol, Any}(:T => Float64, :D => 2)
g = HyperCloudCore.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => Ball((0f0, 0f0), 1f0)), ctx)
println("geom T = ", typeof(g.mins))
try
    ParticleGrid(g, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
    println("grid ok")
catch e
    println("ParticleGrid with T=Float64 spacing: ", typeof(e))
end
s32 = Cuboid((0f0, 0f0), (1f0, 1f0)) - Tagged(Ball((0.5f0, 0.5f0), 0.2f0), 5)
g32 = GeometricDomain(s32)
pg = ParticleGrid(g32, (0.05f0, 0.05f0), ExponentialWeightFunction(1f0, 0.125f0), 1; randomness = (0.01f0, 0.01f0), rng = MersenneTwister(1))
println("F32 shape grid: N=$(pg.meta.N), tags ", sort(unique(view(pg.core.tags, 1:pg.meta.N))), ", eltype normals ", eltype(pg.core.normals), ", vol ", sum(view(pg.core.volumes, 1:pg.meta.N)))
