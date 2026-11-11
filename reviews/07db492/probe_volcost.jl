# Cost of update_volumes! (called every stage of a moving grid) on a shape domain vs the same box as a rectangular domain.
using HyperCloudCore, StaticArrays, Random, Statistics
for n in (50, 100, 200)
    h = 1 / n
    step = Cuboid((0.0, 0.0), (3.0, 1.0)) - Tagged(Cuboid((0.6, -1.0), (4.0, 0.2)), t -> t + 4)
    box3 = Cuboid((0.0, 0.0), (3.0, 1.0)) - Tagged(Cuboid((10.0, 10.0), (11.0, 11.0)), 5)   # same region as the box, via CSG
    for (name, g) in (("rectangular 3x1", get_rectangular_domain(Float64, (0.0, 0.0), (3.0, 1.0))), ("step (CSG)", GeometricDomain(step)), ("box via CSG", GeometricDomain(box3)))
        pg = ParticleGrid(g, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
        update_volumes!(pg, ShepardVolumes())
        t = median([@elapsed update_volumes!(pg, ShepardVolumes()) for _ in 1:5])
        println("n=$n $name: N=$(pg.meta.N), update_volumes! $(round(1e3t, digits = 2)) ms, domain_volume $(domain_volume(pg))")
    end
end
