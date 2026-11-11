# Grid construction time (ParticleGrid and get_points alone), median of 5 after warm-up; run on both trees.
using HyperCloudCore, StaticArrays, Random, Statistics
function setup(D, n)
    T = Float64; h = 1 / n
    geom = get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
    return geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), ntuple(_ -> 0.2h, D)
end
function bench(D, n)
    geom, dx, wf, rnd = setup(D, n)
    cd = HyperCloudCore.ComputationalDomain(geom, dx, 2.5)
    gp() = HyperCloudCore.get_points(cd, geom; nominal_dx = dx, interp_range_factor = 2.5, randomness = rnd, rng = MersenneTwister(1))
    pg() = ParticleGrid(geom, dx, wf, 1; randomness = rnd, rng = MersenneTwister(1))
    gp(); pg()
    tg = [(GC.gc(); @elapsed gp()) for _ in 1:5]
    ag = @allocated gp()
    tp = [(GC.gc(); @elapsed pg()) for _ in 1:3]
    println("D=$D n=$n: get_points median $(round(median(tg), digits = 3)) s (alloc $(round(ag / 2^20, digits = 1)) MiB), ParticleGrid median $(round(median(tp), digits = 3)) s, N=$(length(gp()[1]))")
end
bench(2, 400)
bench(3, 64)
geom, = setup(2, 400)
x = SVector(0.3, 0.4)
geom.is_interior(x); geom.get_tag(x)
println("is_interior alloc ", @allocated(geom.is_interior(x)), ", get_tag alloc ", @allocated(geom.get_tag(SVector(-0.01, 0.4))))
