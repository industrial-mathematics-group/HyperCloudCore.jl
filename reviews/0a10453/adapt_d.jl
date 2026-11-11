# One adapt_particles! call on freshly built static grids (old vs new): how many merges/insertions, and whether
# particles on the wall are moved off it.
using HyperCloudCore, StaticArrays, Random, LinearAlgebra, Printf
wf(R) = ExponentialWeightFunction(1.0, R)
for (nm, g, h) in (("disc r=1", get_spherical_domain(Float64, (0.0, 0.0), 1.0), 0.05),
                   ("box minus disc", GeometricDomain(Cuboid((0.0, 0.0), (2.0, 1.0)) - Tagged(Ball((0.7, 0.5), 0.23), 5)), 0.05),
                   ("box 1.03 x 1.01", get_rectangular_domain(Float64, (0.0, 0.0), (1.03, 1.01)), 0.05)), jit in (0.0, 0.2, 0.3)
    pg = ParticleGrid(g, (h, h), wf(2.5h), 1; randomness = (jit * h, jit * h), rng = MersenneTwister(3))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    onwall0 = count(i -> !pg.core.is_boundary[i] && abs(pg.core.wall_distances[i]) < 1e-9, 1:pg.meta.N)
    r = adapt_particles!(pg, ParticleAdaptivity(Float64), 3)
    onwall1 = count(i -> !pg.core.is_boundary[i] && abs(pg.core.wall_distances[i]) < 1e-9, 1:pg.meta.N)
    @printf("%-18s j=%.1f  inserted %3d removed %3d   interior on the wall %3d -> %3d\n", nm, jit, r.inserted, r.removed, onwall0, onwall1)
end
