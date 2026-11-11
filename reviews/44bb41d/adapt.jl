# adapt_particles! on bounded grids next to walls (mid vs new): mass_residual, change of total_mass and interior_mass,
# interior Σ V vs |Ω| before/after, ghost volumes, with and without volume_estimator; non-uniform U
using HyperCloudCore, StaticArrays, Random, LinearAlgebra, Printf
wf(R) = ExponentialWeightFunction(1.0, R)
vint(pg) = sum(pg.core.volumes[i] for i in 1:pg.meta.N if !pg.core.is_boundary[i])
vgh(pg) = sum((pg.core.volumes[i] for i in 1:pg.meta.N if pg.core.is_boundary[i]); init = 0.0)
for (nm, g, h) in (("disc r=1", get_spherical_domain(Float64, (0.0, 0.0), 1.0), 0.05),
                   ("box minus disc", GeometricDomain(Cuboid((0.0, 0.0), (2.0, 1.0)) - Tagged(Ball((0.7, 0.5), 0.23), 5)), 0.05),
                   ("box 1.03 x 1.01", get_rectangular_domain(Float64, (0.0, 0.0), (1.03, 1.01)), 0.05)),
    jit in (0.3,), (en, est) in (("none", nothing), ("Shepard", ShepardVolumes()), ("MLS2", MLSVolumes(2)))
    pg = ParticleGrid(g, (h, h), wf(2.5h), 1; randomness = (jit * h, jit * h), rng = MersenneTwister(3))
    est === nothing || update_volumes!(pg, est)
    for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{1, Float64}(1.0 + 0.5 * sin(3x[1]) * cos(2x[2])); end
    m0, mi0, v0 = total_mass(pg)[1], interior_mass(pg)[1], vint(pg)
    r = adapt_particles!(pg, ParticleAdaptivity(Float64; volume_estimator = est), 3)
    @printf("%-16s %-7s ins %2d rem %2d  mass_residual %.1e  Δtotal %.1e  Δinterior %.1e  Vint %.6f -> %.6f (|Ω| %.6f)  Vghost %.3f\n",
            nm, en, r.inserted, r.removed, r.mass_residual[1], total_mass(pg)[1] - m0, interior_mass(pg)[1] - mi0, v0, vint(pg), domain_volume(pg), vgh(pg))
end
