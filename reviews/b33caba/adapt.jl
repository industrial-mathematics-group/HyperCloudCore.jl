# The b33caba adaptivity check (unit_schemes.jl ~1290-1317) on any tree: total_mass change == mass_residual
include("common.jl")
kmax = 3
for (D, n) in ((1, 20), (2, 16)), est in (nothing, ShepardVolumes())
    bcD   = Dict{Int, AbstractBoundaryCondition}(t => FixedDirichlet() for t in 1:2D)
    geomD = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = bcD)
    h     = 1 / n
    pgD   = ParticleGrid(geomD, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1;
                         randomness = ntuple(_ -> 0.45h, D), rng = MersenneTwister(1))
    for i in 1:pgD.meta.N
        x = pgD.core.positions[i]
        pgD.rhos[i] = State{1, Float64}(1 + sum(x) + 0.3sinpi(3x[1]))
    end
    update_volumes!(pgD, ShepardVolumes())
    scale = sum(i -> pgD.core.volumes[i] * abs(pgD.rhos[i][1]), 1:pgD.meta.N)
    m0  = total_mass(pgD)[1]
    rep = adapt_particles!(pgD, ParticleAdaptivity(Float64; volume_estimator = est), kmax)
    d = (total_mass(pgD)[1] - m0) - rep.mass_residual[1]
    @printf("D=%d est=%-8s events %d  |Δtotal - residual| %.2e (bound %.2e, pass %s)  total==interior %s\n", D, est === nothing ? "none" : "Shepard",
            rep.inserted + rep.removed, abs(d), 1e-13 * scale, abs(d) < 1e-13 * scale, total_mass(pgD) == interior_mass(pgD))
end
