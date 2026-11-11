# Time step (MUSCL3 CFL) and Shepard volumes on grids with snapped points, old vs new
using HyperCloudCore, StaticArrays, Random, LinearAlgebra, Printf
include("common.jl")
for (nm, mk, D, h) in (("disc r=1", () -> get_spherical_domain(Float64, (0.0, 0.0), 1.0), 2, 0.05),
                       ("box minus disc", () -> GeometricDomain(Cuboid((0.0, 0.0), (2.0, 1.0)) - Tagged(Ball((0.7, 0.5), 0.23), 5)), 2, 0.05),
                       ("union discs", () -> GeometricDomain(union(Ball((0.0, 0.0), 0.5), Ball((0.6, 0.1), 0.4))), 2, 0.04),
                       ("3-D box off-lattice", () -> get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.03, 1.01, 0.98)), 3, 0.1)), jit in (0.0, 0.3, 0.45)
    eq = LinearAdvection(ntuple(d -> (1.0,), D), Float64)
    pg = ParticleGrid(mk(), ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = ntuple(_ -> jit * h, D), rng = MersenneTwister(3))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
    interp = MUSCL(Float64, D, 1, 3, NoLimiter(), RusanovFlux())
    dt = get_time_step(pg, eq, interp)
    V = [pg.core.volumes[i] for i in 1:pg.meta.N if !pg.core.is_boundary[i]]
    @printf("%-20s j=%.2f  dt/h %.4f   interior V/h^D min %.3f max %.3f   ΣV_int/|Ω| %.4f\n", nm, jit, dt / h, minimum(V) / h^D, maximum(V) / h^D, sum(V) / domain_volume(pg))
end
