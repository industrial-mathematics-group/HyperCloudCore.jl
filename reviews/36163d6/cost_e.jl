using HyperCloudCore, StaticArrays, Random, Printf
wf(R) = ExponentialWeightFunction(1.0, R)
cases = [
  ("2-D box 400^2 minus disc j=0", () -> (GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0)) - Tagged(Ball((0.4, 0.5), 0.15), 5)), (1/400, 1/400)), 0.0),
  ("2-D box 400^2 minus disc j=0.3", () -> (GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0)) - Tagged(Ball((0.4, 0.5), 0.15), 5)), (1/400, 1/400)), 0.3),
  ("2-D disc 400^2 j=0.3", () -> (get_spherical_domain(Float64, (0.0, 0.0), 1.0), (2/400, 2/400)), 0.3),
  ("3-D box 64^3 on lattice j=0.3", () -> (get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (1/64, 1/64, 1/64)), 0.3),
  ("3-D box 64^3 off lattice (1.003) j=0.3", () -> (get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.003, 1.0, 0.997)), (1/64, 1/64, 1/64)), 0.3),
  ("3-D ball 64^3 j=0.3", () -> (get_spherical_domain(Float64, (0.0, 0.0, 0.0), 1.0), (2/64, 2/64, 2/64)), 0.3),
]
for (nm, mk, j) in cases
    g, dx = mk()
    build() = ParticleGrid(g, dx, wf(2.5 * dx[1]), 1; randomness = j .* dx, rng = MersenneTwister(1))
    build()
    ts = sort([(@elapsed build()) for _ in 1:5])
    pg = build()
    cd = HyperCloudCore.ComputationalDomain(g, dx, 2.5)
    gp() = HyperCloudCore.get_points(cd, g; nominal_dx = dx, interp_range_factor = 2.5, randomness = j .* dx, rng = MersenneTwister(1))
    gp()
    tp = sort([(@elapsed gp()) for _ in 1:5])
    @printf("%-40s N %8d  build median %.3f s   get_points median %.4f s\n", nm, pg.meta.N, ts[3], tp[3])
end
