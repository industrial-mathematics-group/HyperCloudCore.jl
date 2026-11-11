# Grid construction time (median of 3 after a warm-up), old vs new
using HyperCloudCore, StaticArrays, Random, Printf
wf(R) = ExponentialWeightFunction(1.0, R)
cases = [
  ("2-D box 400^2 (on lattice)", () -> (get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (1/400, 1/400))),
  ("2-D box 400^2 (on lattice) j=0.3", () -> (get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (1/400, 1/400))),
  ("2-D disc 400^2", () -> (get_spherical_domain(Float64, (0.0, 0.0), 1.0), (2/400, 2/400))),
  ("2-D box minus disc 400x200", () -> (GeometricDomain(Cuboid((0.0, 0.0), (2.0, 1.0)) - Tagged(Ball((0.7, 0.5), 0.23), 5)), (1/200, 1/200))),
  ("2-D step 0.61/0.213 h=1/200", () -> (GeometricDomain(Cuboid((0.0, 0.0), (3.0, 1.0)) - Tagged(Cuboid((0.61, -1.0), (4.0, 0.213)), 5)), (1/200, 1/200))),
  ("2-D rot plate h=1/200", () -> (GeometricDomain(Cuboid((-1.0, -0.5), (1.0, 0.5)) - Tagged(Rotated(Cuboid((-0.2, -0.02), (0.2, 0.02)), π / 6), 5)), (1/200, 1/200))),
  ("3-D box 64^3 (on lattice)", () -> (get_rectangular_domain(Float64, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (1/64, 1/64, 1/64))),
  ("3-D ball 64^3", () -> (get_spherical_domain(Float64, (0.0, 0.0, 0.0), 1.0), (2/64, 2/64, 2/64))),
  ("3-D box minus ball h=1/40", () -> (GeometricDomain(Cuboid((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)) - Tagged(Ball((0.7, 0.5, 0.5), 0.23), 5)), (1/40, 1/40, 1/40))),
]
for (nm, mk) in cases
    g, dx = mk()
    j = occursin("j=0.3", nm) ? 0.3 : 0.0
    build() = ParticleGrid(g, dx, wf(2.5 * dx[1]), 1; randomness = j .* dx, rng = MersenneTwister(1))
    build()
    ts = [(@elapsed build()) for _ in 1:3]
    pg = build()
    # point generation alone
    cd = HyperCloudCore.ComputationalDomain(g, dx, 2.5)
    tp = minimum([(@elapsed HyperCloudCore.get_points(cd, g; nominal_dx = dx, interp_range_factor = 2.5, randomness = j .* dx, rng = MersenneTwister(1))) for _ in 1:3])
    @printf("%-36s N %8d  build median %.3f s   get_points min %.3f s\n", nm, pg.meta.N, sort(ts)[2], tp)
end
