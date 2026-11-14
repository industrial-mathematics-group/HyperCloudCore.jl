# Dump the particles of the channel-piston grid (150 x 25, seed 1) for a parent/review comparison
using HyperCloudCore, StaticArrays, Random, Serialization
Nx, Ny = parse(Int, ARGS[2]), parse(Int, ARGS[3]); hp = 1.5 / Nx
gch = GeometricDomain(intersect(Cuboid((0.0, 0.0), (1.5, 0.25)), Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-0.5, 0.0))));
                      mins = (0.0, 0.0), maxs = (1.5, 0.25))
pg = ParticleGrid(gch, (hp, 0.25 / Ny), ExponentialWeightFunction(1.0, 2.5hp), 1; randomness = (0.2hp, 0.2hp), rng = Random.Xoshiro(1))
serialize(ARGS[1], [(Vector(pg.core.positions[i]), pg.core.is_boundary[i], pg.core.tags[i]) for i in 1:pg.meta.N])
