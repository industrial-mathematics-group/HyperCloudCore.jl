# Shape domains whose shape is unbounded along an axis (periodic channel, half space) with mins/maxs given:
# can a ParticleGrid be built (domain_volume)?
using HyperCloudCore, StaticArrays, Random
plate = Rotated(Cuboid((-0.2, -0.02), (0.2, 0.02)), π / 6)
geom  = GeometricDomain(Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - Tagged(plate, 5);
                        is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                        bc_map = Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in (3, 4, 5)))
try
    pg = ParticleGrid(geom, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
    println("docs plate example: grid built, N = ", pg.meta.N)
catch e
    println("docs plate example: ", typeof(e), ": ", sprint(showerror, e)[1:min(end, 200)])
end
g2 = GeometricDomain(HalfSpace((0.5, 0.0), (1.0, 0.0)); mins = (0.0, 0.0), maxs = (1.0, 1.0))
try
    pg = ParticleGrid(g2, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
    println("half space with box: grid built, N = ", pg.meta.N, " domain_volume ", domain_volume(pg))
catch e
    println("half space with box: ", typeof(e), ": ", sprint(showerror, e)[1:min(end, 200)])
end
# periodic channel as an open Cuboid without the rectangular label
g3 = GeometricDomain(Cuboid((0.0, 0.0), (1.0, 1.0); open = (true, false)); is_periodic = (true, false), mins = (0.0, 0.0), maxs = (1.0, 1.0))
try
    pg = ParticleGrid(g3, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
    println("open Cuboid channel: grid built, domain_volume ", domain_volume(pg))
catch e
    println("open Cuboid channel: ", typeof(e), ": ", sprint(showerror, e)[1:min(end, 200)])
end
pg = ParticleGrid(g3, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
println("open Cuboid channel volumes: ", extrema(view(pg.core.volumes, 1:pg.meta.N)), "; interior_mass of 1: ", (for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end; interior_mass(pg)))
