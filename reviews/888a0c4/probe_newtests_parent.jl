# The new Core tests' assertions run against whichever HyperCloudCore is loaded (parent vs review)
using HyperCloudCore, StaticArrays, LinearAlgebra, Random
println("HyperCloudCore at ", pathof(HyperCloudCore))
println("has _cholesky_sym: ", isdefined(HyperCloudCore, :_cholesky_sym), ", has wall_curvatures: ", hasfield(HyperCloudCore.ParticleGridCore, :wall_curvatures))
for (Nx, Ny) in ((150, 25), (300, 50), (600, 100))
    hp = 1.5 / Nx
    gch = GeometricDomain(intersect(Cuboid((0.0, 0.0), (1.5, 0.25)), Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-0.5, 0.0))));
                          mins = (0.0, 0.0), maxs = (1.5, 0.25))
    for seed in 1:3
        pgc = ParticleGrid(gch, (hp, 0.25 / Ny), ExponentialWeightFunction(1.0, 2.5hp), 1; randomness = (0.2hp, 0.2hp), rng = Random.Xoshiro(seed))
        bad = [i for i in 1:pgc.meta.N if !pgc.core.is_boundary[i] && pgc.core.positions[i][1] > 1 - hp / 2]
        dmin = minimum(i -> 1 - pgc.core.positions[i][1], [i for i in 1:pgc.meta.N if !pgc.core.is_boundary[i]]) / hp
        ng5 = count(i -> pgc.core.is_boundary[i] && pgc.core.tags[i] == 5, 1:pgc.meta.N)
        println("$(Nx)x$(Ny) seed $seed: N = $(pgc.meta.N), interior within h/2 of the face: $(length(bad)) $(bad == [] ? "" : string(pgc.core.positions[bad])), min distance to face $(round(dmin, digits = 3)) h, piston ghosts $ng5")
    end
end
