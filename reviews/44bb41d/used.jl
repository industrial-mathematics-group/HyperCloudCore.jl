# Ghosts that are no neighbour of any interior particle (their gathered gradient is never read by a flux)
include("common.jl")
for (D, N) in ((2, 80), (3, 16), (3, 32))
    h = 1 / N
    pg = ParticleGrid(get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D)), ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1)
    used = falses(pg.meta.N)
    for i in interior(pg), k in pg.neighbor.ranges[i]; used[pg.neighbor.indices[k]] = true; end
    G = ghosts(pg)
    @printf("%dD N=%d: ghosts %d, not read by any interior flux %d (%.0f %%)\n", D, N, length(G), count(i -> !used[i], G), 100 * count(i -> !used[i], G) / length(G))
end
