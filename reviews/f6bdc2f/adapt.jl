include("common.jl")
for D in (1, 2)
    dx = 1/20
    geom = get_rectangular_domain(Float64, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = BCD(t => MirrorBC() for t in 1:2D))
    pg = ParticleGrid(geom, ntuple(_ -> dx, D), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = ntuple(_ -> 0.2dx, D), rng = MersenneTwister(1))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(1.0); end
    c = Space{D,Float64}(ntuple(_ -> 0.5, D))
    for i in 1:pg.meta.N
        pg.core.is_boundary[i] || (pg.core.positions[i] = c + 0.9 * (pg.core.positions[i] - c))
    end
    pg.neighbor(pg)
    stale = count(i -> pg.core.wall_distances[i] != wall_geometry(geom, pg.core.positions[i])[2], 1:pg.meta.N)
    N0 = pg.meta.N
    rep = adapt_particles!(pg, ParticleAdaptivity(Float64), 3)
    chk(pg) = count(i -> (n, d) = wall_geometry(geom, pg.core.positions[i]) |> x -> !(x[1] ≈ pg.core.normals[i] && x[2] ≈ pg.core.wall_distances[i]), 1:pg.meta.N)
    new = (N0 - rep.removed + 1):pg.meta.N
    newbad = count(i -> begin (n, d) = wall_geometry(geom, pg.core.positions[i]); !(n ≈ pg.core.normals[i] && d ≈ pg.core.wall_distances[i]) end, (pg.meta.N - rep.inserted + 1):pg.meta.N)
    println("D=$D: moved interior particles with stale wall geometry before adapt: $stale; inserted $(rep.inserted), removed $(rep.removed); inserted with wrong geometry: $newbad")
    # reorder and check every particle's stored geometry still belongs to its position (compare to the pre-reorder map)
    before = Dict(pg.core.positions[i] => (pg.core.normals[i], pg.core.wall_distances[i]) for i in 1:pg.meta.N)
    pg.reorder(pg)
    bad = count(i -> before[pg.core.positions[i]] != (pg.core.normals[i], pg.core.wall_distances[i]), 1:pg.meta.N)
    println("   after reorder: particles whose geometry moved away from them: $bad of $(pg.meta.N)")
end
