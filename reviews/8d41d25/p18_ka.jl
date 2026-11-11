include("common.jl")
h = 1/40
plate = Moving(Tagged(Cuboid((-0.2, -0.04), (0.2, 0.04)), t -> t + 4), rotating(1.0))
chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
slip(eq, U, wall) = State{1, Float64}(U[1] + wall.velocity[1] + 2wall.velocity[2])   # a probe op reading the velocity
geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                        bc_map = BCD(t => MirrorBC(slip; degree = 2) for t in (3, 4, 5, 6, 7, 8)))
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
ts = nothing
res = Dict()
for be in (:poly, :ka)
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1 + pg.core.positions[i][1]^2 + pg.core.positions[i][2]); end
    buf = copy(pg.rhos)
    try
        be == :ka && set_backend!(KABackend(HyperCloudCore.KA.CPU(); workgroupsize = 32))
        for tag in 5:8; MirrorBC(slip; degree = 2)(pg, buf, tag, nothing, eq, 0.3); end   # boundary_time 0, state time 0.3
        a = @allocated MirrorBC(slip; degree = 2)(pg, buf, 5, nothing, eq, 0.3)
        println(be, ": allocated ", a)
    finally
        set_backend!(PolyesterBackend())
    end
    res[be] = [buf[i][1] for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] >= 5]
end
println("max |poly - ka| on moving ghosts (gm not identity): ", maximum(abs.(res[:poly] .- res[:ka])))
# wall velocity at the ghost vs at the wall point
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
gh = ghosts(pg)
dv = maximum(gh) do i
    x = pg.core.positions[i]; n = pg.core.normals[i]; φ = pg.core.wall_distances[i]
    gm = H.tag_motion(geom, pg.core.tags[i], 0.0, 0.0)
    norm(H.velocity(gm, x) - wall_velocity(chan, x - φ * n, 0.0))
end
println("max |WallPoint.velocity (at the ghost) - wall_velocity at the wall point| over the plate ghosts (ω = 1): ", dv, " ; max ghost distance ", maximum(abs(pg.core.wall_distances[i]) for i in gh))
