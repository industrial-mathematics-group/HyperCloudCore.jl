include("common.jl")
h = 1/40
for motion in (rotating(1.0), translating((1.0, 0.0)))
    plate = Moving(Tagged(Cuboid((-0.2, -0.04), (0.2, 0.04)), t -> t + 4), motion)
    chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
    geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => MirrorBC() for t in 3:8))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    H.tag_motion(geom, 5, 0.0, 0.3); H.tag_motions(geom, 0.0, 0.3)
    a1 = @allocated H.tag_motion(geom, 5, 0.0, 0.3)
    a2 = @allocated H.tag_motions(geom, 0.0, 0.3)
    X = pg.core.positions; place_ghosts!(pg, X, 0.0, 0.0)
    a3 = @allocated place_ghosts!(pg, X, 0.0, 0.0)
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    bc = MirrorBC(); buf = copy(pg.rhos); bc(pg, buf, 5, nothing, eq, 0.3); bc(pg, buf, 3, nothing, eq, 0.3)
    a4 = @allocated bc(pg, buf, 5, nothing, eq, 0.3)
    a5 = @allocated bc(pg, buf, 3, nothing, eq, 0.3)
    println(typeof(motion).name.name, ": tag_motion $a1 B, tag_motions $a2 B, place_ghosts! $a3 B, MirrorBC moving tag $a4 B, static tag $a5 B")
end
