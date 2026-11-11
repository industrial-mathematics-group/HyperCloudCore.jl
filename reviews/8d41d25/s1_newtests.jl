include("common.jl"); using Test, Random
@testset "new tests of 8d41d25" begin
    @testset "Sweeps in 2-D: independent events, periodic seams, static walls behind a receding piece" begin
        h = 1 / 40
        f(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
        function piston_grid(w; periodic = false)
            piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
            box  = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, periodic))
            geom = GeometricDomain(intersect(box, piston); is_periodic = (false, periodic), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                                   bc_map = BCD(t => ExtrapolationBC(1) for t in (periodic ? (1, 2, 5) : (1, 2, 3, 4, 5))))
            pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
            for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end
            return pg
        end
        function advance!(pg, t)
            place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t)
            pg.neighbor(pg)
            return sweep_boundaries!(pg, pg.meta.sweep_time, t, 3)
        end
        inner(pg) = [i for i in 1:pg.meta.N if !pg.core.is_boundary[i]]
        # one sweep that removes a whole row (20 particles): the particles away from the face keep their state and
        # volume (all events at once on the old topology changed them by 0.2 and by factors 0.5-2.8 up to 12 h away)
        pg = piston_grid(-0.5)
        before = Dict(pg.core.positions[i] => (pg.rhos[i][1], pg.core.volumes[i]) for i in inner(pg))
        r = advance!(pg, 1.5h)                                          # the face at 1 - 0.75 h
        @test r.removed >= 20
        far = [i for i in inner(pg) if pg.core.positions[i][1] < 1 - 6h]
        @test maximum(abs(pg.rhos[i][1] - before[pg.core.positions[i]][1]) for i in far) < 0.04             # measured 0.023
        @test all(i -> 0.92 < pg.core.volumes[i] / before[pg.core.positions[i]][2] < 1.08, far)               # 0.95 .. 0.99
        # receding, periodic across the face: the box spans the periodic axis; every lattice point once (the seam
        # was inserted twice, coincident particles) and the volumes follow the region
        pg = piston_grid(1.0; periodic = true)
        t = 0.0
        for _ in 1:20; t += 0.2h; advance!(pg, t); end
        ii = inner(pg)
        @test minimum(sqrt(minimum(sum(abs2, pg.neighbor.distances[k]) for k in pg.neighbor.ranges[i])) for i in ii) >= 0.5h - 1e-12
        @test abs(sum(pg.core.volumes[i] for i in ii) - 0.5 * (1 + t)) < 0.25h * 0.5
        # receding next to static walls: their ghost band and wall rows follow (the band was the one at t = 0)
        pg = piston_grid(1.0)
        t = 0.0
        for _ in 1:40; t += 0.2h; advance!(pg, t); end
        face = 1 + t
        bottom = [pg.core.positions[i] for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 3]
        @test maximum(x -> x[1], bottom) >= face - h
        @test count(i -> abs(pg.core.positions[i][2] - h) < 1e-12 && 1 + h < pg.core.positions[i][1] < face - h, inner(pg)) >= 5
        @test all(i -> pg.core.volumes[i] == 0, [i for i in 1:pg.meta.N if pg.core.is_boundary[i]])
        # a piece faster than the documented limit (1.5 spacings per sweep): particles it covered whole have no volume
        # left and go without a remap (they made the remaps divide 0 by 0: NaN up to 10 h from the face)
        pg = piston_grid(-0.5)
        t = 0.0
        for _ in 1:4; t += 3h; advance!(pg, t); end
        @test all(i -> isfinite(pg.rhos[i][1]) && isfinite(pg.core.volumes[i]), 1:pg.meta.N)
        @test abs(sum(pg.core.volumes[i] for i in inner(pg)) - 0.5 * (1 - 0.5t)) < 0.5h
        # a piece reaching across a periodic end is refused (no periodic images of pieces)
        plate = Moving(Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4), translating((1.0, 0.0)))
        chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
        pgp   = ParticleGrid(GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                                             bc_map = BCD(t => MirrorBC(; degree = 1) for t in (3, 4, 5, 6, 7, 8))),
                             (1 / 20, 1 / 20), ExponentialWeightFunction(1.0, 0.125), 1)
        place_ghosts!(pgp, pgp.core.positions, 0.0, 0.85)
        pgp.neighbor(pgp)
        @test_throws ArgumentError sweep_boundaries!(pgp, 0.0, 0.85, 3)
    end

    @testset "Moving pieces: frames, float types, wall velocity, refused combinations" begin
        ball = Tagged(Ball((0.0, 0.0), 0.2), 5)
        # a Moving piece under a transformation would move in the wrapper's frame: refused
        @test_throws ArgumentError GeometricDomain(Cuboid((-1.0, -1.0), (1.0, 1.0)) - Translated(Moving(ball, rotating(1.0)), (0.3, 0.0)))
        @test_throws ArgumentError wall_velocity(Cuboid((-1.0, -1.0), (1.0, 1.0)) - Rotated(Moving(ball, translating((1.0, 0.0))), π / 2),
                                                 SVector(0.0, 0.3), 0.1)
        # motions in another float type than the shape are converted
        m32 = Moving(Ball((0f0, 0f0), 0.2f0), rotating(1f0))
        @test HyperCloudCore.pose(m32, 0.5) isa RigidMotion{2, Float32}
        @test HyperCloudCore.pose(Moving(Ball((0f0, 0f0), 0.2f0), translating((1.0, 0.0))), 0.5).c == SVector(0.5f0, 0f0)
        # the wall velocity a MirrorBC operator receives is the body's at the wall point (it was taken at the ghost)
        vy(eq, U, wall) = State{1, Float64}(wall.velocity[2])
        plate = Moving(Tagged(Cuboid((-0.2, -0.04), (0.2, 0.04)), t -> t + 4), rotating(1.0))
        chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
        geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5),
                                bc_map = BCD(t => MirrorBC(vy; degree = 1) for t in (3, 4, 5, 6, 7, 8)))
        h  = 1 / 40
        pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
        eq0 = LinearAdvection(((0.0,), (0.0,)), Float64)
        buf = copy(pg.rhos)
        for tag in 5:8; MirrorBC(vy; degree = 1)(pg, buf, tag, nothing, eq0, 0.0); end
        gh = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] >= 5]
        xw(i) = pg.core.positions[i] - pg.core.wall_distances[i] * pg.core.normals[i]
        @test maximum(abs(buf[i][1] - wall_velocity(chan, xw(i), 0.0)[2]) for i in gh) < 1e-12
        # moving pieces need a static interior, and the adaptivity's volume estimators integrate the region at t = 0
        pgl = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; mover = LagrangianMover())
        @test_throws ArgumentError HyperCloudCore._sync_boundaries!(pgl, 0.0, 3)
        @test_throws ArgumentError adapt_particles!(pg, ParticleAdaptivity(Float64; volume_estimator = ShepardVolumes()), 3)
        # ... so the builder gives a static cloud with moving pieces no estimator (it gave Shepard: the first call threw)
        ad = HyperCloudCore.build_adaptivity(Dict{Symbol, Any}(:enabled => true), Dict{Symbol, Any}(:T => Float64, :Grid => pg))
        @test ad.volume_estimator === nothing
        @test adapt_particles!(pg, ad, 3) isa NamedTuple
    end
@testset "string shapes" begin
ctx = Dict{Symbol, Any}(:T => Float64, :D => 2)
    gm = HyperCloudCore.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => "r = T(0.5); Ball((T(0), T(0)), r)"), Dict{Symbol, Any}(:T => Float32, :D => 2))
    @test gm.shape isa Ball{2, Float32}
    @test_throws ArgumentError HyperCloudCore.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => "Ball((T(0), T(0)), "), ctx)
end
end
