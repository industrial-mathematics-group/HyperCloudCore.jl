using Test, HyperCloudCore, StaticArrays, LinearAlgebra, Random
t0 = time()
@testset "Shapes: signed distances, normals, tags, volumes" begin
    b = Cuboid((0.0, 0.0), (3.0, 1.0))
    @test signed_distance(b, SVector(1.0, 0.5)) == -0.5
    @test signed_distance(b, SVector(-1.0, 0.5)) == 1.0
    @test signed_distance(b, SVector(4.0, 2.0)) ≈ sqrt(2)                 # outside a corner: Euclidean
    @test boundary_normal(b, SVector(4.0, 2.0)) ≈ SVector(1, 1) / sqrt(2)
    @test boundary_normal(b, SVector(1.0, 0.9)) == SVector(0.0, 1.0)      # inside: the nearest face
    @test boundary_tag(b, SVector(-0.1, 0.5)) == 1 && boundary_tag(b, SVector(3.1, 0.5)) == 2
    @test boundary_tag(b, SVector(1.0, -0.1)) == 3 && boundary_tag(b, SVector(1.0, 1.1)) == 4
    @test boundary_tag(b, SVector(-0.2, -0.1)) == 1                       # the face it is farthest outside of
    @test boundary_tag(b, SVector(-0.1, -0.1)) == 1                       # ties: the lower tag
    @test shape_volume(b) == 3.0 && bounding_box(b) == (SVector(0.0, 0.0), SVector(3.0, 1.0))
    # an open axis has no faces: unbounded along it, no tag from it
    ob = Cuboid((0.0, 0.0), (3.0, 1.0); open = (true, false))
    @test signed_distance(ob, SVector(-5.0, 0.5)) == -0.5 && boundary_tag(ob, SVector(-5.0, 1.2)) == 4
    @test bounding_box(ob)[1][1] == -Inf && shape_volume(ob) == Inf
    @test_throws ArgumentError Cuboid((1.0,), (0.0,))
    # Float32 stays Float32
    bf = Cuboid((0f0,), (1f0,))
    @test signed_distance(bf, SVector(1.5f0)) === 0.5f0 && eltype(bf) == Float32 && ndims(bf) == 1

    ball = Ball((0.0, 0.0), 1.0)
    @test signed_distance(ball, SVector(0.0, 0.0)) == -1.0 && signed_distance(ball, SVector(0.0, 2.0)) == 1.0
    @test boundary_normal(ball, SVector(0.0, 2.0)) == SVector(0.0, 1.0) && boundary_tag(ball, SVector(0.0, 2.0)) == 1
    @test shape_volume(ball) ≈ π && shape_volume(Ball((0.0, 0.0, 0.0), 2.0)) ≈ 32π / 3 && shape_volume(Ball((0.0,), 1.0)) == 2
    @test_throws ArgumentError Ball((0.0,), 0.0)

    hs = HalfSpace((0.0, 0.0), (2.0, 0.0))                                # the normal is normalised
    @test hs.normal == SVector(1.0, 0.0) && signed_distance(hs, SVector(3.0, 7.0)) == 3.0
    @test boundary_normal(hs, SVector(-1.0, 0.0)) == SVector(1.0, 0.0)
    @test_throws ArgumentError HalfSpace((0.0,), (0.0,))
    @test_throws ArgumentError shape_volume(hs)                             # unbounded

    # the Mach 3 step: a channel minus a tagged block; the active piece provides normal and tag
    step = Tagged(Cuboid((0.6, -1.0), (4.0, 0.2)), t -> t + 4)
    dom  = b - step
    @test signed_distance(dom, SVector(1.0, 0.1)) > 0 && signed_distance(dom, SVector(0.5, 0.1)) < 0
    @test signed_distance(dom, SVector(0.5, 0.1)) ≈ -0.1                   # nearest: the step front
    @test boundary_tag(dom, SVector(0.65, 0.1)) == 5 && boundary_tag(dom, SVector(1.0, 0.15)) == 8
    @test boundary_normal(dom, SVector(0.65, 0.1)) ≈ SVector(1.0, 0.0)    # outward of the fluid: into the step
    @test boundary_normal(dom, SVector(1.0, 0.15)) ≈ SVector(0.0, -1.0)
    @test boundary_tag(dom, SVector(-0.1, 0.5)) == 1 && boundary_tag(dom, SVector(1.0, 1.1)) == 4
    @test isapprox(shape_volume(dom), 3.0 - 2.4 * 0.2; rtol = 1e-5)      # lattice integration
    @test shape_volume(Tagged(b, 7)) == 3.0 && boundary_tag(Tagged(b, 7), SVector(-1.0, 0.5)) == 7
    @test complement(complement(b)) === b && signed_distance(complement(b), SVector(1.0, 0.5)) == 0.5
    @test boundary_normal(complement(b), SVector(1.0, 0.9)) == SVector(0.0, -1.0)
    @test_throws ArgumentError union(b, Ball((0.0,), 1.0))

    # transforms and unions
    tb = Translated(ball, (2.0, 0.0))
    @test signed_distance(tb, SVector(2.0, 0.0)) == -1.0 && shape_volume(tb) ≈ π
    u = union(ball, tb)
    @test signed_distance(u, SVector(2.5, 0.0)) == -0.5 && boundary_normal(u, SVector(3.5, 0.0)) == SVector(1.0, 0.0)
    @test bounding_box(u) == (SVector(-1.0, -1.0), SVector(3.0, 1.0))
    @test isapprox(shape_volume(u), 2π; rtol = 1e-5)                      # two tangent unit discs: lattice integration
    @test bounding_box(intersect(b, hs)) == (SVector(0.0, 0.0), SVector(3.0, 1.0))
    @test signed_distance(intersect(b, HalfSpace((0.0, 0.0), (1.0, 1.0))), SVector(1.0, 1.0)) ≈ sqrt(2)
    rb = Rotated(Cuboid((-1.0, -0.5), (1.0, 0.5)), π / 2)
    @test signed_distance(rb, SVector(0.0, 0.9)) < 0 && signed_distance(rb, SVector(0.9, 0.0)) > 0
    @test boundary_normal(rb, SVector(0.0, 1.5)) ≈ SVector(0.0, 1.0) && boundary_tag(rb, SVector(0.0, 1.5)) == 2   # the box's x-max face
    lo, hi = bounding_box(rb)
    @test isapprox(lo, SVector(-0.5, -1.0); atol = 1e-12) && isapprox(hi, SVector(0.5, 1.0); atol = 1e-12)
    @test shape_volume(rb) == 2.0
    rc = Rotated(Cuboid((0.0, 0.0), (1.0, 1.0)), π / 2; center = (1.0, 0.0))   # about its lower right corner: now below the axis
    @test signed_distance(rc, SVector(0.5, -0.5)) < 0 && signed_distance(rc, SVector(0.5, 0.5)) > 0
    r3 = Rotated(Ball((0.0, 0.0, 0.0), 1.0), (0.0, 0.0, 1.0), 0.3)
    @test signed_distance(r3, SVector(0.5, 0.5, 0.5)) ≈ sqrt(0.75) - 1
    @test_throws ArgumentError Rotated(b, SMatrix{2, 2}(1.0, 1.0, 0.0, 1.0))

    # a user shape only needs the signed distance: the normal falls back to central differences
    struct TestParaboloid <: AbstractShape{2, Float64} end
    HyperCloudCore.signed_distance(::TestParaboloid, x::SVector{2, Float64}) = x[2] - x[1]^2   # not a true distance, a level set
    HyperCloudCore.bounding_box(::TestParaboloid) = (SVector(-1.0, -1.0), SVector(1.0, 1.0))
    n = boundary_normal(TestParaboloid(), SVector(1.0, 1.0))
    @test n ≈ SVector(-2.0, 1.0) / sqrt(5) atol = 1e-6
    @test boundary_tag(TestParaboloid(), SVector(1.0, 1.0)) == 1
end
println("shapes testset: ", round(time() - t0, digits = 1), " s"); t0 = time()
@testset "Shape domains: the box tags are the former ones, wall geometry per particle" begin
    for D in (1, 2, 3), per in (false, true)
        mins = ntuple(_ -> -1.0, D); maxs = ntuple(_ -> 1.0, D)
        isper = ntuple(d -> per && d == 1, D)
        geom = get_rectangular_domain(Float64, mins, maxs; is_periodic = isper)
        @test geom.shape isa Cuboid && geom.label == Val(:rectangular)
        dx = ntuple(_ -> 2.0 / (D == 3 ? 8 : 12), D)
        pg = ParticleGrid(geom, dx, ExponentialWeightFunction(1.0, 2.5 * dx[1]), 1;
                          randomness = ntuple(_ -> 0.2 * dx[1], D), rng = MersenneTwister(D))
        for i in 1:pg.meta.N
            x = pg.core.positions[i]
            inside = all(d -> mins[d] <= x[d] <= maxs[d], 1:D)
            @test pg.core.is_boundary[i] == !inside
            n = pg.core.normals[i]
            @test norm(n) ≈ 1
            if inside
                @test pg.core.wall_distances[i] <= 0
            else
                best_tag, best_out = 0, 0.0                       # the rule of get_rectangular_domain up to 0.8
                for d in 1:D
                    isper[d] && continue
                    o_lo = mins[d] - x[d]; o_hi = x[d] - maxs[d]
                    o_lo > best_out && (best_tag = 2d - 1; best_out = o_lo)
                    o_hi > best_out && (best_tag = 2d; best_out = o_hi)
                end
                @test pg.core.tags[i] == best_tag
                @test pg.core.wall_distances[i] > 0
                @test all(d -> (x[d] > maxs[d]) ? n[d] > 0 : (x[d] < mins[d]) ? n[d] < 0 : true, 1:D)
            end
        end
        @test domain_volume(pg) == 2.0^D
    end
    # a disc: radial normals, distance to the circle
    disc = get_spherical_domain(Float64, (0.0, 0.0), 1.0)
    @test disc.shape isa Ball && domain_volume(disc) ≈ π
    pg = ParticleGrid(disc, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
    for i in 1:pg.meta.N
        x = pg.core.positions[i]
        @test pg.core.wall_distances[i] ≈ norm(x) - 1 && (norm(x) == 0 || pg.core.normals[i] ≈ x / norm(x))
    end
    # a general shape: channel minus a block; its ghosts carry the block's tags and the volume is integrated
    shape = Cuboid((0.0, 0.0), (1.5, 1.0)) - Tagged(Cuboid((0.6, -1.0), (2.0, 0.2)), t -> t + 4)
    geom  = GeometricDomain(shape; bc_map = Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in (1, 2, 3, 4, 5, 8)))
    @test geom.label == Val(:shape) && geom.mins == SVector(0.0, 0.0) && geom.maxs == SVector(1.5, 1.0)
    pg = ParticleGrid(geom, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
    tags = unique(pg.core.tags[i] for i in 1:pg.meta.N if pg.core.is_boundary[i])
    @test sort(tags) == [1, 2, 3, 4, 5, 8]
    @test all(i -> !pg.core.is_boundary[i] || !(0.6 < pg.core.positions[i][1] <= 1.5 && 0 <= pg.core.positions[i][2] < 0.2) ||
                   pg.core.tags[i] in (5, 8), 1:pg.meta.N)                 # (beyond x = 1.5 the right wall may be the farther face)
    @test isapprox(domain_volume(pg), 1.5 - 0.9 * 0.2; rtol = 1e-5)
    # Behind the corner where the floor meets the step (outside the floor and inside the step) the max of the two
    # signed distances is not the distance to the region: the wall point is found by repeated projection, the
    # normal points to the corner. Every ghost's mirror image lies in the region.
    n, d = wall_geometry(geom, SVector(0.7, -0.05))
    @test n ≈ SVector(2.0, -1.0) / sqrt(5) && d ≈ sqrt(0.1^2 + 0.05^2)
    @test wall_geometry(geom, SVector(0.3, -0.05)) == (SVector(0.0, -1.0), 0.05)       # one projection: unchanged
    n2, d2 = wall_geometry(geom, SVector(0.5, 0.3))                                       # the step's top-front corner
    @test n2 ≈ SVector(1.0, -1.0) / sqrt(2) && d2 ≈ -sqrt(0.02)
    for i in 1:pg.meta.N
        pg.core.is_boundary[i] || continue
        xm = pg.core.positions[i] - 2 * pg.core.wall_distances[i] * pg.core.normals[i]
        @test signed_distance(shape, xm) <= 1e-12
    end
    # unbounded shapes need the box
    @test_throws ArgumentError GeometricDomain(HalfSpace((0.0, 0.0), (1.0, 0.0)))
    g2 = GeometricDomain(HalfSpace((0.5, 0.0), (1.0, 0.0)); mins = (0.0, 0.0), maxs = (1.0, 1.0))
    @test g2.mins == SVector(0.0, 0.0) && g2.get_tag(SVector(0.7, 0.5)) == 1 && g2.is_interior(SVector(0.3, 0.5))
    # the builder
    ctx = Dict{Symbol, Any}(:T => Float64, :D => 2)
    gb = HyperCloudCore.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => shape, :bc => Dict(1 => :outflow)), ctx)
    @test gb.shape === shape && gb.bc_map[1] isa OutflowBC
    gs = HyperCloudCore.build_domain(Dict{Symbol, Any}(:domain => "shape", :shape => "Ball((0.0, 0.0), 1.0)"), ctx)
    @test gs.shape isa Ball
    @test_throws ErrorException HyperCloudCore.build_domain(Dict{Symbol, Any}(:domain => :shape), ctx)
    # the former constructor still works, without wall geometry
    legacy = GeometricDomain(Val(:legacy_square), SVector(0.0, 0.0), SVector(1.0, 1.0), SVector(false, false),
                             x -> all(0 .<= x .<= 1), x -> all(0 .<= x .<= 1) ? 0 : 1, Dict{Int, AbstractBoundaryCondition}())
    @test legacy.shape === nothing
    HyperCloudCore.domain_volume(::GeometricDomain{Val{:legacy_square}}) = 1.0
    pgl = ParticleGrid(legacy, (0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
    @test all(isnan, view(pgl.core.wall_distances, 1:pgl.meta.N)) && all(x -> all(isnan, x), view(pgl.core.normals, 1:pgl.meta.N))
end
println("shape domains testset: ", round(time() - t0, digits = 1), " s")
