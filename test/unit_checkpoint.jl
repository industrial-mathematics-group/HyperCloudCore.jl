# =========================================================================
# CHECKPOINTS: a run rebuilt from its configuration and given the state of a checkpoint continues bit for bit
# (uses the helpers and test equations of unit_block.jl)
# =========================================================================

# Runs `n1 + n2` steps straight, and `n1` steps, a checkpoint, a fresh run restored from it and `n2` more steps; returns
# the final states of both and the checkpoint. `make()` builds a run: `(; step!(k, t, dt), save(), restore!(ck), state())`
# (the step's time is k·dt in both runs, as the accumulated time of a driver is the same number in both).
function straight_and_resumed(make, n1::Int, n2::Int, dt)
    a = make()
    for k in 0:(n1 + n2 - 1); a.step!(k, k * dt, dt); end
    b = make()
    for k in 0:(n1 - 1); b.step!(k, k * dt, dt); end
    ck = b.save()
    c = make()
    c.restore!(ck)
    for k in n1:(n1 + n2 - 1); c.step!(k, k * dt, dt); end
    return a.state(), c.state(), ck
end

# The state of a grid that a restart must reproduce: the particle set, its geometry, the states, and the neighbour lists
# (as host arrays of the first N particles)
function grid_state(pg)
    N = pg.meta.N
    h(a) = collect(view(a, 1:N))
    nb = pg.neighbor
    return (N = N, positions = h(pg.core.positions), volumes = h(pg.core.volumes), is_boundary = h(pg.core.is_boundary),
            tags = h(pg.core.tags), rhos = h(pg.rhos), counts = h(nb.counts),
            indices = [nb.indices[k] for i in 1:N for k in nb.ranges[i]], weights = [nb.weights[k] for i in 1:N for k in nb.ranges[i]])
end

@testset "Checkpoints" begin
    sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
    eq8 = DiscreteTransport(Val(8), sym)
    eq2 = DiscreteTransport(Val(2), sym)
    box(x, k) = 1 + (abs(x) < 0.3 ? 1.0 : 0.0) * (1 + 0.1k)
    wave(x, k) = 1 + 0.3 * sin(π * x + 0.4k)
    periodic(N, W; mover = NoGridMover()) =
        ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), W; mover = mover)
    function set8!(pg, ic)
        for i in 1:pg.meta.N
            x = pg.core.positions[i][1]
            pg.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
        end
        return pg
    end
    # a run of the standard stepper: the grid from `grid()`, the initial condition `ic`, optional adaptivity after each step
    function standard(grid, ic; mood = MOOD(NoMOOD()), order = 2, limiter = MinmodLimiter(:hard), tableau = RK2_SSP_Tableau(Float64), adapt = nothing)
        return () -> begin
            pg = set8!(grid(), ic)
            ts = GeneralRKTimeStepper(eq8, MUSCL(Float64, 1, 8, order, limiter, RusanovFlux()), mood, tableau)
            step!(k, t, dt) = (ts(eq8, pg, t, dt); adapt === nothing || adapt_particles!(pg, adapt, order); nothing)
            (; step!, save = () -> checkpoint_state(pg), restore! = ck -> restore_checkpoint!(pg, ck), state = () -> grid_state(pg), pg)
        end
    end
    N  = 60
    dt = 0.3 * (2 / N) / 2.0

    @testset "Standard stepper: static clouds (periodic with MOOD, reflecting walls)" begin
        a, c, ck = straight_and_resumed(standard(() -> periodic(N, 8), box; mood = MOOD(EPD1(), MOODu1(1e-3)), order = 3, limiter = NoLimiter(),
                                                 tableau = RK3_SSP_Tableau(Float64)), 25, 25, dt)
        @test isequal(a, c)
        @test ck.format == HyperCloudCore.CHECKPOINT_FORMAT && ck.N == N && ck.T == "Float64" && ck.D == 1 && ck.M == 8
        @test ck.positions isa Vector{SVector{1, Float64}} && !hasproperty(ck, :blocks)
        bcw() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(ReflectOp()), 2 => MirrorBC(ReflectOp()))
        a, c, _ = straight_and_resumed(standard(() -> block_pg_1d(N, 8; bc = bcw()), box; tableau = RK3_SSP_Tableau(Float64)), 30, 30, dt)
        @test isequal(a, c)
    end

    @testset "Moving clouds: a Lagrangian mover across the periodic ends (the restore keeps the positions bit for bit)" begin
        mk = standard(() -> periodic(N, 8; mover = LagrangianMover()), wave)
        a, c, ck = straight_and_resumed(mk, 40, 40, 0.2 * dt)
        @test isequal(a, c)
        # the cloud moved, and the restore kept the positions bit for bit (the search runs without the periodic wrap, which
        # moves a position by rounding: -1 + mod(0.1 + 1, 2) = 0.10000000000000009)
        r = mk()
        restore_checkpoint!(r.pg, ck)
        @test view(r.pg.core.positions, 1:N) == ck.positions
        @test ck.positions != collect(view(periodic(N, 8).core.positions, 1:N))
        wrapped = [SVector(0.1)]
        HyperCloudCore._wrap_position!(1, wrapped, (true,), SVector(2.0), SVector(-1.0))
        @test wrapped[1] != SVector(0.1)
        moved = [SVector(0.1)]
        r.pg.core.positions[1] = moved[1]
        r.pg.neighbor(r.pg; wrap = false)
        @test r.pg.core.positions[1] === moved[1]
    end

    @testset "Adaptivity (the particle count changes) and moving pieces (the sweep, boundary and sweep times)" begin
        ad = ParticleAdaptivity(Float64)
        mover = PrescribedMover((x, t) -> SVector(1.0 * sinpi(x[1])))
        dta = 0.2 * (2 / N) / 3.0
        a, c, ck = straight_and_resumed(standard(() -> periodic(N, 8; mover = mover), wave; adapt = ad), 170, 60, dta)
        @test ck.N != N                                                          # the checkpoint was written after events
        @test isequal(a, c)
        tube() = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-0.5,))))
        bcm() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(ReflectOp()), 2 => MirrorBC(ReflectOp()))
        h = 0.025
        piston() = ParticleGrid(GeometricDomain(tube(); mins = (0.0,), maxs = (1.5,), bc_map = bcm()), (h,), ExponentialWeightFunction(1.0, 2.5h), 8)
        icp(x, k) = 1 + 0.2 * (1 + 0.1k) * exp(-20 * (x - 0.5)^2)
        N0 = piston().meta.N
        a, c, ck = straight_and_resumed(standard(piston, icp), 30, 30, 0.3 * h / 2.0)
        @test ck.N < N0 && ck.boundary_time > 0 && ck.sweep_time == ck.boundary_time
        @test isequal(a, c)
    end

    @testset "Block states: one stepper (a Lagrangian mover), a split over parts, and between the two" begin
        function blocks(; mover = LagrangianMover())
            return () -> begin
                pg = periodic(N, 2; mover = mover)
                bs = BlockState(eq2, pg)
                set_block_states!((x, b) -> State{2, Float64}(wave(x[1], 2b - 1), wave(x[1], 2b)), bs, pg)
                ts = BlockTimeStepper(eq2, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK2_SSP_Tableau(Float64))
                (; step! = (k, t, dt) -> ts(eq2, pg, bs, t, dt), save = () -> checkpoint_state(pg, bs), restore! = ck -> restore_checkpoint!(pg, bs, ck),
                   state = () -> (grid_state(pg), collect(view(parent(bs.data), 1:(bs.nb * bs.N[])))), pg, bs, ts)
            end
        end
        a, c, ck = straight_and_resumed(blocks(), 40, 40, 0.2 * dt)
        @test isequal(a, c)
        @test ck.nb == 4 && length(ck.blocks) == 4N
        # a split over host parts (MOOD with the cross-block admissibility, the parts in lockstep)
        g2 = GuardedTransport(eq2)
        mood = MOOD(EPD1(), MOODu1(1e-3))
        function split_run(ranges)
            return () -> begin
                parts = map(ranges) do r
                    pg = periodic(N, 2)
                    ts = BlockTimeStepper(g2, MUSCL(Float64, 1, 2, 3, NoLimiter(), RusanovFlux()), mood, (), RK3_SSP_Tableau(Float64); blocks = r)
                    bs = BlockState(g2, pg; blocks = r)
                    set_block_states!((x, b) -> State{2, Float64}(box(x[1], 2b - 1), box(x[1], 2b)), bs, pg)
                    BlockPart(pg, ts, bs)
                end
                devs = BlockDevices(g2, parts)
                (; step! = (k, t, dt) -> devs(t, dt), save = () -> checkpoint_state(devs), restore! = ck -> restore_checkpoint!(devs, ck),
                   state = () -> (grid_state(first(devs.parts).pg), host_state(devs)), devs)
            end
        end
        a, c, ck = straight_and_resumed(split_run([1:2, 3:4]), 25, 25, dt)
        @test isequal(a, c)
        # the checkpoint of one split restored into another split: the blocks go by the parts' ranges
        b = split_run([1:2, 3:4])()
        for k in 0:24; b.step!(k, k * dt, dt); end
        ck = b.save()
        d = split_run([1:1, 2:4])()
        restore_checkpoint!(d.devs, ck)
        @test host_state(d.devs) == ck.blocks
        @test all(pt -> view(pt.pg.core.positions, 1:N) == ck.positions && view(pt.pg.core.particle_orders, 1:N) == ck.particle_orders, d.devs.parts)
        # and a single block state restored from the split's checkpoint
        pg1 = periodic(N, 2)
        bs1 = BlockState(g2, pg1)
        restore_checkpoint!(pg1, bs1, ck)
        @test collect(view(parent(bs1.data), 1:4N)) == ck.blocks
    end

    @testset "Refusals: a checkpoint of another kind of run" begin
        r = standard(() -> periodic(N, 8), box)()
        ck = r.save()
        pg32 = ParticleGrid(get_rectangular_domain(Float32, (-1.0f0,), (1.0f0,); is_periodic = true), (2.0f0 / N,), ExponentialWeightFunction(1.0f0, 2.5f0 * 2 / N), 8)
        @test_throws ArgumentError restore_checkpoint!(pg32, ck)                 # float type
        @test_throws ArgumentError restore_checkpoint!(periodic(2N, 8), ck)       # spacing (and radius)
        @test_throws ArgumentError restore_checkpoint!(periodic(N, 2), ck)        # state length (M)
        @test_throws ArgumentError restore_checkpoint!(r.pg, (; ck..., format = 0))
        @test_throws ArgumentError restore_checkpoint!(r.pg, (; ck..., N = N + 1))
        # a grid's checkpoint into a block state and the other way round, a block count that does not match
        pgb = periodic(N, 2)
        bs = BlockState(eq2, pgb)
        @test_throws ArgumentError restore_checkpoint!(pgb, bs, checkpoint_state(pgb))
        ckb = checkpoint_state(pgb, bs)
        @test_throws ArgumentError restore_checkpoint!(pgb, ckb)
        @test_throws ArgumentError restore_checkpoint!(pgb, bs, (; ckb..., nb = 3))
        bsp = BlockState(eq2, pgb; blocks = 2:4)
        @test_throws ArgumentError restore_checkpoint!(pgb, bsp, ckb)            # a part's state alone
        @test_throws ArgumentError checkpoint_state(periodic(2N, 2), bs)         # a block state of another particle count
    end
end
