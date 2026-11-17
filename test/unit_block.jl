# =========================================================================
# BLOCK PIPELINE: the stage kernels of GeneralRKTimeStepper driven block by block on a BlockState
# (static grids, explicit tableaus; the state length nb · W is a run-time number)
# =========================================================================

"Periodic (or tagged) 1-D particle grid on [-1, 1] with `W` components per particle."
function block_pg_1d(N::Int, W::Int; T = Float64, bc = nothing, range = 2.5)
    dx   = T(2) / N
    geom = bc === nothing ? get_rectangular_domain(T, (-1.0,), (1.0,); is_periodic = true) :
                            get_rectangular_domain(T, (-1.0,), (1.0,); bc_map = bc)
    return ParticleGrid(geom, (dx,), ExponentialWeightFunction(one(T), T(range) * dx), W)
end

"Periodic 2-D particle grid on [0, 1]² with `W` components per particle."
function block_pg_2d(N::Int, W::Int; T = Float64, range = 2.5)
    dx   = one(T) / N
    geom = get_rectangular_domain(T, (0.0, 0.0), (1.0, 1.0); is_periodic = true)
    return ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(one(T), T(range) * dx), W)
end

"Advances both pipelines to `tend` with the step `dt` (the last step is shortened)."
function advance!(step!, tend, dt)
    t = 0.0
    while t < tend - 1e-12
        d = min(dt, tend - t)
        step!(t, d)
        t += d
    end
    return t
end

"Largest difference between the block state and the full state vector (component `(b − 1) W + k` of particle `i`)."
function block_vs_full(bs::BlockState{W}, full) where {W}
    dev = 0.0
    for b in 1:bs.nb
        blk = block_view(bs, b)
        for i in eachindex(blk), k in 1:W
            dev = max(dev, abs(blk[i][k] - full[i][(b - 1) * W + k]))
        end
    end
    return dev
end

# A boundary condition with the six-argument signature of earlier versions: the dispatcher must still call it
struct OldStyleBC <: AbstractBoundaryCondition
    value::Float64
end
function (bc::OldStyleBC)(pg::ParticleGrid, rhos_buffer, tag::Int, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (rhos_buffer[i] = State{1, Float64}(bc.value))
    end
    return nothing
end

@testset "Block pipeline" begin
    @testset "State arrays are passed explicitly: a separate array advances like pg.rhos, the BC reference" begin
        eq = Burgers()
        N  = 60
        u0(x) = 0.5 + 0.4 * sin(π * x)
        pg = block_pg_1d(N, 1)
        for i in 1:N
            pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i][1]))
        end
        U = copy(pg.rhos)
        interp = MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux())
        ts = GeneralRKTimeStepper(eq, interp, MOOD(NoMOOD()), RK3_SSP_Tableau(Float64))
        dt = 0.3 * (2 / N) / 0.9
        for _ in 1:20
            ts(eq, pg, 0.0, dt, U)              # the separate array
        end
        @test U != pg.rhos                      # pg.rhos untouched
        for _ in 1:20
            ts(eq, pg, 0.0, dt)                 # the grid's own state
        end
        @test U == pg.rhos                      # the same arithmetic
        @test total_mass(pg, U) == total_mass(pg) && interior_mass(pg, U) == interior_mass(pg)
        @test get_time_step(pg, eq, interp; U = U) == get_time_step(pg, eq, interp)
        # a moving grid keeps its own state
        pgm = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,),
                           ExponentialWeightFunction(1.0, 2.5 * 2 / N), 1; mover = LagrangianMover())
        @test_throws ArgumentError ts(eq, pgm, 0.0, dt, copy(pgm.rhos))

        # FixedDirichlet / OutflowBC read the reference array; an old six-argument condition is still dispatched
        bc = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
        pgb = block_pg_1d(N, 1; bc = bc)
        Nb  = pgb.meta.N
        ref = [State{1, Float64}(7.0) for _ in 1:Nb]
        buf = [State{1, Float64}(Float64(i)) for i in 1:Nb]
        HyperCloudCore.apply_boundary_conditions!(pgb, buf, ts, eq, 0.0, ref)
        left = findall(i -> pgb.core.is_boundary[i] && pgb.core.tags[i] == 1, 1:Nb)
        @test !isempty(left) && all(i -> buf[i] == State{1, Float64}(7.0), left)
        pgo = block_pg_1d(N, 1; bc = Dict{Int, AbstractBoundaryCondition}(1 => OldStyleBC(3.0), 2 => OldStyleBC(4.0)))
        bufo = [State{1, Float64}(0.0) for _ in 1:pgo.meta.N]
        HyperCloudCore.apply_boundary_conditions!(pgo, bufo, ts, eq, 0.0, bufo)
        @test all(i -> !pgo.core.is_boundary[i] || bufo[i][1] == (pgo.core.tags[i] == 1 ? 3.0 : 4.0), 1:pgo.meta.N)
    end

    @testset "Any equation is a one-block equation: bit-identical to the standard stepper (Burgers, RK3, MUSCL2)" begin
        eq = Burgers()
        @test num_blocks(eq) == 1
        N  = 80
        u0(x) = 0.5 + 0.4 * sin(π * x)
        pg_s, pg_b = block_pg_1d(N, 1), block_pg_1d(N, 1)
        for i in 1:N
            pg_s.rhos[i] = State{1, Float64}(u0(pg_s.core.positions[i][1]))
        end
        interp_s = MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux())
        interp_b = MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux())
        ts_s = GeneralRKTimeStepper(eq, interp_s, MOOD(NoMOOD()), RK3_SSP_Tableau(Float64))
        ts_b = BlockTimeStepper(eq, interp_b, RK3_SSP_Tableau(Float64))
        bs   = BlockState(eq, pg_b)
        set_block_states!((x, b) -> State{1, Float64}(u0(x[1])), bs, pg_b)
        @test bs.nb == 1 && length(block_view(bs, 1)) == N
        @test block_view(bs, 1) == view(pg_s.rhos, 1:N)
        dt = 0.3 * (2 / N) / 0.9
        advance!((t, d) -> (ts_s(eq, pg_s, t, d); ts_b(eq, pg_b, bs, t, d)), 0.6, dt)      # through the shock
        @test block_view(bs, 1) == view(pg_s.rhos, 1:N)                                      # bit for bit
        @test get_time_step(pg_b, eq, interp_b, bs) == get_time_step(pg_s, eq, interp_s)
        @test BlockOf(eq, 1) isa HyperbolicPDE{1, 1, Float64}
        @test flux(BlockOf(eq, 1), State{1, Float64}(2.0)) == flux(eq, State{1, Float64}(2.0))
        @test max_eigenvalue(BlockOf(eq, 1), State{1, Float64}(-2.0), 1) == 2.0
    end

    @testset "Discrete transport, 1-D: blocks reproduce the full state vector, converge to the exact solution" begin
        speeds = [(-1.5,), (-0.5,), (0.25,), (0.75,), (1.0,), (1.25,), (1.75,), (2.0,)]
        eq_f = DiscreteTransport(Val(8), speeds)           # one block of 8 (the standard pipeline, M = 8)
        eq_b = DiscreteTransport(Val(2), speeds)           # four blocks of 2
        @test num_blocks(eq_f) == 1 && num_blocks(eq_b) == 4
        ic(x, k)       = 1 + 0.3 * sin(π * x + 0.4k)
        exact(x, k, t) = ic(x - transport_speed(eq_b, k)[1] * t, k)      # (2-periodic in x)
        tend = 0.5
        errs = Float64[]
        devs = Float64[]
        for N in (50, 100)
            dx = 2 / N
            pg_f = block_pg_1d(N, 8)
            for i in 1:N
                x = pg_f.core.positions[i][1]
                pg_f.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
            end
            pg_b = block_pg_1d(N, 2)
            bs   = BlockState(eq_b, pg_b)
            set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg_b)
            interp_f = MUSCL(Float64, 1, 8, 2, VenkatakrishnanLimiter(:hard), RusanovFlux())
            interp_b = MUSCL(Float64, 1, 2, 2, VenkatakrishnanLimiter(:hard), RusanovFlux())
            ts_f = GeneralRKTimeStepper(eq_f, interp_f, MOOD(NoMOOD()), RK2_SSP_Tableau(Float64))
            ts_b = BlockTimeStepper(eq_b, interp_b, RK2_SSP_Tableau(Float64))
            @test get_time_step(pg_b, eq_b, interp_b, bs) == get_time_step(pg_f, eq_f, interp_f)
            dt = 0.3 * dx / 2.0
            t = advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), tend, dt)
            push!(devs, block_vs_full(bs, view(pg_f.rhos, 1:N)))
            err = 0.0
            for b in 1:4, (i, x) in enumerate(view(pg_b.core.positions, 1:N)), k in 1:2
                err += abs(block_view(bs, b)[i][k] - exact(x[1], 2(b - 1) + k, t))
            end
            push!(errs, err / (8N))
        end
        # The same arithmetic per component up to round-off: the Venkatakrishnan limiter's tolerance depends on the state
        # vector as a whole (with MinmodLimiter the states are bit-identical, see the 1-block and the 2-D tests)
        @test all(<(1e-14), devs)
        @test errs[2] < 2e-3
        @test log2(errs[1] / errs[2]) > 1.6                            # second order (MUSCL2 / RK2, VK on smooth data)
    end

    @testset "Discrete transport, 2-D periodic: blocks reproduce the full state vector" begin
        speeds = [(1.0, 0.5), (-0.5, 1.0), (0.3, -0.7), (-1.0, -0.25)]
        eq_f = DiscreteTransport(Val(4), speeds)
        eq_b = DiscreteTransport(Val(2), speeds)
        ic(x, k) = 1 + 0.2 * sin(2π * (x[1] + 0.5 * x[2]) + 0.3k)
        N = 16
        pg_f = block_pg_2d(N, 4)
        for i in 1:pg_f.meta.N
            x = pg_f.core.positions[i]
            pg_f.rhos[i] = State{4, Float64}(ntuple(k -> ic(x, k), 4))
        end
        pg_b = block_pg_2d(N, 2)
        bs   = BlockState(eq_b, pg_b)
        set_block_states!((x, b) -> State{2, Float64}(ic(x, 2b - 1), ic(x, 2b)), bs, pg_b)
        interp_f = MUSCL(Float64, 2, 4, 2, BarthJespersenLimiter(:hard), RusanovFlux())
        interp_b = MUSCL(Float64, 2, 2, 2, BarthJespersenLimiter(:hard), RusanovFlux())
        ts_f = GeneralRKTimeStepper(eq_f, interp_f, MOOD(NoMOOD()), RK2_SSP_Tableau(Float64))
        ts_b = BlockTimeStepper(eq_b, interp_b, RK2_SSP_Tableau(Float64))
        dt = 0.3 / N / 2
        t  = advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), 0.1, dt)
        @test block_vs_full(bs, view(pg_f.rhos, 1:pg_f.meta.N)) < 1e-14
        err = sum(abs(block_view(bs, b)[i][k] - ic(x - transport_speed(eq_b, 2(b - 1) + k) * t, 2(b - 1) + k))
                  for b in 1:2, (i, x) in enumerate(view(pg_b.core.positions, 1:pg_b.meta.N)), k in 1:2) / (4 * pg_b.meta.N)
        @test err < 0.05                     # (a coarse 16² grid and a limiter clipping the extrema: the convergence check is the 1-D one)
    end

    @testset "Boundary conditions per block (inflow Dirichlet, outflow) and the KernelAbstractions CPU backend" begin
        speeds = [(0.5,), (1.0,), (1.5,), (2.0,)]                                 # all to the right
        eq_f = DiscreteTransport(Val(4), speeds)
        eq_b = DiscreteTransport(Val(2), speeds)
        ic(x, k) = 1 + 0.5 * exp(-40 * (x + 0.3)^2) * k
        bc() = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
        function run(backend)
            set_backend!(backend)
            try
                N = 60
                pg_f = block_pg_1d(N, 4; bc = bc())
                for i in 1:pg_f.meta.N
                    x = pg_f.core.positions[i][1]
                    pg_f.rhos[i] = State{4, Float64}(ntuple(k -> ic(x, k), 4))
                end
                pg_b = block_pg_1d(N, 2; bc = bc())
                bs   = BlockState(eq_b, pg_b)
                set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg_b)
                interp_f = MUSCL(Float64, 1, 4, 2, MinmodLimiter(:hard), RusanovFlux())
                interp_b = MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux())
                ts_f = GeneralRKTimeStepper(eq_f, interp_f, MOOD(NoMOOD()), RK3_SSP_Tableau(Float64))
                ts_b = BlockTimeStepper(eq_b, interp_b, RK3_SSP_Tableau(Float64))
                dt = 0.3 * (2 / N) / 2
                advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), 3.0, dt)       # every pulse has left
                return block_vs_full(bs, view(pg_f.rhos, 1:pg_f.meta.N)), bs, pg_b
            finally
                set_backend!(PolyesterBackend())
            end
        end
        dev, bs, pg_b = run(PolyesterBackend())
        @test dev == 0.0
        # the pulse left through the outflow, the inflow keeps its state
        Nb = pg_b.meta.N
        inflow = findall(i -> pg_b.core.is_boundary[i] && pg_b.core.tags[i] == 1, 1:Nb)
        @test all(i -> block_view(bs, 1)[i][1] ≈ ic(pg_b.core.positions[i][1], 1), inflow)
        @test maximum(i -> maximum(abs.(block_view(bs, 2)[i] .- State{2, Float64}(1.0, 1.0))), 1:Nb) < 1e-3   # gone
        dev_ka, _, _ = run(KABackend(HyperCloudCore.KA.CPU()))
        @test dev_ka < 1e-12
    end

    @testset "Scope checks" begin
        speeds = [(0.5,), (1.0,), (1.5,), (2.0,)]
        eq_b = DiscreteTransport(Val(2), speeds)
        interp = MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux())
        @test_throws ArgumentError BlockTimeStepper(eq_b, interp, IMEX_ARS222_Tableau(Float64))     # explicit only
        ts = BlockTimeStepper(eq_b, interp, RK2_SSP_Tableau(Float64))
        pgm = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (0.05,),
                           ExponentialWeightFunction(1.0, 0.125), 2; mover = LagrangianMover())
        bsm = BlockState(eq_b, pgm)
        @test_throws ArgumentError ts(eq_b, pgm, bsm, 0.0, 0.01)                                    # static grids only
        pg = block_pg_1d(40, 2)
        other = DiscreteTransport(Val(2), speeds[1:2])                                              # one block of 2
        @test_throws ArgumentError ts(eq_b, pg, BlockState(other, pg), 0.0, 0.01)                   # block count mismatch
        bs = BlockState(eq_b, pg)
        @test length(parent(bs.data)) >= 2 * pg.meta.N && bs.nb == 2
    end
end
