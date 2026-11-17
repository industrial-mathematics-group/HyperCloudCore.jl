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

# A source that needs the reduction over the blocks: relaxation of every component towards the particle's mean over
# all nb · W components, S_k = (m − f_k) / τ, which conserves the mean (so the implicit stage has the closed form
# f_k = (f̃_k + λ m̃) / (1 + λ), λ = Δt a_ii / τ, exactly like the BGK collision). The block form and the standard form
# sum the components in the same order, so the two pipelines are compared bit for bit.
struct MeanRelaxation{T} <: AbstractImplicitBlockSource; tau::T; end
struct ExplicitMeanRelaxation{T} <: AbstractExplicitBlockSource; tau::T; end
const _MeanRelax = Union{MeanRelaxation, ExplicitMeanRelaxation}
HyperCloudCore.reduction_type(::_MeanRelax) = Float64
@inline function HyperCloudCore.block_reduce(st::_MeanRelax, p, Y, N, nb, pg, t, dt_coeff)
    s, n = 0.0, 0
    @inbounds for b in 1:nb
        U = Y[(b - 1) * N + p]
        for k in eachindex(U)
            s += U[k]
        end
        n += length(U)
    end
    return s / n
end
@inline HyperCloudCore.block_source(st::_MeanRelax, U, b, m, p, pg, t) = (m .- U) ./ st.tau
@inline function HyperCloudCore.block_implicit_solve(st::MeanRelaxation, U, b, m, dt_coeff, p, pg, t)
    λ = dt_coeff / st.tau
    return (U .+ λ * m) ./ (1 + λ)
end
struct MeanRelaxationFull{T} <: AbstractImplicitSourceTerm; tau::T; end
struct ExplicitMeanRelaxationFull{T} <: AbstractExplicitSourceTerm; tau::T; end
@inline function mean_full(U)
    s = 0.0
    @inbounds for k in eachindex(U)
        s += U[k]
    end
    return s / length(U)
end
@inline HyperCloudCore.evaluate_source(st::Union{MeanRelaxationFull, ExplicitMeanRelaxationFull}, U, p::Int, pg, t::Real) = (mean_full(U) .- U) ./ st.tau
@inline function HyperCloudCore.implicit_solve(st::MeanRelaxationFull, U, dt_coeff::Real, p::Int, pg, t::Real)
    λ = dt_coeff / st.tau
    return (U .+ λ * mean_full(U)) ./ (1 + λ)
end

"Both pipelines on the 8-velocity transport with the mean relaxation; returns (deviation, block state, grid)."
function relaxation_runs(sources_block, sources_full, tableau; N = 50, tend = 0.4, backend = PolyesterBackend())
    set_backend!(backend)
    try
        speeds = [(-1.5,), (-0.5,), (0.25,), (0.75,), (1.0,), (1.25,), (1.75,), (2.0,)]
        eq_f = DiscreteTransport(Val(8), speeds)
        eq_b = DiscreteTransport(Val(2), speeds)
        ic(x, k) = 1 + 0.3 * sin(π * x + 0.4k)
        pg_f = block_pg_1d(N, 8)
        for i in 1:N
            x = pg_f.core.positions[i][1]
            pg_f.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
        end
        pg_b = block_pg_1d(N, 2)
        bs   = BlockState(eq_b, pg_b)
        set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg_b)
        interp_f = MUSCL(Float64, 1, 8, 2, MinmodLimiter(:hard), RusanovFlux())
        interp_b = MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux())
        ts_f = GeneralRKTimeStepper(eq_f, interp_f, MOOD(NoMOOD()), sources_full, tableau)
        ts_b = BlockTimeStepper(eq_b, interp_b, sources_block, tableau)
        dt = 0.3 * (2 / N) / 2.0
        advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), tend, dt)
        return block_vs_full(bs, view(pg_f.rhos, 1:N)), bs, pg_b
    finally
        set_backend!(PolyesterBackend())
    end
end



# A block equation with a cross-block admissibility: the mean over all components must stay above 1.5 (and the standard
# form of it), for the PAD of MOOD on blocks
struct GuardedTransport{D, W, T, EQ <: DiscreteTransport{D, W, T}} <: HyperbolicPDE{D, W, T}
    eq::EQ
end
GuardedTransport(eq::DiscreteTransport{D, W, T}) where {D, W, T} = GuardedTransport{D, W, T, typeof(eq)}(eq)
HyperCloudCore.num_blocks(g::GuardedTransport) = HyperCloudCore.num_blocks(g.eq)
@inline HyperCloudCore.block_flux(g::GuardedTransport, U::State, b::Int) = HyperCloudCore.block_flux(g.eq, U, b)
@inline HyperCloudCore.block_max_eigenvalue(g::GuardedTransport, U::State, d::Int, b::Int) = HyperCloudCore.block_max_eigenvalue(g.eq, U, d, b)
@inline flux(g::GuardedTransport, U::State) = flux(g.eq, U)
@inline max_eigenvalue(g::GuardedTransport, U::State, d::Int) = max_eigenvalue(g.eq, U, d)
@inline function HyperCloudCore.block_admissible(g::GuardedTransport{D, W, T}, Y, N::Int, nb::Int, p::Int) where {D, W, T}
    s = zero(T)
    @inbounds for b in 1:nb
        s += sum(Y[(b - 1) * N + p])
    end
    return s / (nb * W) > T(1.5)
end
@inline HyperCloudCore.is_admissible(g::GuardedTransport{D, W, T}, U::State{W, T}) where {D, W, T} = all(isfinite, U) && sum(U) / W > T(1.5)

"Both pipelines on the 8-velocity transport with a box per component (a discontinuity), MUSCL3 without a limiter, with MOOD."
function mood_runs(eq_f, eq_b, mood, tableau; sources_b = (), sources_f = (), N = 80, tend = 0.3, order = 3)
    box(x, k) = 1 + (abs(x) < 0.3 ? 1.0 : 0.0) * (1 + 0.1k)
    pg_f = block_pg_1d(N, 8)
    for i in 1:N
        x = pg_f.core.positions[i][1]
        pg_f.rhos[i] = State{8, Float64}(ntuple(k -> box(x, k), 8))
    end
    pg_b = block_pg_1d(N, 2)
    bs   = BlockState(eq_b, pg_b)
    set_block_states!((x, b) -> State{2, Float64}(box(x[1], 2b - 1), box(x[1], 2b)), bs, pg_b)
    interp_f = MUSCL(Float64, 1, 8, order, NoLimiter(), RusanovFlux())
    interp_b = MUSCL(Float64, 1, 2, order, NoLimiter(), RusanovFlux())
    ts_f = GeneralRKTimeStepper(eq_f, interp_f, mood, sources_f, tableau)
    ts_b = BlockTimeStepper(eq_b, interp_b, mood, sources_b, tableau)
    dt = 0.3 * (2 / N) / 2.0
    advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), tend, dt)
    return block_vs_full(bs, view(pg_f.rhos, 1:N)), ts_f, ts_b, bs, pg_f, pg_b
end
block_max(bs) = maximum(maximum(block_view(bs, b)[i]) for b in 1:bs.nb, i in 1:bs.N[])


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
        @test BlockTimeStepper(eq_b, interp, IMEX_ARS222_Tableau(Float64)) isa BlockTimeStepper        # an IMEX tableau without implicit sources is fine
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

    @testset "Block sources: the reduction over the blocks, implicit (IMEX) and explicit, bit-identical to the standard stepper" begin
        # (the stage states and K_I are bit-identical; the flux pass compiled with another argument tuple differs by an
        # ulp in K_E: fused multiply-adds in the MLS sums, not a formula difference)
        # implicit, ARS222 (its first stage has a_11 = 0: the explicit evaluation path of an implicit source)
        dev, bs, pg_b = relaxation_runs((BlockSourceTerm(MeanRelaxation(0.05)),), (MeanRelaxationFull(0.05),), IMEX_ARS222_Tableau(Float64))
        @test dev < 1e-13
        # stiff: τ ≪ Δt, the components collapse onto the mean (the closed-form solve is exact for any stiffness)
        dev, bs, pg_b = relaxation_runs((BlockSourceTerm(MeanRelaxation(1e-4)),), (MeanRelaxationFull(1e-4),), IMEX_ARS222_Tableau(Float64))
        @test dev < 1e-13
        N = pg_b.meta.N
        spread = maximum(1:N) do i
            vals = [block_view(bs, b)[i][k] for b in 1:4, k in 1:2]
            maximum(vals) - minimum(vals)
        end
        @test spread < 1e-3
        # explicit source with an explicit tableau
        dev, _, _ = relaxation_runs((BlockSourceTerm(ExplicitMeanRelaxation(0.1)),), (ExplicitMeanRelaxationFull(0.1),), RK2_SSP_Tableau(Float64))
        @test dev < 1e-13
        # an explicit and an implicit source together (IMEX)
        dev, _, _ = relaxation_runs((BlockSourceTerm(ExplicitMeanRelaxation(0.2)), BlockSourceTerm(MeanRelaxation(0.05))),
                                    (ExplicitMeanRelaxationFull(0.2), MeanRelaxationFull(0.05)), IMEX_ARS222_Tableau(Float64))
        @test dev < 1e-13
        # two implicit sources, applied one after another: the rates add up to the single source's (to the splitting error)
        _, bs2, pg2 = relaxation_runs((BlockSourceTerm(MeanRelaxation(0.1)), BlockSourceTerm(MeanRelaxation(0.1))), (MeanRelaxationFull(0.05),),
                                      IMEX_ARS222_Tableau(Float64))
        _, bs1, _   = relaxation_runs((BlockSourceTerm(MeanRelaxation(0.05)),), (MeanRelaxationFull(0.05),), IMEX_ARS222_Tableau(Float64))
        @test maximum(maximum(abs.(block_view(bs2, b)[i] - block_view(bs1, b)[i])) for b in 1:4, i in 1:pg2.meta.N) < 2e-3
        # the KernelAbstractions CPU backend (the parameter array is a kernel argument)
        dev_ka, _, _ = relaxation_runs((BlockSourceTerm(MeanRelaxation(0.05)),), (MeanRelaxationFull(0.05),), IMEX_ARS222_Tableau(Float64);
                                       backend = KABackend(HyperCloudCore.KA.CPU()))
        @test dev_ka < 1e-13
        # scope
        speeds = [(0.5,), (1.0,), (1.5,), (2.0,)]
        eq_b = DiscreteTransport(Val(2), speeds)
        interp = MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux())
        @test_throws ArgumentError BlockTimeStepper(eq_b, interp, (BlockSourceTerm(MeanRelaxation(0.1)),), RK2_SSP_Tableau(Float64))
        @test BlockTimeStepper(eq_b, interp, (BlockSourceTerm(ExplicitMeanRelaxation(0.1)),), RK2_SSP_Tableau(Float64)) isa BlockTimeStepper
        struct VectorParams <: AbstractImplicitBlockSource end
        HyperCloudCore.reduction_type(::VectorParams) = Vector{Float64}
        @test_throws ArgumentError BlockSourceTerm(VectorParams())
        @test_throws ErrorException HyperCloudCore.block_reduce(MeanRelaxation(1.0) isa MeanRelaxation ? VectorParams() : VectorParams(), 1, nothing, 1, 1, nothing, 0.0, 0.0)
    end
    @testset "MOOD on blocks: the criterion per block, one order per particle, the cross-block admissibility" begin
        speeds = [(-1.5,), (-0.5,), (0.25,), (0.75,), (1.0,), (1.25,), (1.75,), (2.0,)]
        eq_f = DiscreteTransport(Val(8), speeds)
        eq_b = DiscreteTransport(Val(2), speeds)
        # without MOOD the unlimited third-order scheme overshoots the box (top 2.8)
        dev0, _, _, bs0, _, _ = mood_runs(eq_f, eq_b, MOOD(NoMOOD()), RK3_SSP_Tableau(Float64))
        @test dev0 < 1e-13
        @test block_max(bs0) > 2.8 + 0.02
        # MOODu1 with an SSP tableau (the Euler substeps are checked): the DMP holds, the same particles drop as in the standard run
        dev, ts_f, ts_b, bs, pg_f, pg_b = mood_runs(eq_f, eq_b, MOOD(EPD1(), MOODu1(1e-3)), RK3_SSP_Tableau(Float64))
        @test dev < 1e-12
        @test block_max(bs) < 2.8 + 0.015                            # (the standard run gives the same 2.8097; without MOOD > 2.82)
        @test any(mood_flags(ts_b)) && mood_flags(ts_b) == mood_flags(ts_f)
        @test pg_b.core.particle_orders[1:pg_b.meta.N] == pg_f.core.particle_orders[1:pg_f.meta.N]
        @test mood_inadmissible(ts_b) == 0
        # the forcing halo (StrictEPD0) and the stage candidates of a non-SSP tableau with an implicit source
        dev, ts_f, ts_b, bs, pg_f, pg_b = mood_runs(eq_f, eq_b, MOOD(StrictEPD0(), MOODu1(1e-3)), RK3_SSP_Tableau(Float64))
        @test dev < 1e-12 && mood_flags(ts_b) == mood_flags(ts_f)
        dev, ts_f, ts_b, bs, pg_f, pg_b = mood_runs(eq_f, eq_b, MOOD(EPD1(), MOODu1(1e-3)), IMEX_ARS222_Tableau(Float64);
                                                     sources_b = (BlockSourceTerm(MeanRelaxation(0.05)),), sources_f = (MeanRelaxationFull(0.05),))
        @test dev < 1e-12 && mood_flags(ts_b) == mood_flags(ts_f)
        # OnlyMOOD: every interior particle drops to order 1 at the first checkpoint, identical first-order results
        dev, ts_f, ts_b, bs, pg_f, pg_b = mood_runs(eq_f, eq_b, MOOD(EPD1(), OnlyMOOD()), RK2_SSP_Tableau(Float64); tend = 0.05)
        @test dev < 1e-12
        @test all(i -> pg_b.core.is_boundary[i] || pg_b.core.particle_orders[i] == 1, 1:pg_b.meta.N)
        # the cross-block admissibility: particles outside the box (mean 1 < 1.5) are inadmissible at every checkpoint
        g_f, g_b = GuardedTransport(eq_f), GuardedTransport(eq_b)
        dev, ts_f, ts_b, bs, pg_f, pg_b = mood_runs(g_f, g_b, MOOD(EPD1(), MOODPAD()), RK2_SSP_Tableau(Float64); tend = 0.05)
        @test dev < 1e-12
        @test mood_flags(ts_b) == mood_flags(ts_f) && any(mood_flags(ts_b))
        @test mood_inadmissible(ts_b) == mood_inadmissible(ts_f) > 0
        outside = [i for i in 1:pg_b.meta.N if abs(pg_b.core.positions[i][1]) > 0.35]
        @test all(i -> pg_b.core.particle_orders[i] == 1, outside)
        # scope
        interp = MUSCL(Float64, 1, 2, 2, NoLimiter(), RusanovFlux())
        @test_throws ArgumentError BlockTimeStepper(eq_b, interp, MOOD(EPD1(), MOODu2(1e-3)), (), RK2_SSP_Tableau(Float64))
    end
end
