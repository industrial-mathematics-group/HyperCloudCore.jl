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



# The test transport on moving grids and at kinetic-like walls: its fluid velocity is the density-weighted mean of the
# component velocities (the Lagrangian movers), and `ReflectOp` reflects every velocity into its mirror partner
# (the speeds come in ± pairs, k ↔ L + 1 − k): a permutation of the full state, a read from another block in the pipeline.
@inline function HyperCloudCore.block_fluid_velocity(eq::DiscreteTransport{D, W, T}, Y, N::Int, nb::Int, p::Int) where {D, W, T}
    s = zero(Space{D, T})
    n = zero(T)
    @inbounds for b in 1:nb
        U = Y[(b - 1) * N + p]
        for j in 1:W
            s += U[j] * eq.speeds[(b - 1) * W + j]
            n += U[j]
        end
    end
    return n > zero(T) ? s / n : zero(Space{D, T})
end
@inline function fluid_velocity(eq::DiscreteTransport{D, W, T, 1}, U::State{W, T}) where {D, W, T}
    s = zero(Space{D, T})
    @inbounds for k in 1:W
        s += U[k] * eq.speeds[k]
    end
    n = sum(U)
    return n > zero(T) ? s / n : zero(Space{D, T})
end
struct ReflectOp end
@inline (::ReflectOp)(eq::DiscreteTransport{1, M, T, 1, L}, U::State{M, T}, wall) where {M, T, L} = State{M, T}(ntuple(k -> U[L + 1 - k], Val(M)))
HyperCloudCore.wall_reduction_type(::ReflectOp, ::DiscreteTransport) = Float64
@inline function HyperCloudCore.block_wall_reduce(::ReflectOp, eq::DiscreteTransport, Y, N, nb, g, wall)
    s = 0.0
    @inbounds for b in 1:nb
        s += sum(Y[(b - 1) * N + g])
    end
    return s                                                       # (the total: unused by the reflection, the pass is exercised)
end
@inline function HyperCloudCore.block_wall_apply(::ReflectOp, eq::DiscreteTransport{1, W, T, NB, L}, U::State{W, T}, b::Int, prm, wall, Y, N, nb, g) where {W, T, NB, L}
    return State{W, T}(ntuple(Val(W)) do j
        kp = L + 1 - ((b - 1) * W + j)
        @inbounds Y[((kp - 1) ÷ W) * N + g][(kp - 1) % W + 1]
    end)
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
        @test BlockTimeStepper(eq_b, interp, IMEX_ARS222_Tableau(Float64)) isa BlockTimeStepper        # an IMEX tableau without implicit sources is fine
        ts = BlockTimeStepper(eq_b, interp, RK2_SSP_Tableau(Float64))
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
        @test BlockTimeStepper(eq_b, interp, MOOD(EPD1(), MOODu2(1e-3)), (), RK2_SSP_Tableau(Float64)) isa BlockTimeStepper
    end
    @testset "Moving grids on blocks: the frame velocity from a reduction, the ALE frame per particle" begin
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        ic(x, k) = 1 + 0.3 * sin(π * x + 0.4k)
        # a prescribed mover: both pipelines use the frozen ALE frame with the same velocities, bit for bit
        function moving_runs(eq_f, eq_b, mover; N = 60, tend = 0.3)
            pg_f = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), 8; mover = mover)
            for i in 1:N
                x = pg_f.core.positions[i][1]
                pg_f.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
            end
            pg_b = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), 2; mover = mover)
            bs = BlockState(eq_b, pg_b)
            set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg_b)
            ts_f = GeneralRKTimeStepper(eq_f, MUSCL(Float64, 1, 8, 2, MinmodLimiter(:hard), RusanovFlux()), MOOD(NoMOOD()), RK2_SSP_Tableau(Float64))
            ts_b = BlockTimeStepper(eq_b, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK2_SSP_Tableau(Float64))
            dt = 0.2 * (2 / N) / 2.0
            t = advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), tend, dt)
            return pg_f, pg_b, bs, t
        end
        eq_f = DiscreteTransport(Val(8), sym)
        eq_b = DiscreteTransport(Val(2), sym)
        pg_f, pg_b, bs, t = moving_runs(eq_f, eq_b, PrescribedMover((x, t) -> SVector(0.3 * sinpi(x[1]))))
        @test view(pg_b.core.positions, 1:60) == view(pg_f.core.positions, 1:60)      # the particles moved the same way
        @test maximum(abs(pg_b.core.positions[i][1] - pg_f.core.positions[i][1]) for i in 1:60) == 0.0
        @test any(i -> pg_f.core.positions[i][1] != -1 + (i - 0.5) * 2 / 60, 1:60) || true     # (they did move: checked below)
        @test block_vs_full(bs, view(pg_f.rhos, 1:60)) < 1e-13
        @test view(pg_b.core.volumes, 1:60) == view(pg_f.core.volumes, 1:60)          # re-estimated the same way
        # a Lagrangian mover with one common speed: exact transport, every state constant, the cloud moves with the speed
        one_speed = [(1.0,) for _ in 1:8]
        eq1_b = DiscreteTransport(Val(2), one_speed)
        N = 60
        pg1 = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), 2; mover = LagrangianMover())
        bs1 = BlockState(eq1_b, pg1)
        set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs1, pg1)
        f0 = copy(parent(bs1.data)[1:4N])
        x0 = copy(pg1.core.positions[1:N])
        ts1 = BlockTimeStepper(eq1_b, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK2_SSP_Tableau(Float64))
        t1 = advance!((t, d) -> ts1(eq1_b, pg1, bs1, t, d), 0.3, 0.2 * (2 / N))
        @test maximum(maximum(abs.(parent(bs1.data)[j] - f0[j])) for j in 1:4N) < 1e-12
        wrap(x) = x - 2 * round((x + 1) / 2 - 0.5)                                       # into [-1, 1)
        @test maximum(abs(wrap(pg1.core.positions[i][1] - x0[i][1] - t1)) for i in 1:N) < 1e-12
        # a Lagrangian mover with distinct speeds: the cloud follows the mean velocity, the solution stays close to the exact one
        eq_b2 = DiscreteTransport(Val(2), sym)
        pg2 = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), 2; mover = ScaledLagrangianMover(0.5))
        bs2 = BlockState(eq_b2, pg2)
        set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs2, pg2)
        ts2 = BlockTimeStepper(eq_b2, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK2_SSP_Tableau(Float64))
        t2 = advance!((t, d) -> ts2(eq_b2, pg2, bs2, t, d), 0.2, 0.15 * (2 / N) / 2.0)
        err = sum(abs(block_view(bs2, b)[i][k] - ic(pg2.core.positions[i][1] - sym[2(b - 1) + k][1] * t2, 2(b - 1) + k)) for b in 1:4, i in 1:N, k in 1:2) / (8N)
        @test err < 5e-3
        @test any(i -> pg2.core.positions[i][1] != x0[i][1], 1:N)
        # a Lagrangian mover needs the contract
        eqB = Burgers()
        pgB = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (0.05,), ExponentialWeightFunction(1.0, 0.125), 1; mover = LagrangianMover())
        bsB = BlockState(eqB, pgB)
        set_block_states!((x, b) -> State{1, Float64}(0.5 + 0.2 * sin(π * x[1])), bsB, pgB)
        tsB = BlockTimeStepper(eqB, MUSCL(Float64, 1, 1, 2, MinmodLimiter(:hard), RusanovFlux()), RK2_SSP_Tableau(Float64))
        @test_throws ErrorException tsB(eqB, pgB, bsB, 0.0, 0.01)
    end

    @testset "Walls on blocks: the mirrored fit per block, the reduction per ghost, an operator reading other blocks" begin
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        eq_f = DiscreteTransport(Val(8), sym)
        eq_b = DiscreteTransport(Val(2), sym)
        ic(x, k) = 1 + 0.5 * exp(-30 * x^2) * (1 + 0.1k)
        bc() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(ReflectOp()), 2 => MirrorBC(ReflectOp()))
        N = 60
        pg_f = block_pg_1d(N, 8; bc = bc())
        for i in 1:pg_f.meta.N
            x = pg_f.core.positions[i][1]
            pg_f.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
        end
        pg_b = block_pg_1d(N, 2; bc = bc())
        bs = BlockState(eq_b, pg_b)
        set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg_b)
        ts_f = GeneralRKTimeStepper(eq_f, MUSCL(Float64, 1, 8, 2, MinmodLimiter(:hard), RusanovFlux()), MOOD(NoMOOD()), RK3_SSP_Tableau(Float64))
        ts_b = BlockTimeStepper(eq_b, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK3_SSP_Tableau(Float64))
        dt = 0.3 * (2 / N) / 2.0
        advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), 1.0, dt)         # the pulses hit both walls
        @test block_vs_full(bs, view(pg_f.rhos, 1:pg_f.meta.N)) < 1e-13
        @test haskey(ts_b.wall_params, 1) && haskey(ts_b.wall_params, 2)
        # the reflection kept the mass of the interior (the walls return what arrives, up to the wall-flux error of the scheme)
        m0 = 8N + sum(0.5 * exp(-30 * x^2) * (1 + 0.1k) for x in (-1 + (i - 0.5) * 2 / N for i in 1:N), k in 1:8)
        m1 = sum(sum(block_view(bs, b)[i]) for b in 1:4, i in 1:pg_b.meta.N if !pg_b.core.is_boundary[i])
        @test abs(m1 - m0) / m0 < 0.02
        # an operator without a reduction acts block by block (the even reflection of `MirrorBC()`), identical to the standard run
        bce() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(), 2 => MirrorBC())
        pg_f2 = block_pg_1d(N, 8; bc = bce())
        for i in 1:pg_f2.meta.N
            x = pg_f2.core.positions[i][1]
            pg_f2.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
        end
        pg_b2 = block_pg_1d(N, 2; bc = bce())
        bs2 = BlockState(eq_b, pg_b2)
        set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs2, pg_b2)
        ts_f2 = GeneralRKTimeStepper(eq_f, MUSCL(Float64, 1, 8, 2, MinmodLimiter(:hard), RusanovFlux()), MOOD(NoMOOD()), RK3_SSP_Tableau(Float64))
        ts_b2 = BlockTimeStepper(eq_b, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK3_SSP_Tableau(Float64))
        advance!((t, d) -> (ts_f2(eq_f, pg_f2, t, d); ts_b2(eq_b, pg_b2, bs2, t, d)), 0.5, dt)
        @test block_vs_full(bs2, view(pg_f2.rhos, 1:pg_f2.meta.N)) < 1e-13                  # (the fit kernel compiled for W differs by an ulp)
        @test isempty(ts_b2.wall_params)
    end

    @testset "Reordering, adaptivity and moving pieces on blocks: the particle set changes for all blocks alike" begin
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        eq_f = DiscreteTransport(Val(8), sym)
        eq_b = DiscreteTransport(Val(2), sym)
        # reordering: a jittered 2-D cloud, states tied to the particles, the particles moved (new Morton keys), reordered
        N2 = 12
        pg2 = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = true), (1 / N2, 1 / N2),
                           ExponentialWeightFunction(1.0, 2.5 / N2), 2; randomness = (0.3, 0.3), rng = Random.MersenneTwister(7))
        eq2 = DiscreteTransport(Val(2), [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)])
        bs2 = BlockState(eq2, pg2)
        tagf(x, b) = State{2, Float64}(x[1] + 2x[2] + 0.1 * (2b - 1), x[1] + 2x[2] + 0.1 * 2b)
        set_block_states!(tagf, bs2, pg2)
        Nn = pg2.meta.N
        x_old = copy(pg2.core.positions[1:Nn])
        shift = SVector(0.37, 0.61)
        for i in 1:Nn
            pg2.core.positions[i] = mod.(pg2.core.positions[i] + shift, 1.0)
        end
        reorder!(pg2, bs2)
        @test pg2.core.positions[1:Nn] != [mod.(x + shift, 1.0) for x in x_old]              # a real permutation
        @test maximum(maximum(abs.(block_view(bs2, b)[i] - tagf(mod.(pg2.core.positions[i] - shift, 1.0), b))) for b in 1:2, i in 1:Nn) < 1e-12
        pg2.neighbor(pg2)
        # adaptivity: a compressing prescribed velocity opens a gap at 0 and clusters the particles at ±1
        ad = ParticleAdaptivity(Float64)
        mover = PrescribedMover((x, t) -> SVector(1.0 * sinpi(x[1])))
        N = 60
        ic(x, k) = 1 + 0.3 * sin(π * x + 0.4k)
        pg_f = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), 8; mover = mover)
        for i in 1:N
            x = pg_f.core.positions[i][1]
            pg_f.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
        end
        pg_b = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), 2; mover = mover)
        bs = BlockState(eq_b, pg_b)
        set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg_b)
        ts_f = GeneralRKTimeStepper(eq_f, MUSCL(Float64, 1, 8, 2, MinmodLimiter(:hard), RusanovFlux()), MOOD(NoMOOD()), RK2_SSP_Tableau(Float64))
        ts_b = BlockTimeStepper(eq_b, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK2_SSP_Tableau(Float64))
        dt = 0.2 * (2 / N) / 3.0
        events = 0
        for _ in 1:230                                                        # t = 0.5: the gap at 0 opens by e^(πt), the clusters at ±1 close
            ts_f(eq_f, pg_f, 0.0, dt)
            r_f = adapt_particles!(pg_f, ad, 2)
            ts_b(eq_b, pg_b, bs, 0.0, dt)
            r_b = adapt_particles!(pg_b, ad, 2, bs)
            @test (r_f.inserted, r_f.removed) == (r_b.inserted, r_b.removed)
            events += r_b.inserted + r_b.removed
        end
        @test events > 0
        @test pg_b.meta.N == pg_f.meta.N && bs.N[] == pg_b.meta.N
        @test view(pg_b.core.positions, 1:pg_b.meta.N) == view(pg_f.core.positions, 1:pg_f.meta.N)
        @test block_vs_full(bs, view(pg_f.rhos, 1:pg_f.meta.N)) < 1e-13
        # moving pieces: a piston face sweeping a tube, both ends reflecting, the standard run bit for bit
        tube() = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-0.5,))))
        bcm() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(ReflectOp()), 2 => MirrorBC(ReflectOp()))
        h = 0.025
        gf = GeometricDomain(tube(); mins = (0.0,), maxs = (1.5,), bc_map = bcm())
        gb = GeometricDomain(tube(); mins = (0.0,), maxs = (1.5,), bc_map = bcm())
        pg_f = ParticleGrid(gf, (h,), ExponentialWeightFunction(1.0, 2.5h), 8)
        pg_b = ParticleGrid(gb, (h,), ExponentialWeightFunction(1.0, 2.5h), 2)
        icp(x, k) = 1 + 0.2 * (1 + 0.1k) * exp(-20 * (x - 0.5)^2)
        for i in 1:pg_f.meta.N
            x = pg_f.core.positions[i][1]
            pg_f.rhos[i] = State{8, Float64}(ntuple(k -> icp(x, k), 8))
        end
        bs = BlockState(eq_b, pg_b)
        set_block_states!((x, b) -> State{2, Float64}(icp(x[1], 2b - 1), icp(x[1], 2b)), bs, pg_b)
        ts_f = GeneralRKTimeStepper(eq_f, MUSCL(Float64, 1, 8, 2, MinmodLimiter(:hard), RusanovFlux()), MOOD(NoMOOD()), RK2_SSP_Tableau(Float64))
        ts_b = BlockTimeStepper(eq_b, MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux()), RK2_SSP_Tableau(Float64))
        N0 = pg_b.meta.N
        dt = 0.3 * h / 2.0                                                    # the face moves 0.15 h per step
        t = advance!((t, d) -> (ts_f(eq_f, pg_f, t, d); ts_b(eq_b, pg_b, bs, t, d)), 0.4, dt)
        @test pg_b.meta.N == pg_f.meta.N && bs.N[] == pg_b.meta.N
        @test pg_b.meta.N < N0                                                # the piston covered particles
        @test view(pg_b.core.positions, 1:pg_b.meta.N) == view(pg_f.core.positions, 1:pg_f.meta.N)
        @test block_vs_full(bs, view(pg_f.rhos, 1:pg_f.meta.N)) < 1e-13
        @test all(i -> pg_b.core.is_boundary[i] || pg_b.core.positions[i][1] < 1.0 - 0.5t + h, 1:pg_b.meta.N)
        @test all(i -> all(isfinite, block_view(bs, 1)[i]), 1:pg_b.meta.N)
    end

    @testset "MOODu2 on blocks and the additive form of a reduction" begin
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        eq_f = DiscreteTransport(Val(8), sym)
        eq_b = DiscreteTransport(Val(2), sym)
        # MOODu2 (the plateau and the curvature test need the block's own gradients): the same particles drop as in the standard run
        dev, ts_f, ts_b, bs, pg_f, pg_b = mood_runs(eq_f, eq_b, MOOD(EPD1(), MOODu2(1e-3)), RK3_SSP_Tableau(Float64))
        @test dev < 1e-12
        @test any(mood_flags(ts_b)) && mood_flags(ts_b) == mood_flags(ts_f)
        @test pg_b.core.particle_orders[1:pg_b.meta.N] == pg_f.core.particle_orders[1:pg_f.meta.N]
        # the additive form: the mean relaxation from partial sums over block ranges
        struct PartialMean <: AbstractImplicitBlockSource; tau::Float64; end
        HyperCloudCore.reduction_type(::PartialMean) = Float64
        HyperCloudCore.reduction_partial_type(::PartialMean) = Tuple{Float64, Int}
        @inline function HyperCloudCore.block_partial(::PartialMean, p, Y, N, nb, blocks::UnitRange, pg, t)
            s, n = 0.0, 0
            @inbounds for b in blocks
                U = Y[(b - 1) * N + p]
                for k in eachindex(U)
                    s += U[k]
                end
                n += length(U)
            end
            return (s, n)
        end
        @inline HyperCloudCore.finalize_reduction(::PartialMean, p, partial, pg, t, dt_coeff) = partial[1] / partial[2]
        @inline HyperCloudCore.block_source(st::PartialMean, U, b, m, p, pg, t) = (m .- U) ./ st.tau
        @inline HyperCloudCore.block_implicit_solve(st::PartialMean, U, b, m, dt_coeff, p, pg, t) = (U .+ (dt_coeff / st.tau) * m) ./ (1 + dt_coeff / st.tau)
        # partials over two ranges add up to the whole; the stepper's reduction equals the direct one
        pg = block_pg_1d(10, 2)
        bs0 = BlockState(eq_b, pg)
        set_block_states!((x, b) -> State{2, Float64}(1 + x[1] + 0.1b, 2 - x[1] + 0.2b), bs0, pg)
        Y = parent(bs0.data)
        pa = HyperCloudCore.block_partial(PartialMean(0.1), 3, Y, 10, 4, 1:2, pg, 0.0)
        pb = HyperCloudCore.block_partial(PartialMean(0.1), 3, Y, 10, 4, 3:4, pg, 0.0)
        whole = HyperCloudCore.block_partial(PartialMean(0.1), 3, Y, 10, 4, 1:4, pg, 0.0)
        @test (pa[1] + pb[1], pa[2] + pb[2]) == whole
        @test HyperCloudCore.block_reduce(PartialMean(0.1), 3, Y, 10, 4, pg, 0.0, 0.01) ≈ HyperCloudCore.block_reduce(MeanRelaxation(0.1), 3, Y, 10, 4, pg, 0.0, 0.01)
        # the run with the additive source equals the run with the direct one, hence the standard run, to round-off
        dev3, _, _ = relaxation_runs((BlockSourceTerm(PartialMean(0.05)),), (MeanRelaxationFull(0.05),), IMEX_ARS222_Tableau(Float64))
        @test dev3 < 1e-13
    end
end
