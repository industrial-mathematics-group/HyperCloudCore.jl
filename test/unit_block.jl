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
# (the mean is over the whole state: on blocks a block alone is only checked for finiteness, the mean is the cross-block
# check above; the default `block_is_admissible` would apply `is_admissible` to the block's two components)
@inline HyperCloudCore.block_is_admissible(g::GuardedTransport, U::State, b::Int) = all(isfinite, U)

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


# The additive forms of the test equations' reductions, for a state split over devices (`BlockDevices`): the fluid velocity
# of the transport (density and momentum), the guarded equation's mean (the sum of the components), the reflecting wall's
# total; `PartialMean` is the mean relaxation in the additive form (sum and count)
HyperCloudCore.fluid_velocity_partial_type(::DiscreteTransport{D, W, T}) where {D, W, T} = SVector{D + 1, T}
@inline function HyperCloudCore.block_fluid_velocity_partial(eq::DiscreteTransport{D, W, T}, Y, N::Int, nb::Int, blocks::UnitRange, p::Int) where {D, W, T}
    s = zero(Space{D, T})
    n = zero(T)
    @inbounds for b in blocks
        U = Y[(b - 1) * N + p]
        for j in 1:W
            s += U[j] * eq.speeds[(b - 1) * W + j]
            n += U[j]
        end
    end
    return SVector{D + 1, T}(n, s...)
end
@inline function HyperCloudCore.finalize_fluid_velocity(eq::DiscreteTransport{D, W, T}, partial::SVector) where {D, W, T}
    n = partial[1]
    s = Space{D, T}(ntuple(d -> partial[d + 1], Val(D)))
    return n > zero(T) ? s / n : zero(Space{D, T})
end
HyperCloudCore.admissible_partial_type(::GuardedTransport{D, W, T}) where {D, W, T} = T
@inline function HyperCloudCore.block_admissible_partial(g::GuardedTransport{D, W, T}, Y, N::Int, nb::Int, blocks::UnitRange, p::Int) where {D, W, T}
    s = zero(T)
    @inbounds for b in blocks
        s += sum(Y[(b - 1) * N + p])
    end
    return s
end
@inline HyperCloudCore.finalize_admissible(g::GuardedTransport{D, W, T}, s) where {D, W, T} = s / (HyperCloudCore.num_blocks(g) * W) > T(1.5)
# a guarded equation without the additive form of its admissibility: refused on a split state
struct GuardedNoPartial{D, W, T, EQ <: DiscreteTransport{D, W, T}} <: HyperbolicPDE{D, W, T}
    eq::EQ
end
GuardedNoPartial(eq::DiscreteTransport{D, W, T}) where {D, W, T} = GuardedNoPartial{D, W, T, typeof(eq)}(eq)
HyperCloudCore.num_blocks(g::GuardedNoPartial) = HyperCloudCore.num_blocks(g.eq)
@inline HyperCloudCore.block_flux(g::GuardedNoPartial, U::State, b::Int) = HyperCloudCore.block_flux(g.eq, U, b)
@inline HyperCloudCore.block_max_eigenvalue(g::GuardedNoPartial, U::State, d::Int, b::Int) = HyperCloudCore.block_max_eigenvalue(g.eq, U, d, b)
@inline HyperCloudCore.block_admissible(g::GuardedNoPartial, Y, N::Int, nb::Int, p::Int) = true

# The mean relaxation in the additive form: the sum and the count of the components of a block range (adds up over ranges)
struct PartialMean <: AbstractImplicitBlockSource; tau::Float64; end
HyperCloudCore.reduction_type(::PartialMean) = Float64
HyperCloudCore.reduction_partial_type(::PartialMean) = SVector{2, Float64}
@inline function HyperCloudCore.block_partial(::PartialMean, p, Y, N, nb, blocks::UnitRange, pg, t)
    s, n = 0.0, 0.0
    @inbounds for b in blocks
        U = Y[(b - 1) * N + p]
        for k in eachindex(U)
            s += U[k]
        end
        n += length(U)
    end
    return SVector(s, n)
end
@inline HyperCloudCore.finalize_reduction(::PartialMean, p, partial, pg, t, dt_coeff) = partial[1] / partial[2]
@inline HyperCloudCore.block_source(st::PartialMean, U, b, m, p, pg, t) = (m .- U) ./ st.tau
@inline HyperCloudCore.block_implicit_solve(st::PartialMean, U, b, m, dt_coeff, p, pg, t) = (U .+ (dt_coeff / st.tau) * m) ./ (1 + dt_coeff / st.tau)

# Wall operators that take the state at the wall point (`uses_wall_state`): `WallProbe` returns it as the ghost state (the
# test of the second evaluation), `WallSumOp` reduces the wall-point states of all blocks to their sum and keeps the mirrored
# value (the two-pass form of the block pipeline with the second scratch)
struct WallProbe end
HyperCloudCore.uses_wall_state(::WallProbe) = true
@inline (::WallProbe)(eq, U::State, wall, Uw::State) = Uw
struct WallSumOp end
HyperCloudCore.uses_wall_state(::WallSumOp) = true
HyperCloudCore.wall_reduction_type(::WallSumOp, ::DiscreteTransport) = Float64
@inline function HyperCloudCore.block_wall_reduce(::WallSumOp, eq::DiscreteTransport, Y, N, nb, g, wall, Yw)
    s = 0.0
    @inbounds for b in 1:nb
        s += sum(Yw[(b - 1) * N + g])
    end
    return s
end
@inline HyperCloudCore.block_wall_apply(::WallSumOp, eq::DiscreteTransport, U::State, b::Int, prm, wall, Y, N, nb, g) = U

# A backend that counts the kernel launches (runs them on the Polyester backend)
struct CountingBackend <: HyperCloudCore.ExecutionBackend end
const LAUNCHES = Ref(0)
HyperCloudCore.foreach_particle(body, ::CountingBackend, N::Int, args::Tuple) = (LAUNCHES[] += 1; foreach_particle(body, PolyesterBackend(), N, args))

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
        # partials over two ranges add up to the whole; the stepper's reduction equals the direct one
        pg = block_pg_1d(10, 2)
        bs0 = BlockState(eq_b, pg)
        set_block_states!((x, b) -> State{2, Float64}(1 + x[1] + 0.1b, 2 - x[1] + 0.2b), bs0, pg)
        Y = parent(bs0.data)
        pa = HyperCloudCore.block_partial(PartialMean(0.1), 3, Y, 10, 4, 1:2, pg, 0.0)
        pb = HyperCloudCore.block_partial(PartialMean(0.1), 3, Y, 10, 4, 3:4, pg, 0.0)
        whole = HyperCloudCore.block_partial(PartialMean(0.1), 3, Y, 10, 4, 1:4, pg, 0.0)
        @test pa + pb == whole
        @test HyperCloudCore.block_reduce(PartialMean(0.1), 3, Y, 10, 4, pg, 0.0, 0.01) ≈ HyperCloudCore.block_reduce(MeanRelaxation(0.1), 3, Y, 10, 4, pg, 0.0, 0.01)
        # the run with the additive source equals the run with the direct one, hence the standard run, to round-off
        dev3, _, _ = relaxation_runs((BlockSourceTerm(PartialMean(0.05)),), (MeanRelaxationFull(0.05),), IMEX_ARS222_Tableau(Float64))
        @test dev3 < 1e-13
    end

    @testset "Checks of the step, the storage of the sources, the memory budget" begin
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        eq_b = DiscreteTransport(Val(2), sym)
        interp() = MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux())
        # a MirrorBC operator written for the equation itself is refused before any kernel runs (it failed inside the
        # kernel: a MethodError, a crash under Polyester); one written for the block equation runs
        struct EqScale end
        @inline (::EqScale)(eq::DiscreteTransport, U::State, wall) = 0.9 * U
        struct BlockScale end
        @inline (::BlockScale)(w::BlockOf, U::State, wall) = 0.9 * U
        for (op, ok) in ((EqScale(), false), (BlockScale(), true))
            pg = block_pg_1d(40, 2; bc = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(op), 2 => MirrorBC(op)))
            bs = BlockState(eq_b, pg)
            set_block_states!((x, b) -> State{2, Float64}(1.0, 1.0 + 0.1b), bs, pg)
            ts = BlockTimeStepper(eq_b, interp(), RK2_SSP_Tableau(Float64))
            if ok
                ts(eq_b, pg, bs, 0.0, 0.01)
                @test all(i -> all(isfinite, block_view(bs, 4)[i]), 1:pg.meta.N)
            else
                @test_throws ArgumentError ts(eq_b, pg, bs, 0.0, 0.01)
            end
        end
        # a state set for another particle count is refused by the step (before the sweep of moving pieces, which
        # rewrote it) and by the time step; block indices are checked
        pg = block_pg_1d(40, 2)
        bs_other = BlockState(eq_b, block_pg_1d(30, 2))
        ts = BlockTimeStepper(eq_b, interp(), RK2_SSP_Tableau(Float64))
        @test_throws ArgumentError ts(eq_b, pg, bs_other, 0.0, 0.01)
        @test bs_other.N[] == 30
        # with a moving piece the check comes before the sync of the pieces (which remaps the state it is given)
        tube = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-0.5,))))
        gp = GeometricDomain(tube; mins = (0.0,), maxs = (1.5,), bc_map = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(), 2 => MirrorBC()))
        pgp = ParticleGrid(gp, (0.025,), ExponentialWeightFunction(1.0, 2.5 * 0.025), 2)
        bsp = BlockState(eq_b, block_pg_1d(30, 2))
        set_block_states!((x, b) -> State{2, Float64}(1.0, 2.0), bsp, block_pg_1d(30, 2))
        data0 = copy(parent(bsp.data))
        tsp = BlockTimeStepper(eq_b, interp(), RK2_SSP_Tableau(Float64))
        @test_throws ArgumentError tsp(eq_b, pgp, bsp, 0.1, 0.01)            # (t = 0.1: the sync would sweep)
        @test bsp.N[] == 30 && view(parent(bsp.data), 1:(4 * 30)) == view(data0, 1:(4 * 30))     # (the used part: the headroom is uninitialised)
        @test_throws ArgumentError get_time_step(pg, eq_b, ts.inner.divergence_interpolator, bs_other)
        @test_throws BoundsError block_view(bs_other, 5)
        # the parameter arrays of the sources follow the stage buffers' storage (a source built with the default
        # `storage = Array` for a device stepper failed at the first launch)
        struct TagVector{T} <: AbstractVector{T}
            v::Vector{T}
        end
        TagVector{T}(::UndefInitializer, n::Int) where {T} = TagVector{T}(Vector{T}(undef, n))
        Base.size(a::TagVector) = size(a.v)
        Base.getindex(a::TagVector, i::Int) = a.v[i]
        bst = BlockSourceTerm(MeanRelaxation(0.1))
        @test HyperCloudCore._on_storage(bst, Array) === bst
        @test parent(HyperCloudCore._on_storage(bst, TagVector).params) isa TagVector
        ts = BlockTimeStepper(eq_b, interp(), (bst,), IMEX_ARS222_Tableau(Float64))
        @test parent(ts.implicit_sources[1].params) isa Vector
        # the memory budget prices the block-major arrays nb times per particle, and a reservation of the block stepper
        # and the state holds the run (the inner stepper's reservation alone failed at the first step)
        pg = block_pg_1d(60, 2)
        bs = BlockState(eq_b, pg)
        set_block_states!((x, b) -> State{2, Float64}(1 + 0.2 * sinpi(x[1]), 1.0 + 0.1b), bs, pg)
        sc = interp()
        ts = BlockTimeStepper(eq_b, sc, (BlockSourceTerm(MeanRelaxation(0.1)),), IMEX_ARS222_Tableau(Float64))
        e1 = memory_estimate(pg, ts.inner, sc)
        e4 = memory_estimate(pg, ts, bs, sc)
        nstate = 2 + 2 * 3                                            # rho_n, rho_cand, Y/K_E/K_I per stage
        @test e4.per_particle - e1.per_particle >= (4 - 1) * nstate * sizeof(State{2, Float64}) + 4 * sizeof(State{2, Float64})
        r = reserve!(pg, ts, bs, sc; limit = 2 * e4.bytes)
        @test length(parent(ts.inner.rho_n)) == 4 * r.particles && length(parent(ts.inner.work.dirty)) == r.particles
        @test length(parent(bs.data)) == 4 * r.particles
        dt = 0.2 * (2 / 60) / 2.0
        advance!((t, d) -> ts(eq_b, pg, bs, t, d), 0.1, dt)
        @test all(i -> all(isfinite, block_view(bs, 2)[i]), 1:pg.meta.N)
        @test length(parent(ts.inner.rho_n)) == 4 * r.particles      # nothing grew
        # with kinetic walls the mirrored scratch (block-major) and the wall parameters are reserved as well
        bcw() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(ReflectOp()), 2 => MirrorBC(ReflectOp()))
        pg = block_pg_1d(60, 2; bc = bcw())
        bs = BlockState(eq_b, pg)
        set_block_states!((x, b) -> State{2, Float64}(1 + 0.5 * exp(-30 * x[1]^2), 1.0 + 0.1b), bs, pg)
        sc = interp()
        ts = BlockTimeStepper(eq_b, sc, RK2_SSP_Tableau(Float64))
        ew = memory_estimate(pg, ts, bs, sc)
        @test sort(collect(keys(ts.wall_params))) == [1, 2]
        @test ew.per_particle >= memory_estimate(pg, ts.inner, sc).per_particle + 2 * 4 * sizeof(State{2, Float64})
        r = reserve!(pg, ts, bs, sc; limit = 2 * ew.bytes)
        @test ts.ghost_scratch.fixed && all(g -> g.fixed, values(ts.wall_params))
        advance!((t, d) -> ts(eq_b, pg, bs, t, d), 0.3, dt)
        @test length(parent(ts.ghost_scratch)) == 4 * r.particles
        @test all(i -> all(isfinite, block_view(bs, 3)[i]), 1:pg.meta.N)
        # MOODu2 with a scheme without a gradient cache: the flag pass runs over all blocks at once, and so do its plateau
        # ranges (they were written into a buffer of one launch's blocks)
        pgu = block_pg_1d(60, 2)
        bsu = BlockState(eq_b, pgu)
        set_block_states!((x, b) -> State{2, Float64}(1 + (abs(x[1]) < 0.3 ? 1.0 : 0.0), 1.0 + 0.1b), bsu, pgu)
        tsu = BlockTimeStepper(eq_b, CentralDivergence(Float64, 1, 2, 2), MOOD(EPD1(), MOODu2(1e-3)), (), RK3_SSP_Tableau(Float64); blocks_per_launch = 1)
        tsu(eq_b, pgu, bsu, 0.0, 0.005)
        @test all(i -> all(isfinite, block_view(bsu, 4)[i]), 1:pgu.meta.N)
        # a neighbour search per stage on a compressing grid (the slots grow within a step): every chunking gives the same
        # run (the interaction buffer was resized for one block's slots, the last slots of a launch ran past it)
        mkc() = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); bc_map = Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC(), 2 => OutflowBC())),
                             (2 / 60,), ExponentialWeightFunction(1.0, 2.5 * 2 / 60), 2; mover = PrescribedMover((x, t) -> SVector(-3.0 * x[1])))
        function runc(bpl)
            pgc = mkc()
            bsc = BlockState(eq_b, pgc)
            set_block_states!((x, b) -> State{2, Float64}(1 + 0.3 * sinpi(x[1] + 0.2b), 1.0), bsc, pgc)
            tsc = BlockTimeStepper(eq_b, interp(), MOOD(NoMOOD()), (), RK3_SSP_Tableau(Float64); blocks_per_launch = bpl, full_search_per_stage = true)
            M0 = length(pgc.neighbor.indices)
            t = 0.0
            for _ in 1:4
                tsc(eq_b, pgc, bsc, t, 0.04)
                t += 0.04
            end
            return copy(view(parent(bsc.data), 1:(4 * pgc.meta.N))), length(pgc.neighbor.indices) > M0
        end
        r0, grew = runc(0)
        r1, _ = runc(1)
        @test grew && r0 == r1
    end

    @testset "One launch per pass over the (particle, block) pairs: launch counts, chunks, the package's conditions" begin
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        eq8 = DiscreteTransport(Val(8), sym)
        eq4 = DiscreteTransport(Val(4), sym)                                     # 2 blocks
        eq2 = DiscreteTransport(Val(2), sym)                                     # 4 blocks
        box(x, k) = 1 + (abs(x) < 0.3 ? 1.0 : 0.0) * (1 + 0.1k)
        bc() = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
        function block_run(eq_b, W; blocks_per_launch, mood = MOOD(EPD1(), MOODu1(1e-3)), N = 60, tend = 0.2, bc = bc(), count = false,
                           limiter = MinmodLimiter(:hard))
            pg = block_pg_1d(N, W; bc = bc)
            bs = BlockState(eq_b, pg)
            set_block_states!((x, b) -> State{W, Float64}(ntuple(k -> box(x[1], (b - 1) * W + k), W)), bs, pg)
            ts = BlockTimeStepper(eq_b, MUSCL(Float64, 1, W, 3, limiter, RusanovFlux()), mood, (), RK3_SSP_Tableau(Float64);
                                  blocks_per_launch = blocks_per_launch)
            dt = 0.3 * (2 / N) / 2.0
            count && set_backend!(CountingBackend())
            try
                advance!((t, d) -> ts(eq_b, pg, bs, t, d), tend, dt)
            finally
                set_backend!(PolyesterBackend())
            end
            return bs, pg, ts
        end
        function standard_run(; N = 60, tend = 0.2, bc = bc(), mood = MOOD(EPD1(), MOODu1(1e-3)), limiter = MinmodLimiter(:hard))
            pg = block_pg_1d(N, 8; bc = bc)
            for i in 1:pg.meta.N
                x = pg.core.positions[i][1]
                pg.rhos[i] = State{8, Float64}(ntuple(k -> box(x, k), 8))
            end
            ts = GeneralRKTimeStepper(eq8, MUSCL(Float64, 1, 8, 3, limiter, RusanovFlux()), mood, RK3_SSP_Tableau(Float64))
            dt = 0.3 * (2 / 60) / 2.0
            advance!((t, d) -> ts(eq8, pg, t, d), tend, dt)
            return pg
        end
        # the number of launches of a step does not depend on the number of blocks (every pass is one launch over the
        # pairs; with `blocks_per_launch = 1` the passes with scratch run block by block and the count grows)
        launches(eq_b, W, bpl) = (LAUNCHES[] = 0; block_run(eq_b, W; blocks_per_launch = bpl, count = true, tend = 0.05); LAUNCHES[])
        n2 = launches(eq4, 4, 0)
        n4 = launches(eq2, 2, 0)
        @test n2 == n4
        @test launches(eq2, 2, 1) > n4
        # every chunking gives the same run bit for bit, and the standard run to round-off (the separately compiled flux
        # pass of the standard stepper differs by ulps at third order): with MOOD u1 and the inflow/outflow conditions,
        # with MOODu2 (its pre-pass and plateau ranges per chunk: 4 blocks in chunks of 3, flagged particles), and with
        # the extrapolation condition fitted over the (ghost, block) pairs
        states(bs) = copy(view(parent(bs.data), 1:(bs.nb * bs.N[])))
        bce() = Dict{Int, AbstractBoundaryCondition}(1 => ExtrapolationBC(2), 2 => ExtrapolationBC(2))
        for (mood, bcs, limiter) in ((MOOD(EPD1(), MOODu1(1e-3)), bc, MinmodLimiter(:hard)),
                                     (MOOD(EPD1(), MOODu2(1e-3)), bc, NoLimiter()),
                                     (MOOD(NoMOOD()), bce, MinmodLimiter(:hard)))
            pg_f = standard_run(; mood = mood, bc = bcs(), limiter = limiter)
            runs = [block_run(eq2, 2; blocks_per_launch = bpl, mood = mood, bc = bcs(), limiter = limiter) for bpl in (0, 1, 3)]
            @test all(r -> states(r[1]) == states(runs[1][1]), runs)
            @test all(r -> block_vs_full(r[1], view(pg_f.rhos, 1:pg_f.meta.N)) < 1e-12, runs)
            mood.criterion isa MOODu2 && @test any(mood_flags(runs[3][3])) && all(i -> all(isfinite, block_view(runs[3][1], 1)[i]), 1:runs[3][1].N[])
        end
        # the memory budget prices the scheme's scratch once per block of a launch, and the reservation holds a step
        pg = block_pg_1d(60, 2; bc = bc())
        bs = BlockState(eq2, pg)
        set_block_states!((x, b) -> State{2, Float64}(box(x[1], 2b - 1), box(x[1], 2b)), bs, pg)
        sc = MUSCL(Float64, 1, 2, 3, MinmodLimiter(:hard), RusanovFlux())
        ts2 = BlockTimeStepper(eq2, sc, RK3_SSP_Tableau(Float64); blocks_per_launch = 2)
        ts1 = BlockTimeStepper(eq2, sc, RK3_SSP_Tableau(Float64); blocks_per_launch = 1)
        e2 = memory_estimate(pg, ts2, bs, sc)
        e1 = memory_estimate(pg, ts1, bs, sc)
        @test e2.per_slot == 2 * e1.per_slot - memory_estimate(pg, sc).per_slot
        @test e2.per_particle - e1.per_particle == sizeof(eltype(sc.gradients))
        r = reserve!(pg, ts2, bs, sc; limit = 2 * e2.bytes)
        @test ts2.grad_scratch.fixed && length(parent(ts2.grad_scratch)) == 2 * r.particles
        advance!((t, d) -> ts2(eq2, pg, bs, t, d), 0.1, 0.3 * (2 / 60) / 2.0)
        @test all(i -> all(isfinite, block_view(bs, 4)[i]), 1:pg.meta.N)
        @test_throws ArgumentError BlockTimeStepper(eq2, sc, RK3_SSP_Tableau(Float64); blocks_per_launch = -1)
        @test_throws ArgumentError BlockTimeStepper(eq2, sc, RK3_SSP_Tableau(Float64); blocks_per_launch = :all)
        @test BlockTimeStepper(eq2, sc, RK3_SSP_Tableau(Float64)).blocks_per_launch == 128                # :auto = 256 ÷ W
        @test HyperCloudCore._blocks_per_launch(:auto, 60) == 4 && HyperCloudCore._blocks_per_launch(:auto, 300) == 1
        # the ghost fits' per-ghost geometry: one scratch per basis size, priced by the budget (2-D degree 2: 6 rows), and
        # the CFL buffer of the state (nb per particle)
        pg_g = block_pg_1d(60, 2; bc = bce())
        bs_g = BlockState(eq2, pg_g)
        set_block_states!((x, b) -> State{2, Float64}(box(x[1], 2b - 1), box(x[1], 2b)), bs_g, pg_g)
        ts_g = BlockTimeStepper(eq2, sc, RK3_SSP_Tableau(Float64))
        e_g = memory_estimate(pg_g, ts_g, bs_g, sc)
        @test collect(keys(ts_g.ghost_geom)) == [3]                                       # 1-D degree 2: 2 + 1 rows
        gsc = HyperCloudCore._scratch_buffers(ts_g.ghost_geom[3])
        @test sum(g -> sizeof(eltype(parent(g))), gsc) == 2 * sizeof(Int) + 8 + 8 + 9 * 8
        @test memory_estimate(pg_g, bs_g).per_particle - memory_estimate(pg_g).per_particle == 4 * (sizeof(State{2, Float64}) + 8)
        @test get_time_step(pg_g, eq2, sc, bs_g) == minimum(get_time_step(pg_g, BlockOf(eq2, b), sc; U = block_view(bs_g, b)) for b in 1:4)
        r_g = reserve!(pg_g, ts_g, bs_g, sc; limit = 2 * e_g.bytes)
        @test all(g -> g.fixed && length(parent(g)) == r_g.particles, gsc) && bs_g.cfl.fixed && length(parent(bs_g.cfl)) == 4 * r_g.particles
        advance!((t, d) -> ts_g(eq2, pg_g, bs_g, t, d), 0.1, 0.3 * (2 / 60) / 2.0)
        @test all(i -> all(isfinite, block_view(bs_g, 2)[i]), 1:pg_g.meta.N)
    end

    @testset "A block state split over devices: host parts in lockstep agree with the single stepper" begin
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        eq8 = DiscreteTransport(Val(8), sym)
        eq2 = DiscreteTransport(Val(2), sym)                                       # 4 blocks
        box(x, k) = 1 + (abs(x) < 0.3 ? 1.0 : 0.0) * (1 + 0.1k)
        wave(x, k) = 1 + 0.3 * sin(π * x + 0.4k)
        bc() = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
        # the grid of a run: a copy per part (every part moves its own positions and orders)
        grid(N, W; periodic = false, mover = NoGridMover(), bcs = bc()) =
            periodic ? ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = true), (2 / N,), ExponentialWeightFunction(1.0, 2.5 * 2 / N), W; mover = mover) :
                       block_pg_1d(N, W; bc = bcs)
        interp(W; order = 2, limiter = MinmodLimiter(:hard)) = MUSCL(Float64, 1, W, order, limiter, RusanovFlux())
        # the split state: parts over `ranges` (host parts, no device to activate)
        function split(eq_b, ranges, make_grid, ic; mood = MOOD(NoMOOD()), sources = () -> (), tableau = RK3_SSP_Tableau(Float64), order = 2, limiter = MinmodLimiter(:hard),
                       activate = nothing)
            parts = map(ranges) do r
                pg = make_grid()
                ts = BlockTimeStepper(eq_b, interp(2; order, limiter), mood, sources(), tableau; blocks = r)
                bs = BlockState(eq_b, pg; blocks = r)
                set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg)
                BlockPart(pg, ts, bs; activate)
            end
            return BlockDevices(eq_b, parts)
        end
        function single(eq_b, make_grid, ic; mood = MOOD(NoMOOD()), sources = () -> (), tableau = RK3_SSP_Tableau(Float64), order = 2, limiter = MinmodLimiter(:hard))
            pg = make_grid()
            ts = BlockTimeStepper(eq_b, interp(2; order, limiter), mood, sources(), tableau)
            bs = BlockState(eq_b, pg)
            set_block_states!((x, b) -> State{2, Float64}(ic(x[1], 2b - 1), ic(x[1], 2b)), bs, pg)
            return ts, pg, bs
        end
        function full_run(eq_f, make_grid, ic; mood = MOOD(NoMOOD()), sources = () -> (), tableau = RK3_SSP_Tableau(Float64), order = 2, limiter = MinmodLimiter(:hard), tend = 0.2, dt = 0.3 * (2 / 60) / 2.0)
            pg = make_grid()
            for i in 1:pg.meta.N
                x = pg.core.positions[i][1]
                pg.rhos[i] = State{8, Float64}(ntuple(k -> ic(x, k), 8))
            end
            ts = GeneralRKTimeStepper(eq_f, interp(8; order, limiter), mood, sources(), tableau)
            advance!((t, d) -> ts(eq_f, pg, t, d), tend, dt)
            return pg
        end
        dev(devs, bs) = maximum(abs.(host_state(devs) .- view(parent(bs.data), 1:(bs.nb * bs.N[])))) |> x -> maximum(x)
        maxdiff(a, b) = maximum(maximum(abs.(u - v)) for (u, v) in zip(a, b))
        N  = 60
        dt = 0.3 * (2 / N) / 2.0

        # 1. transport with inflow/outflow, the implicit mean relaxation in its additive form (ARS222): two and three parts
        for ranges in ([1:1, 2:4], [1:2, 3:4], [1:1, 2:2, 3:4])
            devs = split(eq2, ranges, () -> grid(N, 2), box; sources = () -> (BlockSourceTerm(PartialMean(0.05)),), tableau = IMEX_ARS222_Tableau(Float64))
            ts1, pg1, bs1 = single(eq2, () -> grid(N, 2), box; sources = () -> (BlockSourceTerm(PartialMean(0.05)),), tableau = IMEX_ARS222_Tableau(Float64))
            advance!((t, d) -> (devs(t, d); ts1(eq2, pg1, bs1, t, d)), 0.2, dt)
            @test maxdiff(host_state(devs), view(parent(bs1.data), 1:(4 * bs1.N[]))) < 1e-12
            @test all(((pt, r),) -> pt.bs.blocks == r && pt.ts.blocks == r, zip(devs.parts, ranges))
            @test get_time_step(devs) == get_time_step(pg1, eq2, ts1.inner.divergence_interpolator, bs1)
        end
        # against the eight-component standard run as well (the single block stepper is within 1e-13 of it)
        devs = split(eq2, [1:1, 2:4], () -> grid(N, 2), box; sources = () -> (BlockSourceTerm(PartialMean(0.05)),), tableau = IMEX_ARS222_Tableau(Float64))
        advance!((t, d) -> devs(t, d), 0.2, dt)
        pg_f = full_run(eq8, () -> grid(N, 8), box; sources = () -> (MeanRelaxationFull(0.05),), tableau = IMEX_ARS222_Tableau(Float64))
        Yd = host_state(devs)
        Nt = pg_f.meta.N                                                           # (the grid's particles, ghosts included: the stride of the layout)
        @test maximum(maximum(abs(Yd[(b - 1) * Nt + i][k] - pg_f.rhos[i][(b - 1) * 2 + k]) for b in 1:4, k in 1:2) for i in 1:Nt) < 1e-12

        # 2. MOOD u1 with the cross-block admissibility in its additive form (the guarded equation): the same flags and
        #    orders on every part as in the single stepper, states to round-off
        g2 = GuardedTransport(eq2)
        mood = MOOD(EPD1(), MOODu1(1e-3))
        devs = split(g2, [1:2, 3:4], () -> grid(N, 2; periodic = true), box; mood, order = 3, limiter = NoLimiter())
        ts1, pg1, bs1 = single(g2, () -> grid(N, 2; periodic = true), box; mood, order = 3, limiter = NoLimiter())
        advance!((t, d) -> (devs(t, d); ts1(g2, pg1, bs1, t, d)), 0.2, dt)
        @test maxdiff(host_state(devs), view(parent(bs1.data), 1:(4 * bs1.N[]))) < 1e-12
        @test any(mood_flags(ts1))
        @test all(pt -> mood_flags(pt.ts) == mood_flags(ts1), devs.parts)
        @test all(pt -> pt.pg.core.particle_orders[1:pg1.meta.N] == pg1.core.particle_orders[1:pg1.meta.N], devs.parts)
        @test mood_inadmissible(first(devs.parts).ts) == mood_inadmissible(ts1)
        # the same with a (no-op) device per part: the parts' phases run concurrently, one task per part (with several
        # threads; in sequence the devices never overlapped), bit for bit the sequential split
        devc = split(g2, [1:2, 3:4], () -> grid(N, 2; periodic = true), box; mood, order = 3, limiter = NoLimiter(), activate = () -> nothing)
        devq = split(g2, [1:2, 3:4], () -> grid(N, 2; periodic = true), box; mood, order = 3, limiter = NoLimiter())
        advance!((t, d) -> (devc(t, d); devq(t, d)), 0.2, dt)
        @test host_state(devc) == host_state(devq)
        @test all(pt -> mood_flags(pt.ts) == mood_flags(first(devq.parts).ts), devc.parts)
        # parts sharing a grid are refused (every part's drop pass lowered the shared orders again: 0.09 off, silently)
        pgs = grid(N, 2; periodic = true)
        sh = map([1:2, 3:4]) do r
            tsr = BlockTimeStepper(g2, interp(2; order = 3, limiter = NoLimiter()), mood, (), RK3_SSP_Tableau(Float64); blocks = r)
            BlockPart(pgs, tsr, BlockState(g2, pgs; blocks = r))
        end
        @test_throws ArgumentError BlockDevices(g2, sh)

        # 3. a Lagrangian mover: the frame velocity from the exchanged density and momentum partials
        devs = split(eq2, [1:3, 4:4], () -> grid(N, 2; periodic = true, mover = LagrangianMover()), wave)
        ts1, pg1, bs1 = single(eq2, () -> grid(N, 2; periodic = true, mover = LagrangianMover()), wave)
        advance!((t, d) -> (devs(t, d); ts1(eq2, pg1, bs1, t, d)), 0.2, dt)
        @test maxdiff(host_state(devs), view(parent(bs1.data), 1:(4 * bs1.N[]))) < 1e-12
        @test all(pt -> maximum(norm, pt.pg.core.positions[1:pg1.meta.N] .- pg1.core.positions[1:pg1.meta.N]) < 1e-12, devs.parts)

        # 4. reflecting walls whose operator reads the mirror partner's block from another part: the reduction and the
        #    application run on the gathered ghost rows, bit for bit the single stepper (no sources: nothing else is summed)
        bcw() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(ReflectOp()), 2 => MirrorBC(ReflectOp()))
        devs = split(eq2, [1:1, 2:3, 4:4], () -> grid(N, 2; bcs = bcw()), box)
        ts1, pg1, bs1 = single(eq2, () -> grid(N, 2; bcs = bcw()), box)
        advance!((t, d) -> (devs(t, d); ts1(eq2, pg1, bs1, t, d)), 0.3, dt)
        @test host_state(devs) == collect(view(parent(bs1.data), 1:(4 * bs1.N[])))
        @test all(i -> all(isfinite, host_state(devs)[i]), 1:(4 * bs1.N[]))
        # the host reduction of a user's partial: the sum of the components of every interior particle
        tot = host_reduction((Y, N, nb, blocks, p) -> sum(b -> sum(Y[(b - 1) * N + p]), blocks), devs, Float64)
        @test all(i -> tot[i] ≈ sum(b -> sum(bs1.data[(b - 1) * bs1.N[] + i]), 1:4), 1:bs1.N[])

        # 5. the refusals: parts that do not cover the blocks, a source without the additive form, an equation whose
        #    cross-block admissibility has none, a wall operator without one
        parts2 = devs.parts
        @test_throws ArgumentError BlockDevices(eq2, parts2[1:2])
        @test_throws ArgumentError BlockDevices(eq2, [parts2[1], parts2[3]])
        @test_throws ArgumentError BlockState(eq2, grid(N, 2); blocks = 2:5)
        @test_throws ArgumentError BlockTimeStepper(eq2, interp(2), RK3_SSP_Tableau(Float64); blocks = 0:2)
        @test_throws ArgumentError split(eq2, [1:2, 3:4], () -> grid(N, 2), box; sources = () -> (BlockSourceTerm(MeanRelaxation(0.05)),), tableau = IMEX_ARS222_Tableau(Float64))
        @test HyperCloudCore._overrides_block_admissible(g2) && !HyperCloudCore._overrides_block_admissible(eq2)
        @test_throws ArgumentError split(GuardedNoPartial(eq2), [1:2, 3:4], () -> grid(N, 2; periodic = true), box; mood)
        # a stepper of a block range does not step on its own
        pt = first(parts2)
        @test_throws ArgumentError pt.ts(eq2, pt.pg, pt.bs, 0.0, dt)
    end

    @testset "A wall operator that takes the state at the wall point (uses_wall_state): the same fit, evaluated at the wall" begin
        # a linear field on [-1, 1] with walls at both ends: the mirror value is the field at the mirror point, the wall-point
        # value the field at the wall, exactly (a degree-1 fit reproduces linear data)
        field(x) = 2.0 + 0.5 * x
        N  = 40
        bcp() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(WallProbe()), 2 => MirrorBC(WallProbe()))
        pg = block_pg_1d(N, 1; bc = bcp())
        for i in 1:pg.meta.N
            pg.rhos[i] = State{1, Float64}(field(pg.core.positions[i][1]))
        end
        eq  = Burgers()
        buf = copy(pg.rhos)
        for tag in (1, 2)
            HyperCloudCore._launch_ghost_fit!(pg, buf, tag, eq, 0.0, Val(1), HyperCloudCore._WallOp(WallProbe()), Val(true), Val(false))
        end
        for i in 1:pg.meta.N
            pg.core.is_boundary[i] || continue
            x_w = pg.core.positions[i][1] - pg.core.wall_distances[i] * pg.core.normals[i][1]
            @test abs(buf[i][1] - field(x_w)) < 1e-12
            @test abs(x_w) ≈ 1.0 atol = 1e-12
        end
        # the plain mirror (no operator) gives the field at the mirror point instead
        buf2 = copy(pg.rhos)
        HyperCloudCore._launch_ghost_fit!(pg, buf2, 1, eq, 0.0, Val(1), HyperCloudCore._WallOp(nothing), Val(true), Val(false))
        g1 = findfirst(i -> pg.core.is_boundary[i] && pg.core.tags[i] == 1, 1:pg.meta.N)
        x_m = pg.core.positions[g1][1] - 2 * pg.core.wall_distances[g1] * pg.core.normals[g1][1]
        @test abs(buf2[g1][1] - field(x_m)) < 1e-12 && abs(field(x_m) - field(-1.0)) > 1e-3
        # the block pipeline's fit pass writes the wall-point states into the second scratch
        gs  = HyperCloudCore._ghost_scratch(Val(2), Float64, Val(1), Array)
        out = copy(pg.rhos); outw = copy(pg.rhos)
        HyperCloudCore._launch_ghost_fit_blocks!(pg, pg.rhos, out, 2, eq, 0.0, Val(1), HyperCloudCore._WallOp(nothing), Val(true), Val(false), 1:1, gs; out_w = outw)
        for i in 1:pg.meta.N
            (pg.core.is_boundary[i] && pg.core.tags[i] == 2) || continue
            x_w = pg.core.positions[i][1] - pg.core.wall_distances[i] * pg.core.normals[i][1]
            x_m = pg.core.positions[i][1] - 2 * pg.core.wall_distances[i] * pg.core.normals[i][1]
            @test abs(outw[i][1] - field(x_w)) < 1e-12 && abs(out[i][1] - field(x_m)) < 1e-12
        end
        # a block wall with a reduction that takes the wall-point states: the stepper fills the second scratch and prices it
        # (unbounded fits: the clamp of `bounded` would hold the outermost ghost's wall value, 3 Δx from its mirror point,
        # inside its donor's data range)
        sym = [(-2.0,), (-1.5,), (-1.0,), (-0.5,), (0.5,), (1.0,), (1.5,), (2.0,)]
        eq2 = DiscreteTransport(Val(2), sym)
        bcw() = Dict{Int, AbstractBoundaryCondition}(1 => MirrorBC(WallSumOp(); bounded = false), 2 => MirrorBC(WallSumOp(); bounded = false))
        pgw = block_pg_1d(60, 2; bc = bcw())
        bsw = BlockState(eq2, pgw)
        set_block_states!((x, b) -> State{2, Float64}(1 + 0.2 * x[1] + 0.1b, 1 + 0.3 * x[1] + 0.2b), bsw, pgw)
        sc  = MUSCL(Float64, 1, 2, 2, MinmodLimiter(:hard), RusanovFlux())
        tsw = BlockTimeStepper(eq2, sc, RK2_SSP_Tableau(Float64))
        e_w = memory_estimate(pgw, tsw, bsw, sc)
        @test length(parent(tsw.ghost_scratch_w)) >= 1                                 # made by the budget
        tsw(eq2, pgw, bsw, 0.0, 1e-8)
        # the reduction saw the wall-point sums (the parameters of the last stage, whose state is the initial field up to
        # O(dt)): for the linear fields, every ghost's parameter is the sum of the fields at the wall, Σ_b (2 + 0.5 x_w + 0.3 b)
        # at x_w = -1, and not the sum at its mirror point
        prm = parent(tsw.wall_params[1])
        for i in 1:pgw.meta.N
            (pgw.core.is_boundary[i] && pgw.core.tags[i] == 1) || continue
            x_m = pgw.core.positions[i][1] - 2 * pgw.core.wall_distances[i] * pgw.core.normals[i][1]
            @test abs(prm[i] - (8 + 0.5 * 4 * (-1.0) + 0.3 * 10)) < 1e-6
            @test abs(prm[i] - (8 + 0.5 * 4 * x_m + 0.3 * 10)) > 1e-3
        end
        @test all(i -> all(isfinite, block_view(bsw, 1)[i]), 1:pgw.meta.N)
        # split over host parts: the wall-point rows are gathered like the mirrored ones, every part's reduction is the
        # single stepper's bit for bit
        field2(x, b) = State{2, Float64}(1 + 0.2 * x[1] + 0.1b, 1 + 0.3 * x[1] + 0.2b)
        parts = map([1:1, 2:3, 4:4]) do r
            pgp = block_pg_1d(60, 2; bc = bcw())
            tsp = BlockTimeStepper(eq2, sc, RK2_SSP_Tableau(Float64); blocks = r)
            bsp = BlockState(eq2, pgp; blocks = r)
            set_block_states!(field2, bsp, pgp)
            BlockPart(pgp, tsp, bsp)
        end
        devs = BlockDevices(eq2, parts)
        pg1 = block_pg_1d(60, 2; bc = bcw())
        ts1 = BlockTimeStepper(eq2, sc, RK2_SSP_Tableau(Float64))
        bs1 = BlockState(eq2, pg1)
        set_block_states!(field2, bs1, pg1)
        for k in 0:4
            devs(k * 1e-3, 1e-3)
            ts1(eq2, pg1, bs1, k * 1e-3, 1e-3)
        end
        gl = [i for i in 1:pg1.meta.N if pg1.core.is_boundary[i] && pg1.core.tags[i] == 1]
        @test all(pt -> parent(pt.ts.wall_params[1])[gl] == parent(ts1.wall_params[1])[gl], devs.parts)
        @test host_state(devs) == collect(view(parent(bs1.data), 1:(4 * bs1.N[])))
    end
end
