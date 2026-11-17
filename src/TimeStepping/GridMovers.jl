export LagrangianMover, ScaledLagrangianMover, PrescribedMover
export ALEFrame, LagrangianFrame, grid_velocity, fluid_velocity, refresh_geometry!, set_positions!

# =========================================================================
# MOVING GRIDS — ALE COUPLING
# =========================================================================
#
# Particle positions are promoted to stage variables of the same Butcher tableau:
#
#     dx_i/dt = w_i(x, ρ, t)                              (explicit part of the tableau only)
#     dρ_i/dt = -div (F(ρ) − w_i ρ)|_i + S_i                (at the current stage positions)
#
# The second line is the ALE form for a collocation (pointwise) meshfree scheme: for a particle
# moving with w, Dρ/Dt = ∂_t ρ + w·∇ρ = −div F + w·∇ρ = −div(F − w_i ρ) for a frozen w_i.
# The relative flux is handled by wrapping the PDE in an `ALEFrame`, so the numerical flux,
# its dissipation and the limiters all see the relative velocity. There is no operator
# splitting, so the coupled system inherits the order of the tableau.
#
# Within a step the neighbour TOPOLOGY is frozen (indices from the start-of-step search) and only
# distances and weights are refreshed at each stage position; pairs that drift beyond R get zero
# weight, pairs that enter R during the step are picked up by the full search at the end of the
# step. With CFL-limited dt and R ≈ 3 dx this is the usual "neighbour skin" assumption. Set
# `full_search_per_stage = true` on the stepper to rebuild the topology at every stage instead.

# -------------------------------------------------------------------------
# Movers
# -------------------------------------------------------------------------
"""
    LagrangianMover <: GridMover

`w = fluid_velocity(eq, U)` (the PDE must implement `fluid_velocity`).
"""
struct LagrangianMover <: GridMover end

"""
    ScaledLagrangianMover(α) <: GridMover

`w = α · fluid_velocity(eq, U)`; `α = 1` is Lagrangian, `α = 0` is Eulerian.
"""
struct ScaledLagrangianMover{T} <: GridMover
    alpha::T
end

"""
    PrescribedMover(f) <: GridMover

`w = f(x::Space, t)`, independent of the state (e.g. a rigid rotation or a mesh-smoothing field).
"""
struct PrescribedMover{F} <: GridMover
    velocity::F
end

"""
    grid_velocity(mover, eq, p_idx, U, pg, t) -> Space{D, T}

Grid velocity of particle `p_idx` at the current stage (positions in `pg.core.positions`).
"""
@inline grid_velocity(::NoGridMover, eq, p_idx, U, pg::ParticleGrid{D, M, T}, t) where {D, M, T} = zero(Space{D, T})
@inline grid_velocity(::LagrangianMover, eq, p_idx, U, pg, t) = fluid_velocity(eq, U)
@inline grid_velocity(m::ScaledLagrangianMover, eq, p_idx, U, pg, t) = m.alpha * fluid_velocity(eq, U)
@inline grid_velocity(m::PrescribedMover, eq, p_idx, U, pg, t) = m.velocity(pg.core.positions[p_idx], t)

# -------------------------------------------------------------------------
# ALE frame: the relative flux F − w_i ρ
# -------------------------------------------------------------------------
"""
    ALEFrame(eq, w) <: HyperbolicPDE

`eq` seen from a frame moving with the constant velocity `w`: `flux = F(U) − w ⊗ U`, Jacobians
shifted by `−w_d I`, wave speeds by `|λ − w_d|`. The stepper wraps the PDE with the particle's
own grid velocity `w_i` before calling the divergence interpolator, so for a constant `w_i`
`div(F − w_i ρ) = div F − w_i·∇ρ` holds exactly, the numerical dissipation scales with the
relative speed, and for `w = u` every interface flux vanishes (exact Lagrangian transport).
"""
struct ALEFrame{D, M, T, EQ <: HyperbolicPDE{D, M, T}} <: HyperbolicPDE{D, M, T}
    eq::EQ
    w::Space{D, T}
end

@inline function flux(a::ALEFrame{D, M, T}, U::State{M, T}) where {D, M, T}
    F = flux(a.eq, U)
    return Flux{D, M, T}(ntuple(d -> F[d] - a.w[d] * U, Val(D)))
end
@inline max_eigenvalue(a::ALEFrame, U::State, d::Int) = max_eigenvalue(a.eq, U, d, a.w[d])
@inline velocity(a::ALEFrame{D, M, T}, U::State, d::Int) where {D, M, T} =
    velocity(a.eq, U, d) - a.w[d] * SMatrix{M, M, T}(I)
@inline prim2cons(a::ALEFrame, U::State) = prim2cons(a.eq, U)
@inline cons2prim(a::ALEFrame, W::State) = cons2prim(a.eq, W)
@inline evaluate_nc_jump(a::ALEFrame{D, M, T}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}) where {D, M, T} =
    evaluate_nc_jump(a.eq, f_L, f_R, dist_k)

# -------------------------------------------------------------------------
# Lagrangian frame: the conservative form on a cloud that moves with the fluid
# -------------------------------------------------------------------------
"""
    LagrangianFrame(eq, α) <: HyperbolicPDE

`eq` seen from the frame moving with `w(U) = α fluid_velocity(eq, U)`, the velocity of a [`LagrangianMover`](@ref)
(`α = 1`) or a [`ScaledLagrangianMover`](@ref): `flux = F(U) − w(U) ⊗ U`, a function of the state alone, wave speeds
`|λ − w_d|`, Jacobians shifted by `−w_d I`. The stepper wraps the PDE in it for these movers (with the MUSCL scheme),
and the scheme adds to every pair flux difference the term `(w_ij − w_i) ⊗ U_i`, with the interface velocity
`w_ij = (w(U_ij) + w(U_ji)) / 2` ([`ale_volume_term`](@ref)), so that the particle's update is

    dU_i/dt = −div(F − w U) − U_i div w ,

the conservative form of `D U/D t` in point values (`d(V U)/dt = −V div(F − wU)` with `dV/dt = V div w`, both
through the same discrete divergence). With the particle's frozen velocity ([`ALEFrame`](@ref)) the update is
`−div F + w_i·∇U`, a non-conservative product where `w` jumps: a Lagrangian cloud gained mass at a shock at a rate
independent of the resolution and the shock did not converge to its position (Sod with Euler and BGK, VK and
EPD2: +0.1–0.5 % of the mass by t = 0.2, the shock 0.011–0.017 too far at every N; the less dissipative the flux,
the larger the gain). See the moving-grid docs for the measurements.
"""
struct LagrangianFrame{D, M, T, EQ <: HyperbolicPDE{D, M, T}} <: HyperbolicPDE{D, M, T}
    eq::EQ
    alpha::T
end

"The frame's velocity at the state `U`: `α fluid_velocity(eq, U)`."
@inline frame_velocity(fr::LagrangianFrame{D, M, T}, U::State{M, T}) where {D, M, T} = fr.alpha * fluid_velocity(fr.eq, U)

@inline function flux(fr::LagrangianFrame{D, M, T}, U::State{M, T}) where {D, M, T}
    F = flux(fr.eq, U)
    w = frame_velocity(fr, U)
    return Flux{D, M, T}(ntuple(d -> F[d] - w[d] * U, Val(D)))
end
@inline max_eigenvalue(fr::LagrangianFrame, U::State, d::Int) = max_eigenvalue(fr.eq, U, d, frame_velocity(fr, U)[d])
@inline velocity(fr::LagrangianFrame{D, M, T}, U::State, d::Int) where {D, M, T} =
    velocity(fr.eq, U, d) - frame_velocity(fr, U)[d] * SMatrix{M, M, T}(I)
@inline prim2cons(fr::LagrangianFrame, U::State) = prim2cons(fr.eq, U)
@inline cons2prim(fr::LagrangianFrame, W::State) = cons2prim(fr.eq, W)
@inline is_admissible(fr::LagrangianFrame, U::State) = is_admissible(fr.eq, U)
@inline evaluate_nc_jump(fr::LagrangianFrame{D, M, T}, f_L::Flux{D, M, T}, f_R::Flux{D, M, T}, dist_k::Space{D, T}) where {D, M, T} =
    evaluate_nc_jump(fr.eq, f_L, f_R, dist_k)

"""
    ale_volume_term(eq, U_i, U_ij, U_ji) -> Flux{D, M, T}

The term a scheme adds to the flux difference of the pair `(i, j)` on a moving grid: zero for every equation except a
[`LagrangianFrame`](@ref), where it is `(w_ij − w_i) ⊗ U_i` with `w_ij` the mean of the frame velocities of the two
interface states and `w_i` the particle's own. Contracted with the divergence weights it gives `U_i div w`, the
volume change of the particle's share, which turns the pointwise update into the conservative form.
"""
@inline ale_volume_term(::HyperbolicPDE{D, M, T}, f_i::State{M, T}, f_ij::State{M, T}, f_ji::State{M, T}) where {D, M, T} =
    zero(Flux{D, M, T})
@inline function ale_volume_term(fr::LagrangianFrame{D, M, T}, f_i::State{M, T}, f_ij::State{M, T}, f_ji::State{M, T}) where {D, M, T}
    dw = T(0.5) * (frame_velocity(fr, f_ij) + frame_velocity(fr, f_ji)) - frame_velocity(fr, f_i)
    return Flux{D, M, T}(ntuple(d -> dw[d] * f_i, Val(D)))
end

# A frame whose pairs carry the numerical dissipation with the antisymmetrised face vector ½(V_i c_ij − V_j c_ji)
# (MUSCL's `_flux_loop`): on a Lagrangian cloud the particles compress through a shock by the density ratio, the MLS
# weights of the pairs across it are systematically non-antisymmetric (on a jittered static cloud the asymmetry is
# random and averages out), and the Rusanov dissipation of the density, the only pair term of the mass equation in
# the Lagrangian frame, gained mass at a rate independent of N. The central part keeps the consistent weights.
@inline _pairwise_dissipation(::HyperbolicPDE) = false
@inline _pairwise_dissipation(::LagrangianFrame) = true

# The frame the divergence interpolator sees on a moving grid: the conservative Lagrangian frame for the Lagrangian
# movers with a scheme that adds the volume term (`_conservative_ale`: MUSCL, set in MUSCL.jl), the frozen-velocity
# ALE frame for every other mover and scheme (a prescribed smooth velocity has no product problem)
@inline _conservative_ale(interp) = false
@inline _moving_frame(interp, mover, eq, w) = ALEFrame(eq, w)
@inline _moving_frame(interp, ::LagrangianMover, eq::HyperbolicPDE{D, M, T}, w) where {D, M, T} =
    _conservative_ale(interp) ? LagrangianFrame(eq, one(T)) : ALEFrame(eq, w)
@inline _moving_frame(interp, m::ScaledLagrangianMover, eq::HyperbolicPDE{D, M, T}, w) where {D, M, T} =
    _conservative_ale(interp) ? LagrangianFrame(eq, T(m.alpha)) : ALEFrame(eq, w)

# -------------------------------------------------------------------------
# Geometry refresh with frozen topology
# -------------------------------------------------------------------------
"""
    refresh_geometry!(pg)

Recomputes neighbour distances and weights from the current `pg.core.positions` for the existing
neighbour lists. Pairs now farther than `R` get weight zero.
"""
function refresh_geometry!(pg::ParticleGrid{D, M, T}) where {D, M, T}
    nb = pg.neighbor
    foreach_particle(_refresh_slots!, get_backend(), pg.meta.N,
        (pg.core.positions, nb.ranges, nb.indices, nb.distances, nb.weights, pg.domain.L_wrap, pg.domain.invL_wrap,
         pg.meta.R^2, nb.weight_func, zero(T)))
    pg.meta.geometry_version += 1
    return nothing
end

# Kernel body: distances and weights of the existing neighbour slots of particle `i`.
@inline function _refresh_slots!(i, pos, ranges, indices, distances, weights, L, L_inv, R_sq, wf, zero_T)
    @inbounds for k in ranges[i]
        j    = indices[k]
        dist = get_distance(pos, i, j, L, L_inv)
        d2   = sum(abs2, dist)
        distances[k] = dist
        weights[k]   = d2 <= R_sq ? wf(d2) : zero_T
    end
    return nothing
end

"""
    set_positions!(pg, X, full_search::Bool)

Writes stage positions `X` into the grid and refreshes the geometry: distances/weights only
(`full_search = false`) or a complete neighbour search (`true`, also re-wraps periodic axes).
"""
function set_positions!(pg::ParticleGrid, X::AbstractVector, full_search::Bool)
    N = pg.meta.N
    copyto!(pg.core.positions, 1, parent(X), 1, N)
    full_search ? pg.neighbor(pg) : refresh_geometry!(pg)
    return nothing
end
