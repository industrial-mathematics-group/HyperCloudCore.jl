# Discrete-velocity transport ∂_t f_k + v_k·∇f_k = 0, k = 1 … NB·W: the block equation of the tests. Block b holds the
# velocities (b − 1) W + 1 … b W; every component is transported on its own, so the exact solution is
# f_k(x, t) = f_k(x − v_k t, 0). With one block (NB = 1) it is an ordinary PDE of M = W components as well (`flux`),
# which the block pipeline must reproduce. The wave speed bound is the largest speed of all velocities along the
# axis (not of the block), so that the generic Rusanov dissipation is the same in both pipelines.
struct DiscreteTransport{D, W, T, NB, L} <: HyperbolicPDE{D, W, T}
    speeds::SVector{L, Space{D, T}}
    vmax::Space{D, T}
end

function DiscreteTransport(::Val{W}, speeds::AbstractVector{<:NTuple{D, Real}}; T::Type = Float64) where {W, D}
    L = length(speeds)
    L % W == 0 || error("DiscreteTransport: $L velocities do not split into blocks of $W")
    sp = SVector{L, Space{D, T}}(ntuple(k -> Space{D, T}(speeds[k]), L))
    vmax = Space{D, T}(ntuple(d -> maximum(k -> abs(sp[k][d]), 1:L), Val(D)))
    return DiscreteTransport{D, W, T, L ÷ W, L}(sp, vmax)
end

HyperCloudCore.num_blocks(::DiscreteTransport{D, W, T, NB}) where {D, W, T, NB} = NB

@inline function HyperCloudCore.block_flux(eq::DiscreteTransport{D, W, T}, U::State{W, T}, b::Int) where {D, W, T}
    off = (b - 1) * W
    return Flux{D, W, T}(ntuple(d -> State{W, T}(ntuple(k -> (@inbounds eq.speeds[off + k][d]) * U[k], Val(W))), Val(D)))
end
@inline HyperCloudCore.block_max_eigenvalue(eq::DiscreteTransport, U::State, d::Int, b::Int) = @inbounds eq.vmax[d]
# In a moving frame the exact bound over all velocities (the same for the full state and every block, so that the two
# pipelines use the same dissipation; it vanishes where every velocity equals the frame velocity)
@inline function _shifted_bound(eq::DiscreteTransport{D, W, T, NB, L}, d::Int, shift::Real) where {D, W, T, NB, L}
    m = zero(T)
    @inbounds for k in 1:L
        m = max(m, abs(eq.speeds[k][d] - T(shift)))
    end
    return m
end
@inline HyperCloudCore.block_max_eigenvalue(eq::DiscreteTransport, U::State, d::Int, b::Int, shift::Real) = _shifted_bound(eq, d, shift)
@inline max_eigenvalue(eq::DiscreteTransport, U::State, d::Int, shift::Real) = _shifted_bound(eq, d, shift)

# One block: the plain PDE of the standard pipeline
@inline flux(eq::DiscreteTransport{D, W, T, 1}, U::State{W, T}) where {D, W, T} = HyperCloudCore.block_flux(eq, U, 1)
@inline max_eigenvalue(eq::DiscreteTransport, U::State, d::Int) = @inbounds eq.vmax[d]

"Velocity of component `k` of the full system."
transport_speed(eq::DiscreteTransport, k::Int) = eq.speeds[k]
