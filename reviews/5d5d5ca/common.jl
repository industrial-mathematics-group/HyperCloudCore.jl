using HyperCloudCore, StaticArrays, LinearAlgebra, Printf
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity, param2vel
const H = HyperCloudCore
const BCD = Dict{Int, AbstractBoundaryCondition}
# Notice the addition of the L parameter in the struct and the vel field
struct LinearAdvection{D, M, T, L} <: HyperbolicPDE{D, M, T}
    vel::Velocity{D, M, T, L}
    max_eigs::SVector{D, T}
end

# Smart constructor
function LinearAdvection(velocities, ::Type{T}=eltype(velocities[1])) where {T}
    vel_svec = param2vel(velocities, T)
    
    D = length(vel_svec)
    M = size(vel_svec[1], 1)
    L = M * M # Extract the total length parameter
    
    # Precompute the exact spectral radius
    max_eigs = SVector{D, T}(ntuple(d -> T(maximum(abs.(eigvals(Matrix(vel_svec[d]))))), Val(D)))
    
    return LinearAdvection{D, M, T, L}(vel_svec, max_eigs)
end

@inline prim2cons(::LinearAdvection, U::State) = U
@inline cons2prim(::LinearAdvection, W::State) = W

@inline function flux(eq::LinearAdvection{D, M, T}, U::State) where {D, M, T}
    return Flux{D, M, T}(ntuple(Val(D)) do d
        State{M, T}(eq.vel[d] * U)
    end)
end

@inline function max_eigenvalue(eq::LinearAdvection, U::State, d::Int)
    return eq.max_eigs[d]
end

# Exact shifted wave speed for scalar advection (ALEFrame): |c_d − w_d|
@inline max_eigenvalue(eq::LinearAdvection{D, 1, T}, U::State, d::Int, shift::Real) where {D, T} = abs(eq.vel[d][1, 1] - T(shift))

# API implementation returning the strict M x M matrix
@inline velocity(eq::LinearAdvection, U::State, d::Int) = eq.vel[d]

# Transport velocity for Lagrangian grid movers (scalar systems only: the velocity is unambiguous)
@inline function fluid_velocity(eq::LinearAdvection{D, 1, T}, U::State) where {D, T}
    return Space{D, T}(ntuple(d -> eq.vel[d][1, 1], Val(D)))
end

interior(pg) = [i for i in 1:pg.meta.N if !pg.core.is_boundary[i]]
ghosts(pg, tagmin=5) = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] >= tagmin]
sumV(pg) = sum(pg.core.volumes[i] for i in interior(pg))
