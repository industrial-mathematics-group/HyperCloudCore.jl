# Isothermal Euler equations in 1-D, U = (ρ, ρu), p = c² ρ: the simplest system with a density equation, for the
# conservation tests of the Lagrangian frame (its mass flux ρ(u − w) vanishes for w = u, unlike the flux of a scalar).
struct IsothermalEuler{T} <: HyperbolicPDE{1, 2, T}
    c::T
end
IsothermalEuler(c::Real = 1.0) = IsothermalEuler{float(typeof(c))}(float(c))

@inline prim2cons(::IsothermalEuler, U::State) = U
@inline cons2prim(::IsothermalEuler, W::State) = W
@inline function flux(eq::IsothermalEuler{T}, U::State{2, T}) where {T}
    ρ, m = U[1], U[2]
    return Flux{1, 2, T}((State{2, T}(m, m * m / ρ + eq.c^2 * ρ),))
end
@inline max_eigenvalue(eq::IsothermalEuler, U::State, d::Int) = abs(U[2] / U[1]) + eq.c
@inline max_eigenvalue(eq::IsothermalEuler{T}, U::State, d::Int, shift::Real) where {T} = abs(U[2] / U[1] - T(shift)) + eq.c
@inline function velocity(eq::IsothermalEuler{T}, U::State, d::Int) where {T}
    u = U[2] / U[1]
    return SMatrix{2, 2, T}(zero(T), eq.c^2 - u^2, one(T), 2u)      # ∂F/∂U, column major
end
@inline fluid_velocity(::IsothermalEuler{T}, U::State{2, T}) where {T} = Space{1, T}(U[2] / U[1])
@inline HyperCloudCore.is_admissible(::IsothermalEuler, U::State) = all(isfinite, U) && U[1] > 0   # (not imported here: qualified)
