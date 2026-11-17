# Scalar Burgers equation u_t + (u²/2)_x = 0 in 1-D: the simplest nonlinear flux (shocks) with a state-dependent
# transport velocity, for the Lagrangian movers (`fluid_velocity = u`).
struct Burgers{T} <: HyperbolicPDE{1, 1, T} end
Burgers(::Type{T} = Float64) where {T} = Burgers{T}()

@inline prim2cons(::Burgers, U::State) = U
@inline cons2prim(::Burgers, W::State) = W
@inline flux(::Burgers{T}, U::State{1, T}) where {T} = Flux{1, 1, T}((State{1, T}(U[1]^2 / 2),))
@inline max_eigenvalue(::Burgers, U::State, d::Int) = abs(U[1])
@inline max_eigenvalue(::Burgers{T}, U::State, d::Int, shift::Real) where {T} = abs(U[1] - T(shift))   # |λ − w| exactly
@inline velocity(::Burgers{T}, U::State, d::Int) where {T} = SMatrix{1, 1, T}(U[1])
@inline fluid_velocity(::Burgers{T}, U::State{1, T}) where {T} = Space{1, T}(U[1])
