# Grid construction time with a user shape (FD Hessian at every particle) and with a built-in Ball
using HyperCloudCore, StaticArrays, LinearAlgebra
struct UCircle{T} <: AbstractShape{2, T}; c::SVector{2, T}; r::T; end
HyperCloudCore.signed_distance(s::UCircle{T}, x::SVector{2, T}) where {T} = norm(x - s.c) - s.r
HyperCloudCore.bounding_box(s::UCircle) = (Tuple(s.c .- s.r), Tuple(s.c .+ s.r))
for (lab, ob) in (("user circle", UCircle(SVector(0.0, 0.0), 0.3)), ("built-in Ball", Ball((0.0, 0.0), 0.3)))
    geom = GeometricDomain(Cuboid((-1.0, -1.0), (1.0, 1.0)) - Tagged(ob, 5); bc_map = Dict{Int, AbstractBoundaryCondition}(t => OutflowBC() for t in 1:5))
    ParticleGrid(geom, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
    t = @elapsed ParticleGrid(geom, (0.005, 0.005), ExponentialWeightFunction(1.0, 0.0125), 1)
    println(rpad(lab, 14), " 400 x 400 grid built in ", round(t, digits = 2), " s")
end
