# Legacy 7-argument constructor with argument types the old default constructor converted.
using HyperCloudCore, StaticArrays
f(x) = all(0 .<= x .<= 1); g(x) = f(x) ? 0 : 1
for (name, args) in (("Dict{Int,OutflowBC}", (Val(:l), SVector(0.0, 0.0), SVector(1.0, 1.0), SVector(false, false), f, g, Dict(1 => OutflowBC()))),
                     ("Dict{Int,ABC}", (Val(:l), SVector(0.0, 0.0), SVector(1.0, 1.0), SVector(false, false), f, g, Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC()))),
                     ("SVector{2,Float32} mins with T=F64 maxs", (Val(:l), SVector(0f0, 0f0), SVector(1.0, 1.0), SVector(false, false), f, g, Dict{Int, AbstractBoundaryCondition}())),
                     ("is_periodic as Tuple", (Val(:l), SVector(0.0, 0.0), SVector(1.0, 1.0), (false, false), f, g, Dict{Int, AbstractBoundaryCondition}())))
    try
        gd = GeometricDomain(args...)
        println(name, ": ok ", typeof(gd).name.name, " ", length(typeof(gd).parameters), " params")
    catch e
        println(name, ": ", typeof(e))
    end
end
