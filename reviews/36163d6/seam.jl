using HyperCloudCore, StaticArrays
const H = HyperCloudCore
T = Float64
g = GeometricDomain(intersect(HalfSpace((0.0, 0.013), (0.0, -1.0)), HalfSpace((0.0, 0.98), (0.0, 1.0))); is_periodic = (true, false), mins = (0.0, 0.013), maxs = (1.0, 0.98))
wrap(p) = SVector(mod(p[1], 1.0), p[2])
h = 0.04
for (lbl, innerpt, nearpt) in (("interior across seam", SVector(0.999, 0.0135), SVector(0.001, 0.003)),
                               ("two snapped across seam", nothing, SVector(0.001, 0.003)),
                               ("same side (control)", SVector(0.003, 0.0135), SVector(0.001, 0.003)))
    inner = SVector{2, T}[]; ghosts = SVector{2, T}[]
    near = [nearpt]
    if innerpt === nothing
        near = [SVector(0.9995, 0.003), nearpt]
    else
        push!(inner, innerpt)
    end
    H._snap_near_wall!(inner, ghosts, near, g, h / 2, wrap)
    println(lbl, ": inner ", inner, " ghosts ", ghosts)
end
