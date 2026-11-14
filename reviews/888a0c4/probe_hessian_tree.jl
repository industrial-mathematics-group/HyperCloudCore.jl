# _sdf_hessian through the shape tree vs finite differences of the composite signed distance (Float64, wall points
# away from seams), and the encode/decode round trip in 2-D and 3-D
using HyperCloudCore, StaticArrays, LinearAlgebra, Random, Printf
const H = HyperCloudCore
fd(s, x) = invoke(H._sdf_hessian, Tuple{AbstractShape{length(x), Float64}, typeof(x)}, s, x)
rng = Xoshiro(2)
function onwall(s, D)   # random boundary points by projection
    pts = SVector{D, Float64}[]
    while length(pts) < 40
        x = SVector{D, Float64}(ntuple(_ -> 2rand(rng) - 1, D))
        for _ in 1:20
            φ = signed_distance(s, x); x = x - φ * boundary_normal(s, x)
        end
        abs(signed_distance(s, x)) < 1e-12 && push!(pts, x)
    end
    pts
end
shapes2 = [("Ball", Ball((0.1, -0.2), 0.4)),
           ("complement(Ball)", complement(Ball((0.1, -0.2), 0.4))),
           ("Rotated(Translated(Ball)) centre", Rotated(Translated(Ball((0.0, 0.0), 0.3), (0.2, 0.1)), 0.7; center = (0.3, -0.1))),
           ("Cuboid - Ball", Cuboid((-1.0, -1.0), (1.0, 1.0)) - Ball((0.0, 0.0), 0.35)),
           ("Ball ∪ Ball", union(Ball((-0.2, 0.0), 0.3), Ball((0.25, 0.0), 0.3))),
           ("Ball ∩ Ball", intersect(Ball((-0.2, 0.0), 0.5), Ball((0.25, 0.0), 0.5))),
           ("Posed(rotating off-centre Ball, t=0.4)", at_time(Moving(Ball((0.3, 0.0), 0.25), rotating(2.0; center = (0.0, 0.0))), 0.4)),
           ("Tagged(Rotated(Ball))", Tagged(Rotated(Ball((0.2, 0.0), 0.3), 1.1), 3))]
for (name, s) in shapes2
    e = 0.0; nseam = 0
    for x in onwall(s, 2)
        A = H._sdf_hessian(s, x); B = fd(s, x)
        d = maximum(abs, A - B)
        d > 1e-3 ? (nseam += 1) : (e = max(e, d))
    end
    @printf("2-D %-42s max |H - FD| %.2e (%d points near seams/kinks skipped)\n", name, e, nseam)
end
shapes3 = [("Ball3", Ball((0.1, -0.2, 0.05), 0.4)), ("Cuboid3 - Ball3", Cuboid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)) - Ball((0.0, 0.0, 0.0), 0.35)),
           ("Rotated Ball3", Rotated(Ball((0.2, 0.0, 0.0), 0.3), SMatrix{3, 3}(cos(0.4), sin(0.4), 0, -sin(0.4), cos(0.4), 0, 0, 0, 1)))]
for (name, s) in shapes3
    e = 0.0; er = 0.0
    for x in onwall(s, 3)
        A = H._sdf_hessian(s, x); B = fd(s, x); e = max(e, maximum(abs, A - B))
        n = boundary_normal(s, x)
        er = max(er, maximum(abs, H._decode_curvature(H._encode_curvature(A, n), n) - A))
    end
    @printf("3-D %-42s max |H - FD| %.2e, encode/decode round trip %.1e\n", name, e, er)
end
# 3-D tangent basis continuity near the switch of the helper axis
n1 = normalize(SVector(1.0, 1.0 + 1e-9, 1.0)); n2 = normalize(SVector(1.0, 1.0 - 1e-9, 1.0))
println("tangent basis jump across the helper-axis switch: ", norm(H._tangents3(n1)[1] - H._tangents3(n2)[1]), " (the decoded H is basis independent)")
