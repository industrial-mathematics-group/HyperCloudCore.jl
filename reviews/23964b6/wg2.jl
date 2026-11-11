using HyperCloudCore, StaticArrays, Random, LinearAlgebra
segd(p, a, b) = (t = clamp(dot(p - a, b - a) / dot(b - a, b - a), 0, 1); norm(p - (a + t * (b - a))))
for αd in (3, 5, 7, 9, 11, 13, 15)
    α = deg2rad(αd)
    s = intersect(Cuboid((-5.0, 0.0), (5.0, 5.0)), HalfSpace((0.0, 0.0), (-sin(α), cos(α))))
    g = GeometricDomain(s)
    rng = MersenneTwister(1); worst = 0.0; nbad = 0; nout = 0; tt = 0.0; tmax = 0.0; mout = 0
    for _ in 1:5000
        x = 0.3 * (2 * rand(rng, SVector{2, Float64}) .- 1)
        signed_distance(s, x) > 0 || continue
        nout += 1
        ex = min(segd(x, SVector(0.0, 0.0), SVector(5.0, 0.0)), segd(x, SVector(0.0, 0.0), 5 * SVector(cos(α), sin(α))))
        t0 = time_ns(); n, φ = wall_geometry(g, x); t = (time_ns() - t0) * 1e-9
        tt += t; tmax = max(tmax, t)
        e = abs(φ - ex) / ex; worst = max(worst, e); e > 1e-6 && (nbad += 1)
    end
    println("α=$(αd)°: outside $nout, max rel err $(round(worst, sigdigits=3)), >1e-6: $nbad, mean $(round(tt/nout*1e6, digits=1)) us, worst $(round(tmax*1e3, digits=2)) ms")
end
