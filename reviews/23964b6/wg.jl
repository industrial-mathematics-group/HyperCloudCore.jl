using HyperCloudCore, StaticArrays, Random, LinearAlgebra
segd(p, a, b) = (t = clamp(dot(p - a, b - a) / dot(b - a, b - a), 0, 1); norm(p - (a + t * (b - a))))
R(θ) = SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
c = SVector(0.2, 0.1)
for α in (0.1, 0.3, 1.0), θ in (0.0, 0.7)
    tri = intersect(Cuboid((-1.0, 0.0), (1.0, 1.0)), HalfSpace((0.0, 0.0), (-sin(α), cos(α))))
    s = θ == 0 ? tri : Rotated(tri, θ; center = (0.2, 0.1))
    V = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, tan(α))]
    V = [R(θ) * (v - c) + c for v in V]
    g = GeometricDomain(s; mins = (-3.0, -3.0), maxs = (3.0, 3.0))
    rng = MersenneTwister(2); worst = 0.0; wp = nothing; nbad = 0; nout = 0; tmax = 0.0
    for _ in 1:20000
        x = SVector(-1.0, -1.0) + 3 * rand(rng, SVector{2, Float64})
        signed_distance(s, x) > 0 || continue
        nout += 1
        ex = minimum(segd(x, V[k], V[mod1(k + 1, 3)]) for k in 1:3)
        t0 = time_ns(); φ = wall_geometry(g, x)[2]; t = (time_ns() - t0) * 1e-9
        tmax = max(tmax, t)
        e = abs(φ - ex) / ex
        e > worst && (worst = e; wp = x)
        e > 1e-6 && (nbad += 1)
    end
    println("α=$(round(rad2deg(α),digits=1))° θ=$θ: outside samples $nout, max rel err $worst at $wp, >1e-6: $nbad, worst call $(round(tmax*1e6, digits=1)) us")
end
# grid construction cost: 17° wedge domain vs a box, h = 1/200
for (nm, s) in (("box", Cuboid((-1.0, 0.0), (1.0, 1.0))), ("wedge 17°", intersect(Cuboid((-1.0, 0.0), (1.0, 1.0)), HalfSpace((0.0, 0.0), (-sin(0.3), cos(0.3))))), ("wedge 6°", intersect(Cuboid((-1.0, 0.0), (1.0, 1.0)), HalfSpace((0.0, 0.0), (-sin(0.1), cos(0.1))))))
    g = GeometricDomain(s)
    ParticleGrid(g, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1; randomness = (0.01, 0.01))
    t = @elapsed pg = ParticleGrid(g, (0.005, 0.005), ExponentialWeightFunction(1.0, 0.0125), 1; randomness = (0.001, 0.001))
    println("$nm h=1/200 jitter 0.2: ParticleGrid $(round(t, digits=2)) s, N = $(pg.meta.N)")
end
