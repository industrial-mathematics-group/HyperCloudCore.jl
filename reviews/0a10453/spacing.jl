# Minimum spacing (all pairs, interior-interior, and pairs involving a point on the wall) and minimum ghost φ,
# over many shapes, resolutions, jitters, 1-3-D, Float32.
using HyperCloudCore, StaticArrays, Random, LinearAlgebra, Printf
wf(T, R) = ExponentialWeightFunction(one(T), T(R))
function cellpairs(P, r)
    # brute force with a hash grid (radius r)
    best = (Inf, 0, 0)
    D = length(P[1]); key(p) = ntuple(d -> floor(Int, p[d] / r), D)
    H = Dict{Any, Vector{Int}}()
    for (i, p) in enumerate(P); push!(get!(H, key(p), Int[]), i); end
    for (i, p) in enumerate(P), off in Iterators.product(ntuple(_ -> -1:1, D)...)
        for j in get(H, key(p) .+ off, Int[])
            j > i || continue
            d = norm(P[i] - P[j]); d < best[1] && (best = (d, i, j))
        end
    end
    best
end
function report(name, pg, h)
    N = pg.meta.N
    P = [Vector{Float64}(pg.core.positions[i]) for i in 1:N]
    isb = [pg.core.is_boundary[i] for i in 1:N]; wd = [Float64(pg.core.wall_distances[i]) for i in 1:N]
    d, i, j = cellpairs(P, h)
    Ii = findall(!, isb); di, a, b = cellpairs(P[Ii], h)
    onw = [k for k in Ii if abs(wd[k]) < 1e-6 * h]
    mg = minimum(wd[isb]; init = Inf)
    @printf("%-46s Nint %6d  min all %.3fh  min int %.3fh (φ %.2f/%.2f h)  min ghost φ %.3fh  on wall %d\n", name, length(Ii), d / h, di / h,
            wd[Ii[a]] / h, wd[Ii[b]] / h, mg / h, length(onw))
end
const Ts = (Float64, Float32)
for T in Ts, jit in (0.0, 0.2, 0.3, 0.45)
    for (nm, shape, h) in (
        ("disc r=1 h=0.1", Ball((0.0, 0.0), 1.0), 0.1), ("disc r=1 h=0.037", Ball((0.0, 0.0), 1.0), 0.037),
        ("disc r=0.13 h=0.05 (small)", Ball((0.01, 0.02), 0.13), 0.05),
        ("box 1.03", Cuboid((0.0, 0.0), (1.03, 1.0)), 0.05),
        ("box minus disc", Cuboid((0.0, 0.0), (2.0, 1.0)) - Tagged(Ball((0.7, 0.5), 0.23), 5), 0.05),
        ("step 0.61/0.213", Cuboid((0.0, 0.0), (3.0, 1.0)) - Tagged(Cuboid((0.61, -1.0), (4.0, 0.213)), 5), 0.05),
        ("wedge 30", intersect(Cuboid((-1.0, 0.0), (1.0, 1.0)), HalfSpace((0.0, 0.0), (-sin(pi/6), cos(pi/6)))), 0.05),
        ("rot plate", Cuboid((-1.0, -0.5), (1.0, 0.5)) - Tagged(Rotated(Cuboid((-0.2, -0.02), (0.2, 0.02)), π / 6), 5), 0.025),
        ("L notch (reflex)", Cuboid((0.0, 0.0), (1.0, 1.0)) - Cuboid((0.503, 0.503), (2.0, 2.0)), 0.05),
        ("union discs", union(Ball((0.0, 0.0), 0.5), Ball((0.6, 0.1), 0.4)), 0.04),
        )
        Tf = T
        s = shape
        if T == Float32
            s = HyperCloudCore.Shapes === nothing ? s : s
        end
        try
            g = GeometricDomain(s)
            T == Float32 && continue
            pg = ParticleGrid(g, (h, h), wf(T, 2.5h), 1; randomness = (jit * h, jit * h), rng = MersenneTwister(3))
            report("$nm j=$jit", pg, h)
        catch e
            println(nm, " ", sprint(showerror, e)[1:min(end, 200)])
        end
    end
end
# Float32 + 1-D + 3-D via the domain helpers
for T in (Float64, Float32), jit in (0.0, 0.3)
    g = get_spherical_domain(T, (T(0), T(0)), T(1)); h = T(0.05)
    report("F=$T disc helper h=0.05 j=$jit", ParticleGrid(g, (h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
    g = get_spherical_domain(T, (T(0), T(0), T(0)), T(1)); h = T(0.1)
    report("F=$T ball 3-D h=0.1 j=$jit", ParticleGrid(g, (h, h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
    g = get_rectangular_domain(T, (T(0),), (T(1.03),)); h = T(0.05)
    report("F=$T 1-D 1.03 j=$jit", ParticleGrid(g, (h,), wf(T, 2.5h), 1; randomness = (T(jit) * h,), rng = MersenneTwister(3)), h)
    g = get_rectangular_domain(T, (T(0), T(0), T(0)), (T(1.03), T(1.01), T(0.98))); h = T(0.1)
    report("F=$T box 3-D off-lattice j=$jit", ParticleGrid(g, (h, h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
end
# anisotropic
let h = (0.1, 0.025)
    g = get_spherical_domain(Float64, (0.0, 0.0), 1.0)
    report("disc anisotropic 0.1x0.025 (h=min)", ParticleGrid(g, h, wf(Float64, 0.25), 1), 0.025)
end
