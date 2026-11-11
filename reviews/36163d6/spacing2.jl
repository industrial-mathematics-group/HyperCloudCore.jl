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
for T in (Float64, Float32), jit in (0.0, 0.3)
    g = get_spherical_domain(T, (T(0), T(0)), T(1)); h = T(0.05)
    report("F=$T disc helper h=0.05 j=$jit", ParticleGrid(g, (h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
    g = get_spherical_domain(T, (T(0), T(0), T(0)), T(1)); h = T(0.1)
    report("F=$T ball 3-D h=0.1 j=$jit", ParticleGrid(g, (h, h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
    g = get_rectangular_domain(T, (T(0),), (T(1.03),)); h = T(0.05)
    report("F=$T 1-D 1.03 j=$jit", ParticleGrid(g, (h,), wf(T, 2.5h), 1; randomness = (T(jit) * h,), rng = MersenneTwister(3)), h)
    g = get_rectangular_domain(T, (T(0), T(0), T(0)), (T(1.03), T(1.01), T(0.98))); h = T(0.1)
    report("F=$T box 3-D off-lattice j=$jit", ParticleGrid(g, (h, h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
    sh = Cuboid((T(0), T(0)), (T(2), T(1))) - Tagged(Ball((T(0.7), T(0.5)), T(0.23)), 5); h = T(0.05)
    report("F=$T box minus disc j=$jit", ParticleGrid(GeometricDomain(sh), (h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
    sh = Cuboid((T(0), T(0), T(0)), (T(2), T(1), T(1))) - Tagged(Ball((T(0.7), T(0.5), T(0.5)), T(0.23)), 5); h = T(0.1)
    report("F=$T 3-D box minus ball j=$jit", ParticleGrid(GeometricDomain(sh), (h, h, h), wf(T, 2.5h), 1; randomness = (T(jit) * h, T(jit) * h, T(jit) * h), rng = MersenneTwister(3)), h)
end
let h = (0.1, 0.025)
    g = get_spherical_domain(Float64, (0.0, 0.0), 1.0)
    report("disc anisotropic 0.1x0.025 (h=min)", ParticleGrid(g, h, wf(Float64, 0.25), 1), 0.025)
end
# union of discs j = 0.45: who is the closest pair? and on-wall pairs only
let h = 0.04, s = union(Ball((0.0, 0.0), 0.5), Ball((0.6, 0.1), 0.4))
    pg = ParticleGrid(GeometricDomain(s), (h, h), wf(Float64, 2.5h), 1; randomness = (0.45h, 0.45h), rng = MersenneTwister(3))
    N = pg.meta.N; P = [Vector(pg.core.positions[i]) for i in 1:N]; wd = pg.core.wall_distances
    d, i, j = cellpairs(P, h)
    println("closest pair ", P[i], " ", P[j], " d/h ", d / h, " φ/h ", wd[i] / h, " ", wd[j] / h, " isb ", pg.core.is_boundary[i], pg.core.is_boundary[j])
    W = [k for k in 1:N if !pg.core.is_boundary[k] && abs(wd[k]) < 1e-9]
    dw, a, b = cellpairs(P[W], h); println("min spacing among exact on-wall points ", dw / h)
end
