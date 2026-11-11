# Property: every interior particle on the wall (snapped or lattice) keeps >= h/4 (minimum image) from every
# other interior particle; ghost phi >= h/2. Many shapes, jitters, seeds, F32/F64, periodic seams.
using HyperCloudCore, StaticArrays, Random, LinearAlgebra, Printf
wf(T, R) = ExponentialWeightFunction(one(T), T(R))
function check(name, pg, h, per, L)
    N = pg.meta.N; P = pg.core.positions; isb = pg.core.is_boundary; wd = pg.core.wall_distances
    I = [i for i in 1:N if !isb[i]]
    onw = [i for i in I if abs(wd[i]) <= 1e-5 * h]
    md(a, b) = sqrt(sum(d -> (δ = Float64(a[d] - b[d]); per[d] && (δ -= L[d] * round(δ / L[d])); δ^2), 1:length(a)))
    best = Inf; seam = 0
    for i in onw, j in I
        j == i && continue
        d = md(P[i], P[j]); d < best && (best = d)
        if d < 0.25h && norm(Float64.(P[i] - P[j])) > 0.5h; seam += 1; end
    end
    mg = minimum((wd[i] for i in 1:N if isb[i]); init = Inf)
    @printf("%-52s onwall %5d  min onwall-int %.3fh  close-across-seam %d  min ghost φ %.3fh\n", name, length(onw), best / h, seam, mg / h)
    return best / h
end
worst = Inf
for T in (Float64, Float32), seed in 1:3, jit in (0.0, 0.2, 0.3, 0.45)
    cases = Any[
        ("disc", GeometricDomain(Ball((T(0), T(0)), T(1))), T(0.037), (false, false)),
        ("union discs", GeometricDomain(union(Ball((T(0), T(0)), T(0.5)), Ball((T(0.6), T(0.1)), T(0.4)))), T(0.04), (false, false)),
        ("wedge 30", GeometricDomain(intersect(Cuboid((T(-1), T(0)), (T(1), T(1))), HalfSpace((T(0), T(0)), (T(-sin(pi/6)), T(cos(pi/6)))))), T(0.05), (false, false)),
        ("box minus disc", GeometricDomain(Cuboid((T(0), T(0)), (T(2), T(1))) - Tagged(Ball((T(0.7), T(0.5)), T(0.23)), 5)), T(0.05), (false, false)),
        # periodic in x, walls off the lattice, L/cell not an integer, obstacle crossing the seam
        ("chan per-x 0.013..0.98, seam obstacle", GeometricDomain(intersect(HalfSpace((T(0), T(0.013)), (T(0), T(-1))), HalfSpace((T(0), T(0.98)), (T(0), T(1)))) -
            union(Ball((T(0), T(0.5)), T(0.2)), Ball((T(1), T(0.5)), T(0.2))); is_periodic = (true, false), mins = (T(0), T(0.013)), maxs = (T(1), T(0.98))), T(1/27), (true, false)),
        ("box 3-D 1.03x1.01x0.98", get_rectangular_domain(T, (T(0), T(0), T(0)), (T(1.03), T(1.01), T(0.98))), T(0.1), (false, false, false)),
        ("box 3-D per-x 1.03x1.01x0.98", get_rectangular_domain(T, (T(0), T(0), T(0)), (T(1.03), T(1.01), T(0.98)); is_periodic = (true, false, false)), T(0.1), (true, false, false)),
        ("ball 3-D", get_spherical_domain(T, (T(0), T(0), T(0)), T(1)), T(0.1), (false, false, false)),
    ]
    for (nm, g, h, per) in cases
        D = length(per)
        L = Float64.(g.maxs - g.mins)
        pg = ParticleGrid(g, ntuple(_ -> h, D), wf(T, 2.5h), 1; randomness = ntuple(_ -> T(jit) * h, D), rng = MersenneTwister(seed))
        global worst = min(worst, check("$T s$seed j=$jit $nm", pg, h, per, L))
    end
end
println("WORST ", worst)
