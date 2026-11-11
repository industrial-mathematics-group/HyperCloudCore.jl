# 1-D spherical domains: old test (x-c)^2 <= r^2 vs new |x-c| - r <= 0 on the generated lattice, incl. the
# HyperCloud sims.jl case (:spherical, mins 0.1, maxs 1.0, Ns 250).
using HyperCloudCore, StaticArrays, Random
tot = 0; nflip = 0
for T in (Float64, Float32), (lo, hi) in ((0.1, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.3, 0.7), (2.3, 5.1)), n in (10, 20, 30, 47, 100, 250, 1000), Rf in (2.5, 5.5)
    c = T((lo + hi) / 2); r = T((hi - lo) / 2)
    dx = (T((hi - lo) / n),)
    geom = get_spherical_domain(T, (c,), r)
    cd = HyperCloudCore.ComputationalDomain(geom, dx, Rf)
    allg = HyperCloudCore.GeometricDomain(geom.label, geom.mins, geom.maxs, geom.is_periodic, x -> true, x -> 0, geom.bc_map)
    pos, = HyperCloudCore.get_points(cd, allg; nominal_dx = dx, interp_range_factor = Rf)
    flips = filter(x -> ((x[1] - c)^2 <= r^2) != geom.is_interior(x), pos)
    global tot += 1
    isempty(flips) && continue
    global nflip += 1
    println("T=$T [$lo,$hi] n=$n R=$Rf: ", [(x[1], geom.is_interior(x) ? "new in" : "new out", abs(big(x[1]) - big(c)) <= big(r) ? "exact in" : "exact out") for x in flips])
end
println("$nflip of $tot cases differ")
