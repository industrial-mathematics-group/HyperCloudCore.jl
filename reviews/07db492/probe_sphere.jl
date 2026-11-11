# Which lattice points of a spherical domain flip between the old test (|x-c|^2 <= r^2) and the new
# (sqrt(|x-c|^2) - r <= 0), and which side exact arithmetic puts them on.
using HyperCloudCore, StaticArrays, Random
for T in (Float64, Float32), D in (2, 3), (c, r) in ((ntuple(_ -> 0.0, D), 1.0), (ntuple(_ -> 0.5, D), 0.5)), n in (D == 2 ? (20, 33) : (10, 14)), Rf in (2.5, 4.0)
    dx = ntuple(_ -> T(2r / n), D)
    geom = get_spherical_domain(T, T.(c), T(r))
    cd = HyperCloudCore.ComputationalDomain(geom, dx, Rf)
    pos, = HyperCloudCore.get_points(cd, HyperCloudCore.GeometricDomain(geom.label, geom.mins, geom.maxs, geom.is_periodic, x -> true, x -> 0, geom.bc_map); nominal_dx = dx, interp_range_factor = Rf)
    cs = SVector{D, T}(T.(c)); rT = T(r)
    flips = filter(x -> (sum(abs2, x - cs) <= rT^2) != geom.is_interior(x), pos)
    isempty(flips) && continue
    exact_in = count(x -> sum(abs2, big.(x) - big.(cs)) <= big(rT)^2, flips)
    newin = count(geom.is_interior, flips)
    println("T=$T D=$D c=$(c[1]) r=$r n=$n: $(length(flips)) flips, new interior $newin, exactly inside $exact_in; e.g. ", flips[1], " |x-c|-r = ", Float64(sqrt(sum(abs2, big.(flips[1]) - big.(cs))) - big(rT)))
end
