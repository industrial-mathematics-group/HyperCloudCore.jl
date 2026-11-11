# get_tag / is_interior of rectangular and spherical domains on random, tie, wall and non-finite points;
# writes one line per domain with a hash, plus the mismatching points are found by diffing the full dumps.
using HyperCloudCore, StaticArrays, Random
io = open(ARGS[1], "w")
rng = MersenneTwister(3)
for T in (Float64, Float32), D in (1, 2, 3), per in (false, true, :first)
    isper = per === :first ? ntuple(d -> d == 1, D) : ntuple(_ -> per, D)
    mins = ntuple(d -> T(-0.5d), D); maxs = ntuple(d -> T(1 + 0.25d), D)
    geoms = Any[("rect", get_rectangular_domain(T, mins, maxs; is_periodic = isper))]
    per === false && push!(geoms, ("sph", get_spherical_domain(T, ntuple(d -> T(0.1d), D), T(0.9))))
    special = T[0, -0.5, -1, 1.25, 1.5, 1.75, 0.9, -0.8, 1.0, Inf, -Inf, floatmax(T), -floatmax(T), nextfloat(T(-0.5)), prevfloat(T(-0.5)), nextfloat(T(1.25)), prevfloat(T(1.25))]
    pts = SVector{D, T}[]
    for _ in 1:30000; push!(pts, SVector{D, T}(ntuple(_ -> T(5rand(rng) - 2.5), D))); end
    for _ in 1:20000; push!(pts, SVector{D, T}(ntuple(_ -> rand(rng, special), D))); end
    for _ in 1:10000                                   # diagonal ties around the corners
        a = T(rand(rng)); push!(pts, SVector{D, T}(ntuple(d -> isodd(d) ? mins[d] - a : maxs[d] + a, D)))
    end
    for (name, g) in geoms
        for p in pts
            println(io, name, " ", T, " ", D, " ", per, " ", p, " ", g.is_interior(p), " ", g.get_tag(p))
        end
    end
end
close(io)
