# Builds many rectangular/spherical grids and writes a fingerprint line per grid (positions, tags,
# is_boundary, volumes, neighbour counts). Run on 5409209 and on 07db492/f6bdc2f and diff the outputs.
using HyperCloudCore, StaticArrays, Random
out = ARGS[1]
io = open(out, "w")
fp(v) = string(hash(collect(v)), base = 16)
function dump(id, pg)
    N = pg.meta.N
    pos = view(pg.core.positions, 1:N)
    println(io, id, " N=", N, " pos=", fp(pos), " tag=", fp(view(pg.core.tags, 1:N)),
            " bnd=", fp(view(pg.core.is_boundary, 1:N)), " vol=", fp(view(pg.core.volumes, 1:N)),
            " ntags=", sort(unique(view(pg.core.tags, 1:N))))
end
global count = 0
for T in (Float64, Float32)
    boxes = Dict(1 => [((0.0,), (1.0,)), ((-1.0,), (1.0,)), ((2.3,), (5.1,)), ((2.5,), (2.75,))],
                 2 => [((0.0, 0.0), (1.0, 1.0)), ((-1.0, -1.0), (1.0, 1.0)), ((2.3, -7.0), (5.1, -3.2)), ((0.0, 0.0), (3.0, 0.5))],
                 3 => [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), ((0.0, -1.0, 2.0), (1.0, 0.5, 2.6))])
    for D in (1, 2, 3)
        pers = D == 1 ? [(false,), (true,)] : D == 2 ? [(false, false), (true, false), (false, true), (true, true)] :
               [(false, false, false), (false, false, true), (true, false, false), (false, true, true)]
        ns = D == 1 ? (10, 30, 47) : D == 2 ? (12, 30) : (8, 12)
        for (bi, (mn, mx)) in enumerate(boxes[D]), per in pers, n in ns, aniso in (false, true), jit in (0.0, 0.2, 0.45), Rf in (2.5, 5.5)
            D == 3 && (Rf == 5.5 && n == 12) && continue
            L = mx .- mn
            dx = ntuple(d -> T(L[d] / (aniso && d == D ? 2n : n)), D)
            hmax = maximum(dx)
            geom = get_rectangular_domain(T, T.(mn), T.(mx); is_periodic = per)
            pg = ParticleGrid(geom, dx, ExponentialWeightFunction(one(T), T(Rf) * hmax), 1;
                              randomness = ntuple(d -> T(jit) * dx[d], D), rng = MersenneTwister(17))
            dump("rect T=$T D=$D box=$bi per=$per n=$n an=$aniso j=$jit R=$Rf", pg)
            global count += 1
        end
        D == 1 && continue
        for (c, r) in ((ntuple(_ -> 0.0, D), 1.0), (ntuple(d -> 0.37 * d, D), 0.8), (ntuple(_ -> 0.5, D), 0.5)),
            n in (D == 2 ? (20, 33) : (10, 14)), jit in (0.0, 0.3), Rf in (2.5, 4.0)
            dx = ntuple(_ -> T(2r / n), D)
            geom = get_spherical_domain(T, T.(c), T(r))
            pg = ParticleGrid(geom, dx, ExponentialWeightFunction(one(T), T(Rf) * dx[1]), 1;
                              randomness = ntuple(d -> T(jit) * dx[d], D), rng = MersenneTwister(5))
            dump("sph T=$T D=$D c=$c r=$r n=$n j=$jit R=$Rf", pg)
            global count += 1
        end
    end
end
close(io)
println("grids: ", count)
