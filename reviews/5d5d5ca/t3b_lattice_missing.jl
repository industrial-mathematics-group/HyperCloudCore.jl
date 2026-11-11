# Which lattice coordinates does _sweep_lattice miss, and does the generated cloud have particles there?
include("common.jl")
for (T, N) in ((Float32, 52), (Float32, 100), (Float32, 26), (Float64, 20), (Float64, 400), (Float64, 52))
    h = T(1) / N
    piston = Moving(Tagged(HalfSpace((T(1), T(0)), (T(1), T(0))), 5), translating((T(1), T(0))))
    geom = GeometricDomain(intersect(Cuboid((T(0), T(0)), (T(2), T(0.5))), piston); mins = (T(0), T(0)), maxs = (T(2), T(0.5)),
                           bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(T(1), T(2.5) * h), 1)
    cd = pg.domain
    ys = Set{T}(); xs = Set{T}()
    H._sweep_lattice(pg, geom, (cd.canvas_mins, cd.canvas_maxs)) do x; push!(xs, x[1]); push!(ys, x[2]); end
    gx = Set(pg.core.positions[i][1] for i in 1:pg.meta.N); gy = Set(pg.core.positions[i][2] for i in 1:pg.meta.N)
    mx = setdiff(gx, xs); my = setdiff(gy, ys)
    cnt(d, v) = count(i -> pg.core.positions[i][d] == v, 1:pg.meta.N)
    @printf("%s N=%d canvas [%s, %s]: generated coords missing from the sweep lattice: x %s (particles %s), y %s (particles %s); canvas_max - last = %s\n",
            T, N, cd.canvas_mins, cd.canvas_maxs, collect(mx), [cnt(1, v) for v in mx], collect(my), [cnt(2, v) for v in my],
            [cd.canvas_maxs[d] - (cd.canvas_mins[d] + round(Int, (cd.canvas_maxs[d] - cd.canvas_mins[d]) / pg.meta.dx[d]) * pg.meta.dx[d]) for d in 1:2])
end
