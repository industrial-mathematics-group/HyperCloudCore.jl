# _sweep_lattice in Float32 and Float64: does the whole canvas box give every lattice row of the generated cloud
# (canvas ends included)? Compares the x and y coordinates the sweep lattice produces with get_points' lattice.
include("common.jl")
for T in (Float32, Float64), N in (20, 26, 52, 100, 160, 400, 1000)
    h = T(1) / N
    piston = Moving(Tagged(HalfSpace((T(1), T(0)), (T(1), T(0))), 5), translating((T(1), T(0))))
    geom = GeometricDomain(intersect(Cuboid((T(0), T(0)), (T(2), T(0.5))), piston); mins = (T(0), T(0)), maxs = (T(2), T(0.5)),
                           bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(T(1), T(2.5) * h), 1)
    cd = pg.domain; dx = pg.meta.dx
    ny_gen = round(Int, (cd.canvas_maxs[2] - cd.canvas_mins[2]) / dx[2]) + 1
    nx_gen = round(Int, (cd.canvas_maxs[1] - cd.canvas_mins[1]) / dx[1]) + 1
    ys = Set{T}(); xs = Set{T}()
    H._sweep_lattice(pg, geom, (cd.canvas_mins, cd.canvas_maxs)) do x
        push!(xs, x[1]); push!(ys, x[2])
    end
    last_y = cd.canvas_mins[2] + (ny_gen - 1) * dx[2]
    @printf("%s N=%4d: rows y %d of %d, columns x %d of %d; last generated row y = %.9g, canvas max %.9g\n", T, N, length(ys), ny_gen, length(xs), nx_gen, last_y, cd.canvas_maxs[2])
end
