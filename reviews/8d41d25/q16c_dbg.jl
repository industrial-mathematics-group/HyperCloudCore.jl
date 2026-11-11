include("common.jl")
N = 40; h = 1 / N
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => MirrorBC(; degree = 1)))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
println("meta.dx = ", pg.meta.dx, " R = ", pg.meta.R / h, " h; canvas ", pg.domain.canvas_mins, " .. ", pg.domain.canvas_maxs)
println("t=0 top ghosts at x≈0.8: ", sort([round(pg.core.positions[i][2], digits = 5) for i in 1:pg.meta.N if pg.core.is_boundary[i] && abs(pg.core.positions[i][1] - 0.8) < 1e-9 && pg.core.positions[i][2] > 0.4]))
println("t=0 bottom ghosts at x≈0.8: ", sort([round(pg.core.positions[i][2], digits = 5) for i in 1:pg.meta.N if pg.core.is_boundary[i] && abs(pg.core.positions[i][1] - 0.8) < 1e-9 && pg.core.positions[i][2] < 0.1]))
t = 0.0
for _ in 1:20; global t; t += 0.4h; place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg); sweep_boundaries!(pg, 0.0, t, 3); end
xc = 1.1
println("t=$t, column x=$xc: ", sort([(round(pg.core.positions[i][2], digits = 5), pg.core.is_boundary[i], pg.core.tags[i]) for i in 1:pg.meta.N if abs(pg.core.positions[i][1] - xc) < 1e-9]))
H._sweep_lattice(pg, geom, (SVector(xc - 1e-9, 0.4), SVector(xc + 1e-9, 1.0))) do x
    println("  lattice point ", x)
end
idx, dv = Int[], SVector{2, Float64}[]
for y in (0.575, 0.5749999, 0.55, -0.075)
    n = H._collect_near!(pg, SVector(xc, y), (3.5h)^2, idx, dv)
    println("_collect_near!((", xc, ", ", y, "), 3.5h): ", n, " particles, interior ", count(k -> !pg.core.is_boundary[idx[k]], 1:n))
end
