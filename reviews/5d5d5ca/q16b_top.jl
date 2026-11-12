include("common.jl")
uc(x, t) = 1 + 0.5 * cos(2π * (x[1] - t - 1)) * cos(2π * x[2])
N = 40; h = 1 / N
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => MirrorBC(; degree = 1)))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(uc(pg.core.positions[i], 0.0)); end
rowstat(sel) = (vs = [pg.core.volumes[i] / h^2 for i in interior(pg) if sel(pg.core.positions[i])]; isempty(vs) ? "-" : @sprintf("n=%d V/h² in [%.3f, %.3f]", length(vs), extrema(vs)...))
println("t=0 wall row x<1: ", rowstat(x -> x[2] == 0 && 0.2 < x[1] < 1 - 3h), "; row 1: ", rowstat(x -> abs(x[2] - h) < 1e-12 && 0.2 < x[1] < 1 - 3h))
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
dt = 0.4h; t = 0.0
while t < 0.5 - 1e-12; global t; d = min(dt, 0.5 - t); ts(eq, pg, t, d); t += d; end
face = 1 + t
println("t=0.5 wall row x in (0.7, 1-3h): ", rowstat(x -> x[2] == 0 && 0.7 < x[1] < 1 - 3h), "; uncovered (1+3h, face-3h): ", rowstat(x -> x[2] == 0.5 && 1 + 3h < x[1] < face - 3h))
println("t=0.5 row 1 x in (0.7, 1-3h): ", rowstat(x -> abs(x[2] - h) < 1e-12 && 0.7 < x[1] < 1 - 3h), "; uncovered: ", rowstat(x -> abs(x[2] - h) < 1e-12 && 1 + 3h < x[1] < face - 3h))
for i in interior(pg)
    x = pg.core.positions[i]
    (x[2] == 0.5 && 1 + 3h < x[1] < face - 3h) || continue
    nb = pg.neighbor.ranges[i]
    @printf("  wall-row particle x=%.3f err %.3f V/h² %.3f orders %d neighbours %d (ghosts %d) normal %s φ %.3g\n", x[1], abs(pg.rhos[i][1] - uc(x, t)), pg.core.volumes[i] / h^2,
            pg.core.particle_orders[i], length(nb), count(k -> pg.core.is_boundary[pg.neighbor.indices[k]], nb), string(round.(pg.core.normals[i], digits = 3)), pg.core.wall_distances[i])
end
# the same quantities for an initial wall-row particle
for i in interior(pg)
    x = pg.core.positions[i]
    (x[2] == 0.5 && abs(x[1] - 0.8) < 1e-9) || continue
    nb = pg.neighbor.ranges[i]
    @printf("  initial wall-row particle x=%.3f err %.3f V/h² %.3f neighbours %d (ghosts %d) normal %s φ %.3g\n", x[1], abs(pg.rhos[i][1] - uc(x, t)), pg.core.volumes[i] / h^2,
            length(nb), count(k -> pg.core.is_boundary[pg.neighbor.indices[k]], nb), string(round.(pg.core.normals[i], digits = 3)), pg.core.wall_distances[i])
end
tops = [pg.core.positions[i] for i in 1:pg.meta.N if !pg.core.is_boundary[i] && pg.core.positions[i][2] > 0.5 - 0.6h && pg.core.positions[i][1] > 1]
println("interior particles within 0.6h of the top wall, x > 1: ", length(tops), " y values: ", unique(round.([p[2] for p in tops], digits = 6)))
gt = [pg.core.positions[i] for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 4 && pg.core.positions[i][1] > 1.05]
println("top-wall ghosts x > 1.05: ", length(gt), " y values ", unique(round.([p[2] for p in gt], digits = 4)))
gb = [pg.core.positions[i] for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 3 && pg.core.positions[i][1] > 1.05]
println("bottom-wall ghosts x > 1.05: ", length(gb), " y values ", unique(round.([p[2] for p in gb], digits = 4)))
cd = pg.domain
println("canvas y range ", cd.canvas_mins[2], " .. ", cd.canvas_maxs[2], "; wall index lo/hi: ", H._wall_index(0.0, cd.canvas_mins[2], h), " ", H._wall_index(0.5, cd.canvas_mins[2], h))
