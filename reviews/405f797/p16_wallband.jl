include("common.jl")
N = 40; h = 1/N
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
box = Cuboid((0.0, 0.0), (2.0, 0.5))
geom = GeometricDomain(intersect(box, piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => ExtrapolationBC(1)))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
g3 = [pg.core.positions[i][1] for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 3]
println("bottom-wall ghosts (tag 3) at t = 0: x in [", minimum(g3), ", ", maximum(g3), "], count ", length(g3))
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
t = 0.0; dt = 0.4h
for _ in 1:round(Int, 0.5/dt); global t; ts(eq, pg, t, dt); t += dt; end
g3 = [pg.core.positions[i][1] for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 3]
inner = interior(pg)
P = [pg.core.positions[i] for i in inner]
println("t = $t, face at ", 1 + t, ": bottom-wall ghosts x in [", minimum(g3), ", ", maximum(g3), "]; interior particles with x > ", maximum(g3), ": ", count(p -> p[1] > maximum(g3), P))
println("interior particles on the walls (y = 0 or 0.5) with x > 1: ", count(p -> p[1] > 1 + 1e-9 && (abs(p[2]) < 1e-9 || abs(p[2] - 0.5) < 1e-9), P),
        "; initial lattice had wall-row particles at x <= 1: ", count(p -> p[1] <= 1 && (abs(p[2]) < 1e-9 || abs(p[2] - 0.5) < 1e-9), P))
println("min y of interior particles with x > 1.1: ", minimum(p[2] for p in P if p[1] > 1.1), "; max y: ", maximum(p[2] for p in P if p[1] > 1.1))
# neighbours of a near-wall particle at the far end: how many ghosts below it?
i = inner[argmax([p[1] - 10abs(p[2]) for p in P])]
println("particle ", pg.core.positions[i], " has ", count(k -> pg.core.is_boundary[pg.neighbor.indices[k]], pg.neighbor.ranges[i]), " ghost neighbours of ", length(pg.neighbor.ranges[i]))
