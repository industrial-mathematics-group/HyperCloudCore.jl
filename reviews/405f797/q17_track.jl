include("common.jl")
uc(x, t) = 1 + 0.5 * cos(2π * (x[1] - t - 1)) * cos(2π * x[2])
N = 40; h = 1 / N
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => MirrorBC(; degree = 1)))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(uc(pg.core.positions[i], 0.0)); end
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
dt = 0.4h; t = 0.0
targets = (SVector(1.2, 0.5), SVector(1.2, 0.0), SVector(1.2, 0.475), SVector(1.2, 0.025))
find(x) = findfirst(i -> !pg.core.is_boundary[i] && sum(abs2, pg.core.positions[i] - x) < 1e-12, 1:pg.meta.N)
while t < 0.5 - 1e-12
    global t
    d = min(dt, 0.5 - t); ts(eq, pg, t, d); t += d
    line = ""
    for x in targets
        i = find(x)
        line *= i === nothing ? "      -       " : @sprintf("  %+.4f (V %.2f)", pg.rhos[i][1] - uc(x, t), pg.core.volumes[i] / h^2)
    end
    any(x -> find(x) !== nothing, targets) && @printf("t=%.3f face %.3f:%s\n", t, 1 + t, line)
end
