# N2 follow-up: volumes and errors of the particles 3-12 h from a receding co-moving face (mid-channel)
include("common.jl")
uc(x, t) = 1 + 0.5 * cos(2π * (x[1] - t - 1)) * cos(2π * x[2])
for N in (80, 160)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
    geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => MirrorBC(; degree = 1)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(uc(pg.core.positions[i], 0.0)); end
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4h; t = 0.0
    while t < 0.5 - 1e-12; d = min(dt, 0.5 - t); ts(eq, pg, t, d); t += d; end
    face = 1 + t
    for (a, b) in ((0.5, 1.5), (1.5, 3), (3, 6), (6, 12), (12, 40))
        ids = [i for i in interior(pg) if 0.1 < pg.core.positions[i][2] < 0.4 && a * h <= face - pg.core.positions[i][1] < b * h]
        isempty(ids) && continue
        V = [pg.core.volumes[i] / h^2 for i in ids]
        e = [pg.rhos[i][1] - uc(pg.core.positions[i], t) for i in ids]
        @printf("N=%d  [%4.1f,%4.1f)h mid-channel: n=%4d V/h² in [%.3f, %.3f] mean %.3f; error in [%+.4f, %+.4f]\n", N, a, b, length(ids), extrema(V)..., sum(V) / length(V), extrema(e)...)
    end
    xs = sort(unique(round.([pg.core.positions[i][1] for i in interior(pg) if face - pg.core.positions[i][1] < 6h], digits = 6)))
    println("  columns within 6h of the face (x - face)/h: ", round.((xs .- face) ./ h, digits = 3))
end
