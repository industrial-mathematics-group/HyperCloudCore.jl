# N2 follow-up: where is the error of the uncovered part behind a receding co-moving piston along walls, and does it converge?
include("common.jl")
Ns = parse.(Int, split(get(ENV, "NS", "20,40,80,160"), ","))
uc(x, t) = 1 + 0.5 * cos(2π * (x[1] - t - 1)) * cos(2π * x[2])
for N in Ns
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
    geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => (get(ENV, "FACE", "mirror") == "extrap" ? ExtrapolationBC(1) : MirrorBC(; degree = 1))))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(uc(pg.core.positions[i], 0.0)); end
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4h; t = 0.0
    el = @elapsed while t < 0.5 - 1e-12; d = min(dt, 0.5 - t); ts(eq, pg, t, d); t += d; end
    face = 1 + t
    function band(a, b)
        ids = [i for i in interior(pg) if 1 + 3h < pg.core.positions[i][1] && a * h <= face - pg.core.positions[i][1] < b * h]
        isempty(ids) && return "-"
        e = [abs(pg.rhos[i][1] - uc(pg.core.positions[i], t)) for i in ids]
        k = argmax(e); x = pg.core.positions[ids[k]]
        @sprintf("%.4f @ (%.3f, %.3f)", e[k], x[1], x[2])
    end
    @printf("N=%3d: uncovered part, distance to the face [3,6)h: %s; [6,12)h: %s; >=12h: %s; (%.0f s)\n", N, band(3, 6), band(6, 12), band(12, 1e9), el); ids = [i for i in interior(pg) if pg.core.positions[i][1] > t + 0.2]; e = [abs(pg.rhos[i][1] - uc(pg.core.positions[i], t)) for i in ids]; k = argmax(e); @printf("       overall max %.4f at (%.4f, %.4f), face %.4f\n", e[k], pg.core.positions[ids[k]]..., face)
end
