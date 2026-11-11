# Channel along a wall that is NOT on the lattice: the bottom wall at y = δ (δ = frac*h above the lattice row y = 0),
# made by intersecting the periodic channel with a half space. Gaussian centred on the wall (even extension exact).
include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
const Y0 = Ref(0.0)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + (x[2] - Y0[])^2) / 0.02)
struct ExactBC <: HyperCloudCore.AbstractBoundaryCondition end
function (::ExactBC)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = State{1,Float64}(u0(pg.core.positions[i], t)))
    end
end
function run(bc, N, tend, jit, frac)
    h = 1 / N
    Y0[] = frac * h
    shape = intersect(Tagged(Cuboid((0.0, 0.0), (1.0, 1.0); open = (true, false)), 4), Tagged(HalfSpace((0.0, frac * h), (0.0, -1.0)), 3))
    geom = GeometricDomain(shape; is_periodic = (true, false), mins = (0.0, 0.0), maxs = (1.0, 1.0), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(7))
    gb = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 3]
    minφ = minimum(pg.core.wall_distances[i] for i in gb) / h
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
    e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg)]
    maximum(e), minφ
end
for frac in (0.1, 0.5), jit in (0.2,), tend in (0.25, 1.0)
    println("== wall at y = $(frac)h, jitter $jit, t = $tend: max err N = 20/40/80 (min ghost φ/h)")
    for (name, bc) in (("Exact", ExactBC()), ("Outflow", OutflowBC()), ("Mirror2 b", MirrorBC(; degree = 2)))
        r = [run(bc, N, tend, jit, frac) for N in (20, 40, 80)]
        @printf("  %-10s %s   (%s)\n", name, join([@sprintf("%.2e", x[1]) for x in r], " "), join([@sprintf("%.3f", x[2]) for x in r], " "))
    end
end
