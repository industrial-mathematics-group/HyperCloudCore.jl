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
function run(bc, N, tends, frac)
    h = 1 / N; Y0[] = frac * h
    shape = intersect(Tagged(Cuboid((0.0, 0.0), (1.0, 1.0); open = (true, false)), 4), Tagged(HalfSpace((0.0, frac * h), (0.0, -1.0)), 3))
    geom = GeometricDomain(shape; is_periodic = (true, false), mins = (0.0, 0.0), maxs = (1.0, 1.0), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0; out = Float64[]
    for tend in tends
        while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
        push!(out, maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg)))
    end
    out
end
for frac in (0.1, 0.5), (nm, bc) in (("Exact", ExactBC()), ("Mirror2 b", MirrorBC(; degree = 2)))
    for N in (40, 80, 160)
        e = run(bc, N, (0.25, 0.5, 1.0, 2.0), frac)
        @printf("frac %.2f %-10s N=%3d  max err at t = 0.25/0.5/1/2: %s\n", frac, nm, N, join([@sprintf("%.3e", x) for x in e], " "))
    end
    flush(stdout)
end
