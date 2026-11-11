# The wall 0.5h above a lattice row (not snapped): where is the error, and does the bound matter?
include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
const Y0 = Ref(0.0)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + (x[2] - Y0[])^2) / 0.02)
function run(bc, N, tend, frac; order = 3)
    h = 1 / N; Y0[] = frac * h
    shape = intersect(Tagged(Cuboid((0.0, 0.0), (1.0, 1.0); open = (true, false)), 4), Tagged(HalfSpace((0.0, frac * h), (0.0, -1.0)), 3))
    geom = GeometricDomain(shape; is_periodic = (true, false), mins = (0.0, 0.0), maxs = (1.0, 1.0), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    # static ghost error vs the even extension
    buf = copy(pg.rhos); for (tag, b) in pg.geometry.bc_map; b(pg, buf, tag, nothing, eqx, 0.0); end
    eg = maximum(abs(buf[i][1] - u0(mirrorpt(pg, i), 0.0)) for i in ghosts(pg))
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
    I = interior(pg); e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in I]
    k = argmax(e)
    e[k], eg, (pg.core.positions[I[k]][2] - Y0[]) / h, pg.core.positions[I[k]][1]
end
for frac in (0.5, 0.1), (nm, bc) in (("Mirror2 b", MirrorBC(; degree = 2)), ("Mirror2 unb", MirrorBC(; degree = 2, bounded = false)), ("Outflow", OutflowBC()))
    for N in (40, 80, 160)
        e, eg, yh, x = run(bc, N, 1.0, frac)
        @printf("frac %.2f %-12s N=%3d  max err %.3e at (y-wall)/h = %.2f, x = %.3f   static ghost err %.2e\n", frac, nm, N, e, yh, x, eg)
    end
    flush(stdout)
end
