# MOODu2 next to a wall: the ghosts' curvatures enter the u2 plateau/ratio test (zero before #39).
# Channel (lattice wall), Gaussian centred on the wall, MUSCL3; count MOOD drops per step (rows 0-2 vs the rest) and the error at t = 0.5
include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
struct ExactBC <: HyperCloudCore.AbstractBoundaryCondition end
function (::ExactBC)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = State{1,Float64}(u0(pg.core.positions[i], t)))
    end
end
function run(bc, N, crit; tend = 0.5)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(crit), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    near = 0; far = 0; steps = 0
    while t < tend - 1e-12
        d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; steps += 1
        for i in interior(pg)
            ts.work.triggered_step[i] || continue
            pg.core.positions[i][2] < 2.5h ? (near += 1) : (far += 1)
        end
    end
    e = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg))
    e, near / steps, far / steps
end
for (cn, crit) in (("u1 1e-3", MOODu1(1e-3)), ("u2 1e-3", MOODu2(1e-3)), ("u2 1e-4", MOODu2(1e-4))), (bn, bc) in (("Exact", ExactBC()), ("Mirror2", MirrorBC(; degree = 2)))
    r = [run(bc, N, crit) for N in (40, 80)]
    @printf("%-8s %-8s  N=40: err %.3e drops/step near %.1f far %.1f | N=80: err %.3e near %.1f far %.1f\n", cn, bn, r[1]..., r[2]...)
    flush(stdout)
end
