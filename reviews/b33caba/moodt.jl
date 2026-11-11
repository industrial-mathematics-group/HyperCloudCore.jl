# The new MOODu2 wall test (unit_core.jl, run_lattice) verbatim plus variants: drops on both walls, MOODu1, N = 80
include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
struct ExactGhosts <: AbstractBoundaryCondition end
function (::ExactGhosts)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = State{1, Float64}(u0(pg.core.positions[i], t)))
    end
    return nothing
end
function run_lattice(bc, N; mood = MOOD(), drops = nothing, top = nothing, far = nothing)
    h    = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false),
                                  bc_map = Dict{Int, AbstractBoundaryCondition}(3 => bc, 4 => bc))
    pg   = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, mood, RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0; steps = 0
    while t < 0.5 - 1e-12
        d = min(dt, 0.5 - t); ts(eqx, pg, t, d); t += d; steps += 1
        drops === nothing || (drops[] += count(i -> ts.work.triggered_step[i] && pg.core.positions[i][2] < 2.5h, interior_indices(pg)))
        top === nothing || (top[] += count(i -> ts.work.triggered_step[i] && pg.core.positions[i][2] > 1 - 2.5h, interior_indices(pg)))
        far === nothing || (far[] += count(i -> ts.work.triggered_step[i] && 2.5h <= pg.core.positions[i][2] <= 1 - 2.5h, interior_indices(pg)))
    end
    return maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior_indices(pg)), steps
end
for (cn, crit) in (("u2 1e-4", MOODu2(1e-4)), ("u1 1e-4", MOODu1(1e-4)), ("u2 1e-5", MOODu2(1e-5))), (bn, bc) in (("Mirror2", MirrorBC(; degree = 2)), ("Exact", ExactGhosts())), N in (40, 80)
    (cn != "u2 1e-4" && N == 80) && continue
    nd, nt, nf = Ref(0), Ref(0), Ref(0)
    e, s = run_lattice(bc, N; mood = MOOD(crit), drops = nd, top = nt, far = nf)
    @printf("%-8s %-8s N=%d: err %.3e  drops bottom rows %d (%.2f/step) top %d far %d  steps %d  test(<0.1 && 0) %s\n", cn, bn, N, e, nd[], nd[] / s, nt[], nf[], s, e < 0.1 && nd[] == 0)
    flush(stdout)
end
