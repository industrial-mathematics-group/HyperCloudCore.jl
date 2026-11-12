# The new test of #39 (unit_core.jl, "Ghost gradients") on any tree
include("common.jl")
const BCD = Dict{Int, AbstractBoundaryCondition}
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
struct ExactGhosts <: AbstractBoundaryCondition end
function (::ExactGhosts)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = State{1, Float64}(u0(pg.core.positions[i], t)))
    end
end
function run_lattice(bc, N; tend = 0.5)
    h = 1 / N
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
    maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg))
end
ex40, ex80 = run_lattice(ExactGhosts(), 40), run_lattice(ExactGhosts(), 80)
m80 = run_lattice(MirrorBC(; degree = 2), 80)
@printf("exact 40 %.3e  80 %.3e  ratio %.3f (test < 0.25: %s)   mirror80 %.3e  ratio %.3f (test < 1.3: %s)\n",
        ex40, ex80, ex80 / ex40, ex80 < 0.25ex40, m80, m80 / ex80, m80 < 1.3ex80)
