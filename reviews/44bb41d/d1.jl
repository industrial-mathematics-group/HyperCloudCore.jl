# D1 rerun: channel along a wall (periodic in x, walls at y = frac*h and y = 1), Gaussian centred on the wall,
# exact ghost values vs MirrorBC vs OutflowBC; max interior error at t = 0.25/0.5/1, location of the max, error away from the wall
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
function run(bc, N, tends, frac, order, Rf)
    h = 1 / N; Y0[] = frac * h
    shape = intersect(Tagged(Cuboid((0.0, 0.0), (1.0, 1.0); open = (true, false)), 4), Tagged(HalfSpace((0.0, frac * h), (0.0, -1.0)), 3))
    geom = GeometricDomain(shape; is_periodic = (true, false), mins = (0.0, 0.0), maxs = (1.0, 1.0), bc_map = BCD(3 => bc, 4 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, Rf * h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
    interp = MUSCL(Float64, 2, 1, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqx, interp, MOOD(), order == 3 ? RK3_SSP_Tableau(Float64) : RK4_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0; out = []
    I = interior(pg)
    for tend in tends
        while t < tend - 1e-12; d = min(dt, tend - t); ts(eqx, pg, t, d); t += d; end
        e = [abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in I]
        k = argmax(e)
        far = maximum(e[m] for m in eachindex(I) if pg.core.positions[I[m]][2] - Y0[] > 5h && pg.core.positions[I[m]][2] < 0.5)
        push!(out, (e[k], (pg.core.positions[I[k]][2] - Y0[]) / h, far))
    end
    out
end
order = parse(Int, ARGS[1]); fracs = parse.(Float64, split(ARGS[2], ",")); Ns = parse.(Int, split(ARGS[3], ","))
Rf = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.5
bcs = (("Exact", ExactBC()), ("Mirror2", MirrorBC(; degree = 2)), ("Outflow", OutflowBC()))
println("== MUSCL$order, R = $(Rf)h; per N: max err at t = 0.25/0.5/1 [ (y-wall)/h of the max at t=1 | max err for y-wall > 5h at t = 1 ]")
for frac in fracs, (nm, bc) in bcs
    for N in Ns
        r = run(bc, N, (0.25, 0.5, 1.0), frac, order, Rf)
        @printf("frac %.2f %-8s N=%3d  %s  [ %.2f | %.3e ]\n", frac, nm, N, join([@sprintf("%.3e", x[1]) for x in r], " "), r[end][2], r[end][3])
        flush(stdout)
    end
end
