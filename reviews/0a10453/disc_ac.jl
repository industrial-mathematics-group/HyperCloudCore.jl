# Acoustics (p, u, v) in the unit disc, slip wall: p = J0(k r) cos(k t), u = J1(k r) sin(k t) x/r, k = j_{1,1}.
# MirrorBC with the normal-velocity reflection vs exact ghost values (the interior scheme alone), old vs new grids.
include("common.jl")
const kB = 3.8317059702075125
function besselJ(nu, x)
    s = 0.0; t = (x / 2)^nu / factorial(nu)
    for m in 0:40
        s += t; t *= -(x / 2)^2 / ((m + 1) * (m + 1 + nu))
    end
    s
end
function exact(x, t)
    r = sqrt(x[1]^2 + x[2]^2)
    p = besselJ(0, kB * r) * cos(kB * t)
    a = r < 1e-14 ? 0.0 : besselJ(1, kB * r) * sin(kB * t) / r
    State{3, Float64}(p, a * x[1], a * x[2])
end
A = SMatrix{3,3,Float64}(0,1,0, 1,0,0, 0,0,0); B = SMatrix{3,3,Float64}(0,0,1, 0,0,0, 1,0,0)
eqa = LinearAdvection(SVector(A, B), Float64)
struct Slip end
(::Slip)(eq, U, w) = (n = w.normal; un = U[2] * n[1] + U[3] * n[2]; State{3, Float64}(U[1], U[2] - 2un * n[1], U[3] - 2un * n[2]))
struct ExactBC <: HyperCloudCore.AbstractBoundaryCondition end
function (::ExactBC)(pg, buf, tag, ts, eq, t)
    for i in 1:pg.meta.N
        (pg.core.is_boundary[i] && pg.core.tags[i] == tag) && (buf[i] = exact(pg.core.positions[i], t))
    end
end
function run(bc, n, jit, tend; order = 3)
    h = 2 / n
    geom = GeometricDomain(Ball((0.0, 0.0), 1.0); bc_map = BCD(1 => bc))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 3; randomness = (jit * h, jit * h), rng = MersenneTwister(5))
    for i in 1:pg.meta.N; pg.rhos[i] = exact(pg.core.positions[i], 0.0); end
    interp = MUSCL(Float64, 2, 3, order, NoLimiter(), RusanovFlux())
    ts = GeneralRKTimeStepper(eqa, interp, MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.5 * get_time_step(pg, eqa, interp); t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eqa, pg, t, d); t += d; end
    maximum(abs(pg.rhos[i][1] - exact(pg.core.positions[i], t)[1]) for i in interior(pg))
end
for jit in (0.0, 0.3), tend in (1.0, 3.0)
    println("== disc acoustics, jitter $jit, t = $tend, MUSCL3: max |p - p_exact| over interior, n = 20/40/80 per diameter")
    for (nm, bc) in (("Exact", ExactBC()), ("Mirror1 slip", MirrorBC(Slip(); degree = 1)), ("Mirror2 slip", MirrorBC(Slip(); degree = 2)))
        e = [run(bc, n, jit, tend) for n in (20, 40, 80)]
        @printf("  %-13s %s  EOC %.2f %.2f\n", nm, join([@sprintf("%.2e", x) for x in e], " "), log2(e[1] / e[2]), log2(e[2] / e[3]))
    end
    flush(stdout)
end
