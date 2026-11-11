# Reproduce the volumes.md table / docstrings: 1-D unit interval x^2, exp(x) at N = 40 (and more N, also [-1,1]);
# 2-D unit square exp(x+y) N = 20/40/80 for Shepard, MLS0-2; the new unit_schemes test checks; Float32 variant.
include("common.jl")
ests = (("Shepard", ShepardVolumes()), ("MLS0", MLSVolumes(0)), ("MLS1", MLSVolumes(1)), ("MLS2", MLSVolumes(2)))
rq(pg, f, If) = (sum(pg.core.volumes[i] * f(pg.core.positions[i]) for i in interior(pg)) - If) / If
println("== 1-D, R = 2.5h, relative error x^2 | exp(x)")
for (a, b) in ((0.0, 1.0), (-1.0, 1.0)), (en, est) in ests
    rows = String[]
    for N in (40, 80, 160)
        h = (b - a) / N
        pg = ParticleGrid(get_rectangular_domain(Float64, (a,), (b,)), (h,), ExponentialWeightFunction(1.0, 2.5h), 1)
        update_volumes!(pg, est)
        e1 = rq(pg, x -> x[1]^2, (b^3 - a^3) / 3); e2 = rq(pg, x -> exp(x[1]), exp(b) - exp(a))
        push!(rows, @sprintf("N=%d %.2e %.2e (rel/h %.2f %.2f)", N, abs(e1), abs(e2), abs(e1) / h, abs(e2) / h))
    end
    @printf("[%g,%g] %-7s %s\n", a, b, en, join(rows, " | "))
end
println("== 2-D unit square, exp(x+y), R = 2.5h")
for T in (Float64, Float32), (en, est) in ests
    rows = String[]
    for N in (20, 40, 80)
        h = T(1 / N)
        pg = ParticleGrid(get_rectangular_domain(T, (zero(T), zero(T)), (one(T), one(T))), (h, h), ExponentialWeightFunction(one(T), T(2.5) * h), 1)
        update_volumes!(pg, est)
        e = rq(pg, x -> exp(x[1] + x[2]), T((exp(1) - 1)^2))
        push!(rows, @sprintf("N=%d %.3e (rel/h %.2f)", N, abs(e), abs(e) / h))
    end
    @printf("%s %-7s %s\n", T, en, join(rows, " | "))
    flush(stdout)
end
