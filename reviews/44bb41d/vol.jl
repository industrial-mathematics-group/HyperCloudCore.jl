# Volumes: interior quadrature, ghost share, MLS clamps, Float32, discs/balls (run on mid = parent of #40 and new)
include("common.jl")
inner(pg) = interior(pg)
function quadstats(pg, f, If)
    I = inner(pg); G = ghosts(pg)
    q = sum(pg.core.volumes[i] * f(pg.core.positions[i]) for i in I)
    (rel = (q - If) / If, vint = sum(pg.core.volumes[i] for i in I), vgh = isempty(G) ? 0.0 : sum(pg.core.volumes[i] for i in G),
     nneg0 = count(i -> pg.core.volumes[i] <= 0, I), vmin = minimum(pg.core.volumes[i] for i in I) / prod(pg.meta.dx))
end
ests = (("Shepard", ShepardVolumes()), ("MLS0", MLSVolumes(0)), ("MLS1", MLSVolumes(1)), ("MLS2", MLSVolumes(2)))
println("== 1-D [-1,1], R = 2.5h: relative error of Σ_int V f for f = x^2 | exp(x); Σ_int V; Σ_ghost V")
for jit in (0.0, 0.3), (en, est) in ests
    rows = String[]
    for N in (40, 80, 160, 320)
        h = 2 / N
        pg = ParticleGrid(get_rectangular_domain(Float64, (-1.0,), (1.0,)), (h,), ExponentialWeightFunction(1.0, 2.5h), 1;
                          randomness = (jit * h,), rng = MersenneTwister(1))
        update_volumes!(pg, est)
        a = quadstats(pg, x -> x[1]^2, 2 / 3); b = quadstats(pg, x -> exp(x[1]), exp(1) - exp(-1))
        push!(rows, @sprintf("N=%d %.1e %.1e (Vint %.4f Vg %.3f neg %d)", N, abs(a.rel), abs(b.rel), a.vint, a.vgh, a.nneg0))
    end
    @printf("jit %.1f %-7s %s\n", jit, en, join(rows, " | "))
end
println("== 2-D unit box, R = 2.5h: relative error for f1 = exp(x)cos(y) (the test's) | f2 = exp(x+y) | f3 = x^2 + y")
F2 = ((x -> exp(x[1]) * cos(x[2]), (exp(1) - 1) * sin(1)), (x -> exp(x[1] + x[2]), (exp(1) - 1)^2), (x -> x[1]^2 + x[2], 1 / 3 + 1 / 2))
for jit in (0.0, 0.3), (en, est) in ests
    rows = String[]
    for N in (20, 40, 80)
        h = 1 / N
        pg = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1;
                          randomness = (jit * h, jit * h), rng = MersenneTwister(1))
        update_volumes!(pg, est)
        s = [quadstats(pg, f, If) for (f, If) in F2]
        push!(rows, @sprintf("N=%d %.1e %.1e %.1e (Vint %.4f Vg %.3f neg %d vmin %.2f)", N, abs(s[1].rel), abs(s[2].rel), abs(s[3].rel), s[1].vint, s[1].vgh, s[1].nneg0, s[1].vmin))
    end
    @printf("jit %.1f %-7s %s\n", jit, en, join(rows, " | "))
    flush(stdout)
end
println("== unit disc (2-D) and unit ball (3-D): relative error of Σ_int V (1 + r^2); Σ_int V / |Ω|")
for (D, Ns) in ((2, (20, 40, 80)), (3, (10, 20))), jit in (0.0, 0.3), (en, est) in ests
    D == 3 && en == "MLS2" && continue
    rows = String[]
    for n in Ns
        h = 2 / n
        geom = get_spherical_domain(Float64, ntuple(_ -> 0.0, D), 1.0)
        pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = ntuple(_ -> jit * h, D), rng = MersenneTwister(1))
        update_volumes!(pg, est)
        If = D == 2 ? π + π / 2 : 4π / 3 + 4π / 5
        s = quadstats(pg, x -> 1 + sum(abs2, x), If)
        push!(rows, @sprintf("n=%d %.1e (Vint/|Ω| %.4f Vg %.3f neg %d vmin %.2f)", n, abs(s.rel), s.vint / domain_volume(pg), s.vgh, s.nneg0, s.vmin))
    end
    @printf("%dD jit %.1f %-7s %s\n", D, jit, en, join(rows, " | "))
    flush(stdout)
end
println("== Float32: 2-D box N = 40 and disc n = 40")
for (en, est) in ests
    pg = ParticleGrid(get_rectangular_domain(Float32, (0f0, 0f0), (1f0, 1f0)), (1f0/40, 1f0/40), ExponentialWeightFunction(1f0, 2.5f0/40), 1)
    update_volumes!(pg, est)
    s = quadstats(pg, x -> exp(x[1] + x[2]), Float32((exp(1) - 1)^2))
    pd = ParticleGrid(get_spherical_domain(Float32, (0f0, 0f0), 1f0), (0.05f0, 0.05f0), ExponentialWeightFunction(1f0, 0.125f0), 1)
    update_volumes!(pd, est)
    sd = quadstats(pd, x -> 1 + sum(abs2, x), Float32(1.5π))
    @printf("%-7s box: rel %.2e Vint %.6f Vg %.2e eltype %s | disc rel %.2e Vint/|Ω| %.6f Vg %.2e\n", en, abs(s.rel), s.vint, s.vgh, eltype(pg.core.volumes), abs(sd.rel), sd.vint / domain_volume(pd), sd.vgh)
end
println("== total_mass vs interior_mass on a 2-D box, U = 1 + x")
pg = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(1 + pg.core.positions[i][1]); end
@printf("total %.6f interior %.6f (exact 1.5)\n", total_mass(pg)[1], interior_mass(pg)[1])
