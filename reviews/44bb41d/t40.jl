# The new/changed volume tests of #40 (unit_schemes.jl) as printed checks, on any tree
include("common.jl")
let
    geom = get_rectangular_domain(Float64, (-1.0,), (1.0,); is_periodic = false)
    dx = 2 / 80
    pg = ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, 2.5dx), 1; randomness = (0.3dx,), rng = MersenneTwister(1))
    N = pg.meta.N; x = i -> pg.core.positions[i][1]; quad = f -> sum(pg.core.volumes[i] * f(x(i)) for i in 1:N)
    ghost = view(pg.core.is_boundary, 1:N)
    c1 = sum(view(pg.core.volumes, 1:N)) ≈ 2.0; c2 = all(i -> !ghost[i] || pg.core.volumes[i] == 0, 1:N)
    es = abs(quad(x -> x^2) - 2 / 3)
    update_volumes!(pg, MLSVolumes(2; samples_per_dx = 4))
    c3 = all(i -> !ghost[i] || pg.core.volumes[i] == 0, 1:N); em = abs(quad(x -> x^2) - 2 / 3); e1 = abs(quad(x -> x))
    println("1-D: ΣV≈2 $c1, Shepard ghosts 0 $c2, MLS ghosts 0 $c3, err_mls $(round(em, sigdigits=3)) (<2e-5: $(em < 2e-5)), err_shepard $(round(es, sigdigits=3)), |quad x| $(round(e1, sigdigits=3)) (<2e-5: $(e1 < 2e-5))")
end
let
    geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)); h = 1 / 20
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    inner = interior(pg); N = pg.meta.N
    f(x) = exp(x[1]) * cos(x[2]); If = (exp(1) - 1) * sin(1)
    g(x) = exp(x[1] + x[2]); Ig = (exp(1) - 1)^2
    quad = (f) -> sum(pg.core.volumes[i] * f(pg.core.positions[i]) for i in inner)
    s1 = sum(pg.core.volumes[i] for i in inner); r1 = abs(quad(f) - If) / If; r1g = abs(quad(g) - Ig) / Ig
    update_volumes!(pg, MLSVolumes(1))
    r2 = abs(quad(f) - If) / If; r2g = abs(quad(g) - Ig) / Ig
    for i in 1:N; pg.rhos[i] = State{1,Float64}(3.0); end
    println("2-D: Σ_int V $(round(s1, sigdigits=4)) | Shepard rel f1 $(round(r1, sigdigits=3)) (<5e-4: $(r1 < 5e-4)), exp(x+y) $(round(r1g, sigdigits=3)) | MLS1 rel f1 $(round(r2, sigdigits=3)) (<1e-4: $(r2 < 1e-4)), exp(x+y) $(round(r2g, sigdigits=3)) | total_mass $(total_mass(pg)[1])")
end
