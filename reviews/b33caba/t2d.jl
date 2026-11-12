# The new 2-D quadrature test of unit_schemes.jl (087cb73) as printed checks, Float64 and Float32, on any tree
include("common.jl")
for T in (Float64, Float32)
    geom = get_rectangular_domain(T, (zero(T), zero(T)), (one(T), one(T))); h = T(1 / 20)
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(one(T), T(2.5) * h), 1)
    N = pg.meta.N; inner = interior(pg)
    f(x) = exp(x[1] + x[2]); If = T((exp(1) - 1)^2)
    quad = () -> sum(pg.core.volumes[i] * f(pg.core.positions[i]) for i in inner)
    s = sum(pg.core.volumes[i] for i in inner); g0 = all(i -> !pg.core.is_boundary[i] || pg.core.volumes[i] == 0, 1:N)
    eS = abs(quad() - If) / If
    update_volumes!(pg, MLSVolumes(1)); e1 = abs(quad() - If) / If
    update_volumes!(pg, MLSVolumes(2)); e20 = abs(quad() - If) / If
    pg40 = ParticleGrid(geom, (h / 2, h / 2), ExponentialWeightFunction(one(T), T(1.25) * h), 1; volume_estimator = MLSVolumes(2))
    e40 = abs(sum(pg40.core.volumes[i] * f(pg40.core.positions[i]) for i in interior(pg40)) - If) / If
    for i in 1:N; pg.rhos[i] = State{1, T}(T(3)); end
    for i in 1:N; pg.core.is_boundary[i] && (pg.rhos[i] = State{1, T}(T(100))); end
    tm = total_mass(pg)[1]; im = interior_mass(pg)[1]
    @printf("%s: ΣV_int %.7f ghosts0 %s | Shepard %.3e (<1.2e-2 %s) MLS1 %.3e (<4e-3 %s) MLS2 N20 %.3e (<1e-4 %s) N40 %.3e ratio %.3f (<0.3 %s) | total %.6f interior %.6f (≈3 %s, == %s)\n",
            T, s, g0, eS, eS < 1.2e-2, e1, e1 < 4e-3, e20, e20 < 1e-4, e40, e40 / e20, e40 < 0.3e20 / 1e20 * e20, tm, im, tm ≈ 3, tm == im)
end
