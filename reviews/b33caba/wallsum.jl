# Are the skipped interior->ghost slots a consistent boundary flux? Sum the face vectors S_ij = 2 V_i c_ij of the
# wall slots per ghost tag and compare with the exact ∫ n dA of that wall (unit box: |face| = 1, outward normal).
# Also: Σ over all slots of particle i (interior + wall) ~ 0 (consistency for a constant flux).
include("common.jl")
const HC = HyperCloudCore
function wallsums(pg, m, q)
    N = pg.meta.N; nb = pg.neighbor; invL = 1 / minimum(pg.meta.dx)
    S = zeros(SVector{2,Float64}, length(nb.indices))
    for i in 1:N
        sl = nb.ranges[i]
        qq = HC._divergence_index(Val(0), Val(2), q, length(sl)) - 1
        HC._face_vectors!(S, sl, nb.distances, nb.weights, invL, 2 * pg.core.volumes[i], qq, Val(2))
    end
    sums = Dict{Int, SVector{2,Float64}}(); allsum = zero(SVector{2,Float64}); worst = 0.0
    for i in interior(pg)
        si = zero(SVector{2,Float64})
        for k in nb.ranges[i]
            j = nb.indices[k]; si += S[k]
            pg.core.is_boundary[j] && (sums[pg.core.tags[j]] = get(sums, pg.core.tags[j], zero(SVector{2,Float64})) + S[k])
        end
        worst = max(worst, norm(si) / pg.core.volumes[i])
    end
    sums, worst
end
for N in (20, 40, 80)
    h = 1 / N
    pg = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for jit in (0.0, 0.3)
        jit > 0 && (pg = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(3)))
        s, w = wallsums(pg, nothing, 3)
        println(@sprintf("N=%d jit %.1f: ", N, jit), join([@sprintf("tag %d: (%.4f, %.4f)", k, v...) for (k, v) in sort(collect(s))], "  "), @sprintf("  | max_i |Σ_j S_ij|/V_i %.2e", w))
    end
end
