# conservation_defect on a bounded 2-D lattice (ghosts with V = 0 after #40)
include("common.jl")
h = 0.05
pg = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
m = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
fill!(view(pg.core.particle_orders, 1:pg.meta.N), 3)
r = HyperCloudCore.conservation_defect(m, pg)
I = interior(pg)
deep = [i for i in I if all(abs(x - 0.5) < 0.5 - 3h for x in pg.core.positions[i])]
@printf("total.rel %.3e equal.rel %.3e onesided pairs %d | max particle defect: interior near wall %.3e, deep interior %.3e\n",
        r.total.rel, r.equal.rel, r.onesided.pairs, maximum(r.particle[i] for i in setdiff(I, deep)), maximum(r.particle[i] for i in deep))
