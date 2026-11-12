# conservation_defect: bounded lattice / jittered box / periodic / disc / 1-D, par (1fcc2ce) vs new (wall field)
include("common.jl")
m2 = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
m1 = MUSCL(Float64, 1, 1, 3, NoLimiter(), RusanovFlux())
function show(name, m, pg; drop = false)
    fill!(view(pg.core.particle_orders, 1:pg.meta.N), 3)
    if drop   # MOOD-style drop on the first interior row next to the wall at y=0 (a few particles)
        for i in interior(pg); pg.core.wall_distances[i] < 0.6 * pg.meta.dx[1] && pg.core.positions[i][1] < 0.3 && (pg.core.particle_orders[i] = 1); end
    end
    r = HyperCloudCore.conservation_defect(m, pg)
    I = interior(pg); h = minimum(pg.meta.dx)
    near = [i for i in I if pg.core.wall_distances[i] < 3h]; deep = setdiff(I, near)
    w = hasproperty(r, :wall) ? r.wall.pairs : -1
    ghostp = maximum((r.particle[i] for i in ghosts(pg)); init = 0.0)
    nanp = count(isnan, r.particle)
    @printf("%-28s total %.3e (pairs %d) equal %.3e mixed %.3e (%d) onesided %d wall %d | max near %.3e deep %.3e ghost %.3e nan %d\n",
            name, r.total.rel, r.total.pairs, r.equal.rel, r.mixed.rel, r.mixed.pairs, r.onesided.pairs, w,
            isempty(near) ? 0.0 : maximum(r.particle[i] for i in near), isempty(deep) ? 0.0 : maximum(r.particle[i] for i in deep), ghostp, nanp)
end
h = 0.05
box() = ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
show("box lattice (test)", m2, box())
show("box lattice, drops at wall", m2, box(); drop = true)
show("box jitter 0.3", m2, ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (0.3h, 0.3h), rng = MersenneTwister(3)))
show("periodic jitter 0.3", m2, ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, true)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (0.3h, 0.3h), rng = MersenneTwister(3)))
show("channel (periodic x)", m2, ParticleGrid(get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false)), (h, h), ExponentialWeightFunction(1.0, 2.5h), 1))
show("disc n=40", m2, ParticleGrid(get_spherical_domain(Float64, (0.0, 0.0), 1.0), (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1))
show("disc n=40 jitter 0.3", m2, ParticleGrid(get_spherical_domain(Float64, (0.0, 0.0), 1.0), (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1; randomness = (0.015, 0.015), rng = MersenneTwister(3)))
show("1-D [0,1] N=40", m1, ParticleGrid(get_rectangular_domain(Float64, (0.0,), (1.0,)), (0.025,), ExponentialWeightFunction(1.0, 0.0625), 1))
