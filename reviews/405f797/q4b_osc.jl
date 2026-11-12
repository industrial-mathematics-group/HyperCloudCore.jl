include("common.jl")
h = 1/20
bcs = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => ExtrapolationBC(1))
osc = t -> RigidMotion(SVector(0.3sin(2π * t), 0.0), SMatrix{2,2,Float64,4}(1, 0, 0, 1), SVector(0.6π * cos(2π * t), 0.0), zero(SMatrix{2,2,Float64,4}))
pc = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), osc)
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), pc); mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = bcs)
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
println("t = 0: ghosts ", count(i -> pg.core.is_boundary[i], 1:pg.meta.N), " interior ", count(i -> !pg.core.is_boundary[i], 1:pg.meta.N))
t = 0.0; dt = 0.1h / (0.6π)
nst = round(Int, 3 / dt)
for k in 1:nst
    global t; t += dt
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg); sweep_boundaries!(pg, 0.0, t, 3)
    if k % round(Int, 0.25 / dt) == 0
        @printf("t = %.2f face %.3f: ghosts %d (static tags 1-4: %d), interior %d, ΣV %.5f (%.5f)\n", t, 1 + 0.3sin(2π * t), count(i -> pg.core.is_boundary[i], 1:pg.meta.N),
                count(i -> pg.core.is_boundary[i] && pg.core.tags[i] <= 4, 1:pg.meta.N), count(i -> !pg.core.is_boundary[i], 1:pg.meta.N), sumV(pg), 0.5 * (1 + 0.3sin(2π * t)))
    end
end
