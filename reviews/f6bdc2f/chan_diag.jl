include("common.jl")
eqx = LinearAdvection(((1.0,), (0.0,)), Float64)
u0(x, t) = exp(-((mod(x[1] - t, 1.0) - 0.5)^2 + x[2]^2) / 0.02)
N = 40; h = 1/N
for jit in (0.0, 0.2)
bc = MirrorBC(; degree = 2)
geom = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0); is_periodic = (true, false), bc_map = BCD(3 => bc, 4 => bc))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit*h, jit*h), rng = MersenneTwister(7))
for i in 1:pg.meta.N; pg.rhos[i] = State{1,Float64}(u0(pg.core.positions[i], 0.0)); end
buf = applybc(pg, 1, Float64)
e = [(abs(buf[i][1] - u0(pg.core.positions[i], 0.0)), i) for i in ghosts(pg)]
em, im = maximum(e)
println("jit=$jit t=0: max ghost error vs exact even extension: $em at x=$(pg.core.positions[im]) φ=$(pg.core.wall_distances[im]) n=$(pg.core.normals[im]) mirror=$(mirrorpt(pg, im))")
# x-range of ghosts / interior
println("  x-extent of particles: ", extrema(p[1] for p in pg.core.positions[1:pg.meta.N]), "  ghosts tags: ", unique(pg.core.tags[ghosts(pg)]))
interp = MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux())
ts = GeneralRKTimeStepper(eqx, interp, MOOD(), RK3_SSP_Tableau(Float64))
dt = 0.5 * get_time_step(pg, eqx, interp); t = 0.0
while t < 0.25 - 1e-12; d = min(dt, 0.25 - t); ts(eqx, pg, t, d); t += d; end
e = [(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)), i) for i in interior(pg)]
em, im = maximum(e)
println("  t=$t: max interior err $em at $(pg.core.positions[im]); value $(pg.rhos[im][1]) exact $(u0(pg.core.positions[im], t))")
# error excluding the region x<0.1 or x>0.9
e2 = maximum(abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)) for i in interior(pg) if 0.15 < pg.core.positions[i][1] < 0.85)
println("  err away from the x seam: $e2")
end
