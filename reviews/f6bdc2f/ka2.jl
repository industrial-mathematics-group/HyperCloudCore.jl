include("common.jl")
const H = HyperCloudCore
negate(eq, U, w) = -U
T = Float64; D = 2; dx = 1/20
bcs = BCD(1 => MirrorBC(negate; degree = 2), 2 => ExtrapolationBC(2), 3 => MirrorBC(; degree = 1, bounded = false), 4 => ExtrapolationBC(1))
geom = get_rectangular_domain(T, (0.0, 0.0), (1.0, 1.0); bc_map = bcs)
pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), 2; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(2))
for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{2,T}(1 + sin(3x[1]), cos(2sum(x))); end
eq = LinearAdvection(((1.0, 0.5), (1.0, 0.5)), T)
function ap(pg, eq)
    buf = copy(pg.rhos)
    for (tag, bc) in pg.geometry.bc_map
        t0 = time(); bc(pg, buf, tag, nothing, eq, 0.0); println("   tag $tag $(typeof(bc).name.name) $(round(time()-t0, digits=1)) s"); flush(stdout)
    end
    buf
end
println("Polyester"); flush(stdout); ref = ap(pg, eq)
println("Threads"); flush(stdout); set_backend!(ThreadsBackend()); th = ap(pg, eq); set_backend!(PolyesterBackend()); println(th == ref)
println("KA"); flush(stdout); set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64)); ka = ap(pg, eq); set_backend!(PolyesterBackend())
println(maximum(maximum(abs, ka[i] - ref[i]) for i in 1:pg.meta.N))
