include("common.jl")
h = 1/16
ball = Moving(Tagged(Ball((0.4, 0.5, 0.5), 0.2), 7), translating((0.5, 0.0, 0.0)))
geom = GeometricDomain(Cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) - ball; mins = (0.0, 0.0, 0.0), maxs = (1.0, 1.0, 1.0), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:7))
pg = ParticleGrid(geom, (h, h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
f(x) = 1.0 + x[2]
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end
println("initial interior U range: ", extrema(pg.rhos[i][1] for i in interior(pg)))
# manual: place + neighbour + sweep, no PDE
t = 0.0; dt = 0.025
for s in 1:20
    global t
    U0 = Dict(pg.core.positions[i] => pg.rhos[i][1] for i in interior(pg))
    place_ghosts!(pg, pg.core.positions, t, t + dt); pg.neighbor(pg)
    r = sweep_boundaries!(pg, t, t + dt, 3); t += dt
    inner = interior(pg)
    lo, hi = extrema(pg.rhos[i][1] for i in inner)
    bad = [i for i in inner if !(1 - 1e-9 <= pg.rhos[i][1] <= 2 + 1e-9)]
    @printf("step %2d: +%d -%d U in [%.4f, %.4f], %d out of [1,2]", s, r.inserted, r.removed, lo, hi, length(bad))
    if !isempty(bad)
        i = bad[argmax([abs(pg.rhos[i][1] - 1.5) for i in bad])]
        x = pg.core.positions[i]
        @printf("  worst at %s (new particle: %s), dist to ball %.3fh, to box wall %.3fh, U=%.4f f=%.4f", string(round.(x, digits = 3)), !haskey(U0, x),
                (norm(x - SVector(0.4 + 0.5t, 0.5, 0.5)) - 0.2)/h, minimum(min.(x, 1 .- x))/h, pg.rhos[i][1], f(x))
    end
    println()
end
