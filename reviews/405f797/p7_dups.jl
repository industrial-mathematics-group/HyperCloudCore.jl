include("common.jl")
N = 20; h = 1/N; w = 1.0
piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0)))
box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, true))
geom = GeometricDomain(intersect(box, piston); is_periodic = (false, true), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                       bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 5 => MirrorBC(; degree = 1)))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
u0(x, t) = 1 + exp(-((x[1] - 0.8 - t)^2 + (x[2] - 0.25)^2) / 0.02)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
eq = LinearAdvection(((1.0,), (0.0,)), Float64)
ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
println("initial interior ", length(interior(pg)), " ΣV ", sumV(pg))
t = 0.0; dt = 0.4h
for s in 1:40
    global t
    V0 = sumV(pg); n0 = length(interior(pg))
    ts(eq, pg, t, dt); t += dt
    inner = interior(pg)
    P = [pg.core.positions[i] for i in inner]
    dmin = minimum(norm(P[a] - P[b]) for a in eachindex(P) for b in a+1:length(P))
    # wrapped y distance
    dminw = minimum(begin d = P[a] - P[b]; d = SVector(d[1], d[2] - 0.5round(d[2]/0.5)); norm(d) end for a in eachindex(P) for b in a+1:length(P))
    if s <= 6 || s % 10 == 0
        @printf("step %2d t=%.3f n=%d (+%d) ΣV=%.5f (Δ %.5f, exact Δ %.5f) min pair dist %.3fh (wrapped %.3fh) ys of max-x particles: %s\n", s, t, length(inner), length(inner) - n0,
                sumV(pg), sumV(pg) - V0, 0.5 * w * dt, dmin/h, dminw/h,
                string(sort(unique(round.([p[2] for p in P if p[1] > maximum(q[1] for q in P) - 1e-9], digits = 4)))))
    end
end
