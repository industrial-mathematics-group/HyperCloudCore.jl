# Static ghost-state accuracy on curved walls and obstacles: interior = smooth u, apply the condition,
# error at the ghosts against u at the target (ghost position for Extrapolation/Outflow, mirror point for Mirror).
include("common.jl")
eqs = Dict(2 => LinearAdvection(((1.0,), (0.0,)), Float64), 3 => LinearAdvection(((1.0,), (0.0,), (0.0,)), Float64))
function applybc2(pg, D)
    buf = copy(pg.rhos)
    for i in ghosts(pg); buf[i] = State{1,Float64}(99.0); end
    for (tag, bc) in pg.geometry.bc_map; bc(pg, buf, tag, nothing, eqs[D], 0.0); end
    buf
end
u2(x) = sin(2x[1] + 1) * cos(3x[2] - 0.5)
u3(x) = sin(2x[1] + 1) * cos(3x[2] - 0.5) * cos(x[3] + 0.3)
function shapes(D)
    D == 2 ? [("disc r=1", () -> Ball((0.0, 0.0), 1.0), 2.0),
              ("box - rot plate", () -> Tagged(Cuboid((-1.0, -0.5), (1.0, 0.5)), 1) - Tagged(Rotated(Cuboid((-0.2, -0.05), (0.2, 0.05)), π / 6), 1), 1.0),
              ("step .61/.213", () -> Tagged(Cuboid((0.0, 0.0), (3.0, 1.0)), 1) - Tagged(Cuboid((0.61, -1.0), (4.0, 0.213)), 1), 1.0)] :
             [("ball r=1", () -> Ball((0.0, 0.0, 0.0), 1.0), 2.0)]
end
for D in (2, 3), (nm, mk, L) in shapes(D), jit in (0.0, 0.3)
    f = D == 2 ? u2 : u3
    println("== $D-D $nm jitter $jit")
    for (bn, bc) in (("Outflow", OutflowBC()), ("Extrap1", ExtrapolationBC(1)), ("Extrap2", ExtrapolationBC(2)),
                     ("Mirror1", MirrorBC(; degree = 1)), ("Mirror2", MirrorBC(; degree = 2)), ("Mirror2 unb", MirrorBC(; degree = 2, bounded = false)))
        errs = Float64[]; errs_near = Float64[]
        for n in (D == 2 ? (20, 40, 80) : (10, 20, 30))
            h = L / n
            geom = GeometricDomain(mk(); bc_map = BCD(1 => bc))
            pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(1.0, 2.5h), 1;
                              randomness = ntuple(_ -> jit * h, D), rng = MersenneTwister(2))
            for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f(pg.core.positions[i])); end
            buf = applybc2(pg, D)
            e = 0.0; en = 0.0
            for i in ghosts(pg)
                xt = bn[1] == 'M' ? mirrorpt(pg, i) : pg.core.positions[i]
                ei = abs(buf[i][1] - f(xt)); e = max(e, ei)
                pg.core.wall_distances[i] < h && (en = max(en, ei))
            end
            push!(errs, e); push!(errs_near, en)
        end
        @printf("  %-12s max %s   (ghosts φ<h: %s)  EOC %.2f\n", bn, join([@sprintf("%.2e", x) for x in errs], " "),
                join([@sprintf("%.2e", x) for x in errs_near], " "), log2(errs[end-1] / errs[end]) / (D == 3 ? log2(30/20) : 1))
    end
    flush(stdout)
end
