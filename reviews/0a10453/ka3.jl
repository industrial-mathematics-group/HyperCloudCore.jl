include("common.jl")
const H = HyperCloudCore
negate(eq, U, w) = -U
function mk(T, D, bcs; jit = 0.2, Rf = 2.5, n = D == 3 ? 8 : (D == 2 ? 20 : 40), ball = false)
    dx = T(1/n)
    geom = ball ? get_spherical_domain(T, ntuple(_ -> 0.0, D), 1.0; bc_map = bcs) :
                  get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = bcs)
    h = ball ? T(2/n) : dx
    pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(T(1), T(Rf) * h), 2; randomness = ntuple(_ -> T(jit) * h, D), rng = MersenneTwister(2))
    for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{2,T}(1 + sin(3x[1]) + x[end]^2, cos(2sum(x))); end
    pg
end
function ap(pg, eq)
    buf = copy(pg.rhos)
    for i in ghosts(pg); buf[i] = buf[i] .* 0 .+ 99; end
    for (tag, bc) in pg.geometry.bc_map; bc(pg, buf, tag, nothing, eq, 0.0); end
    buf
end
function parity(T, D)
    bcs = D == 1 ? BCD(1 => MirrorBC(negate; degree = 2), 2 => ExtrapolationBC(2)) :
          BCD(1 => MirrorBC(negate; degree = 2), 2 => ExtrapolationBC(2), 3 => MirrorBC(; degree = 1, bounded = false), 4 => ExtrapolationBC(1),
              (D == 3 ? (5 => MirrorBC(; degree = 2), 6 => OutflowBC()) : ())...)
    pg = mk(T, D, bcs; Rf = D == 3 ? 3.0 : 2.5)
    eq = LinearAdvection(ntuple(_ -> (1.0, 0.5), D), T)
    println("  poly"); flush(stdout); ref = ap(pg, eq)
    println("  threads"); flush(stdout); set_backend!(ThreadsBackend()); th = ap(pg, eq)
    println("  ka"); flush(stdout); set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64)); ka = ap(pg, eq); println("  ka done"); flush(stdout)
    set_backend!(PolyesterBackend())
    sc = maximum(maximum(abs, ref[i]) for i in 1:pg.meta.N)
    @printf("%s D=%d: ghosts %d, Threads == Polyester: %s, max |KA - Polyester| / scale = %.2e, all finite: %s\n", T, D, length(ghosts(pg)),
            th == ref, maximum(maximum(abs, ka[i] - ref[i]) for i in 1:pg.meta.N) / sc, all(i -> all(isfinite, ka[i]), 1:pg.meta.N))
    flush(stdout)
end
which = ARGS[1]
if which == "parity"
    for T in (Float64, Float32), D in (1, 2, 3); parity(T, D); end
elseif which == "ball"
    for D in (2, 3)
        pg = mk(Float64, D, BCD(1 => MirrorBC(; degree = 2, bounded = false)); ball = true, n = D == 3 ? 10 : 24, Rf = 2.5)
        eq = LinearAdvection(ntuple(_ -> (1.0, 0.5), D), Float64)
        f(x) = 1 + x[1] - 2x[2] + x[1]^2 + x[1]*x[end]
        for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{2,Float64}(f(x), 0); end
        ref = ap(pg, eq)
        e = maximum(abs(ref[i][1] - f(mirrorpt(pg, i))) for i in ghosts(pg))
        insideok = count(i -> norm(mirrorpt(pg, i)) > 1 + 1e-12, ghosts(pg))
        set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64)); ka = ap(pg, eq); set_backend!(PolyesterBackend())
        @printf("ball D=%d: ghosts %d, max err at mirror point %.2e, mirror points outside the ball: %d, KA diff %.2e\n", D, length(ghosts(pg)), e, insideok,
                maximum(maximum(abs, ka[i] - ref[i]) for i in 1:pg.meta.N)); flush(stdout)
    end
elseif which == "cost"
    for (name, bc, Rf) in (("OutflowBC", OutflowBC(), 2.5), ("FixedDirichlet", FixedDirichlet(), 2.5), ("ExtrapolationBC(1)", ExtrapolationBC(1), 2.5),
                       ("MirrorBC(1)", MirrorBC(; degree = 1), 2.5), ("MirrorBC(negate; 2)", MirrorBC(negate; degree = 2), 2.5),
                       ("OutflowBC R3.5", OutflowBC(), 3.5), ("ExtrapolationBC(3) R3.5", ExtrapolationBC(3), 3.5), ("MirrorBC(3) R3.5", MirrorBC(; degree = 3), 3.5))
        pg = mk(Float64, 2, BCD(t => bc for t in 1:4); n = 200, Rf = Rf)
        eq = LinearAdvection(((1.0, 0.5), (1.0, 0.5)), Float64)
        buf = copy(pg.rhos)
        g!(pg, buf, bc, eq) = (for t in 1:4; bc(pg, buf, t, nothing, eq, 0.0); end)
        g!(pg, buf, bc, eq)
        a = @allocated g!(pg, buf, bc, eq)
        tm = minimum(@elapsed(g!(pg, buf, bc, eq)) for _ in 1:20)
        @printf("%-24s N=%d ghosts=%d  allocated %d B, time %.3f ms (all 4 tags, -t 2)\n", name, pg.meta.N, length(ghosts(pg)), a, 1e3tm); flush(stdout)
    end
    pg = mk(Float64, 2, BCD(t => MirrorBC(negate; degree = 2) for t in 1:4))
    eq = LinearAdvection(((1.0, 0.5), (1.0, 0.5)), Float64)
    buf = copy(pg.rhos); bc = MirrorBC(negate; degree = 2)
    set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64))
    h!(pg, buf, bc, eq) = bc(pg, buf, 1, nothing, eq, 0.0)
    h!(pg, buf, bc, eq)
    @printf("KA launch allocates %d B\n", @allocated h!(pg, buf, bc, eq))
    set_backend!(PolyesterBackend())
    # order independence
    pg = mk(Float64, 2, BCD(1 => MirrorBC(; degree = 2), 2 => ExtrapolationBC(2), 3 => OutflowBC(), 4 => FixedDirichlet()))
    bcs = [(t, pg.geometry.bc_map[t]) for t in 1:4]
    function seq(order)
        buf = copy(pg.rhos); for i in ghosts(pg); buf[i] = buf[i] .* 0 .+ 99; end
        for k in order; t, bc = bcs[k]; bc(pg, buf, t, nothing, eq, 0.0); end
        buf
    end
    println("tag order independence (Mirror, Extrap, Outflow, Dirichlet; 3 orders): ", seq([1,2,3,4]) == seq([4,3,2,1]) == seq([3,1,4,2]))
end
if which == "ka3d3"
    pg = mk(Float64, 3, BCD(t => MirrorBC(; degree = 3) for t in 1:6); Rf = 3.0)
    eq = LinearAdvection(ntuple(_ -> (1.0, 0.5), 3), Float64)
    bc = MirrorBC(; degree = 3); buf = copy(pg.rhos)
    t0 = time(); bc(pg, buf, 1, nothing, eq, 0.0); println("Polyester first call (compile) 3-D degree 3: ", round(time() - t0, digits = 1), " s"); flush(stdout)
    set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64))
    t0 = time(); bc(pg, buf, 1, nothing, eq, 0.0); println("KA CPU first call (compile) 3-D degree 3: ", round(time() - t0, digits = 1), " s"); flush(stdout)
end
