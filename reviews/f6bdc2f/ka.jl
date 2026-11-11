include("common.jl")
const H = HyperCloudCore
D_of(pg::ParticleGrid{D}) where {D} = D
const EQ = Dict((D, T) => LinearAdvection(ntuple(_ -> (1.0, 0.5), D), T) for D in 1:3, T in (Float32, Float64))
negate(eq, U, w) = -U
struct SlipOp end
(::SlipOp)(eq, U, w) = U   # placeholder bits op
function mk(T, D, bcs; jit = 0.2, Rf = 2.5, n = D == 3 ? 8 : (D == 2 ? 20 : 40), ball = false)
    dx = T(1/n)
    geom = ball ? get_spherical_domain(T, ntuple(_ -> 0.0, D), 1.0; bc_map = bcs) :
                  get_rectangular_domain(T, ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D); bc_map = bcs)
    h = ball ? T(2/n) : dx
    pg = ParticleGrid(geom, ntuple(_ -> h, D), ExponentialWeightFunction(T(1), T(Rf) * h), 2; randomness = ntuple(_ -> T(jit) * h, D), rng = MersenneTwister(2))
    for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{2,T}(1 + sin(3x[1]) + x[end]^2, cos(2sum(x))); end
    pg
end
function ap(pg)
    buf = copy(pg.rhos)
    for i in ghosts(pg); buf[i] = buf[i] .* 0 .+ 99; end
    H.apply_boundary_conditions!(pg, buf, nothing, nothing, 0.0)
    buf
end
# apply_boundary_conditions! wants ts::TimeStepper, eq::HyperbolicPDE: call functors directly
function ap(pg)
    buf = copy(pg.rhos)
    for i in ghosts(pg); buf[i] = buf[i] .* 0 .+ 99; end
    for (tag, bc) in pg.geometry.bc_map; bc(pg, buf, tag, nothing, EQ[(D_of(pg), eltype(pg.meta.dx))], 0.0); end
    buf
end
println("== KA CPU vs Polyester (and Threads)")
for T in (Float64, Float32), D in (1, 2, 3)
    bcs = D == 1 ? BCD(1 => MirrorBC(negate; degree = 2), 2 => ExtrapolationBC(2)) :
          BCD(1 => MirrorBC(negate; degree = 2), 2 => ExtrapolationBC(2), 3 => MirrorBC(; degree = 1, bounded = false), 4 => ExtrapolationBC(1),
              (D == 3 ? (5 => MirrorBC(; degree = 3), 6 => OutflowBC()) : ())...)
    pg = mk(T, D, bcs; Rf = D == 3 ? 3.0 : 2.5)
    ref = ap(pg)
    set_backend!(ThreadsBackend()); th = ap(pg)
    set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64)); ka = ap(pg)
    set_backend!(PolyesterBackend())
    sc = maximum(maximum(abs, ref[i]) for i in 1:pg.meta.N)
    @printf("%s D=%d: ghosts %d, Threads == Polyester: %s, max |KA - Polyester| / scale = %.2e, all finite: %s\n", T, D, length(ghosts(pg)),
            th == ref, maximum(maximum(abs, ka[i] - ref[i]) for i in 1:pg.meta.N) / sc, all(i -> all(isfinite, ka[i]), 1:pg.meta.N))
end
println("== ball (curved wall), D=2,3: MirrorBC(bounded=false) degree 2 exact at the mirror point? KA parity")
for T in (Float64,), D in (2, 3)
    pg = mk(T, D, BCD(1 => MirrorBC(; degree = 2, bounded = false)); ball = true, n = D == 3 ? 10 : 24, Rf = 2.5)
    f(x) = 1 + x[1] - 2x[2] + x[1]^2 + x[1]*x[end]
    for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{2,T}(f(x), 0); end
    ref = ap(pg)
    e = maximum(abs(ref[i][1] - f(mirrorpt(pg, i))) for i in ghosts(pg))
    insideok = all(i -> norm(mirrorpt(pg, i)) <= 1 + 1e-12, ghosts(pg))
    set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64)); ka = ap(pg); set_backend!(PolyesterBackend())
    @printf("ball D=%d: ghosts %d, max err at mirror point %.2e, mirror points inside: %s, KA diff %.2e\n", D, length(ghosts(pg)), e, insideok,
            maximum(maximum(abs, ka[i] - ref[i]) for i in 1:pg.meta.N))
end
println("== allocation and cost (2-D, 200^2-ish grid, Float64, Polyester -t 2)")
for (name, bc) in (("OutflowBC", OutflowBC()), ("FixedDirichlet", FixedDirichlet()), ("ExtrapolationBC(1)", ExtrapolationBC(1)), ("ExtrapolationBC(3)", ExtrapolationBC(3)),
                   ("MirrorBC(1)", MirrorBC(; degree = 1)), ("MirrorBC(negate; 2)", MirrorBC(negate; degree = 2)), ("MirrorBC(3)", MirrorBC(; degree = 3)))
    pg = mk(Float64, 2, BCD(t => bc for t in 1:4); n = 200, Rf = name == "MirrorBC(3)" || name == "ExtrapolationBC(3)" ? 3.5 : 2.5)
    buf = copy(pg.rhos)
    f!(pg, buf) = (for t in 1:4; bc(pg, buf, t, nothing, EQ[(D_of(pg), eltype(pg.meta.dx))], 0.0); end)
    f!(pg, buf)
    a = @allocated f!(pg, buf)
    tm = minimum(@elapsed(f!(pg, buf)) for _ in 1:20)
    @printf("%-22s N=%d ghosts=%d  allocated %d B, time %.3f ms (all 4 tags)\n", name, pg.meta.N, length(ghosts(pg)), a, 1e3tm)
end
println("== KA allocation")
let pg = mk(Float64, 2, BCD(t => MirrorBC(negate; degree = 2) for t in 1:4))
    buf = copy(pg.rhos); bc = MirrorBC(negate; degree = 2)
    set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64))
    bc(pg, buf, 1, nothing, EQ[(2, Float64)], 0.0)
    @printf("KA launch allocates %d B\n", @allocated bc(pg, buf, 1, nothing, EQ[(2, Float64)], 0.0))
    set_backend!(PolyesterBackend())
end
println("== order independence across tags (Mirror, Extrap, Outflow, Dirichlet)")
let pg = mk(Float64, 2, BCD(1 => MirrorBC(; degree = 2), 2 => ExtrapolationBC(2), 3 => OutflowBC(), 4 => FixedDirichlet()))
    bcs = [(t, pg.geometry.bc_map[t]) for t in 1:4]
    function seq(order)
        buf = copy(pg.rhos); for i in ghosts(pg); buf[i] = buf[i] .* 0 .+ 99; end
        for k in order; t, bc = bcs[k]; bc(pg, buf, t, nothing, EQ[(D_of(pg), eltype(pg.meta.dx))], 0.0); end
        buf
    end
    b1 = seq([1, 2, 3, 4]); b2 = seq([4, 3, 2, 1]); b3 = seq([3, 1, 4, 2])
    println("identical for 3 orders: ", b1 == b2 == b3)
end
