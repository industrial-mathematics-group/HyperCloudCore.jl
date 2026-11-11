include("common.jl")
const H = HyperCloudCore
T = Float32; dx = T(1/10)
f(x) = 1 + x[1] + x[2] + x[3] + x[1]^2 - x[2]*x[3]
for (name, bc) in (("ExtrapolationBC(2)", ExtrapolationBC(2)), ("MirrorBC(2, unbounded)", MirrorBC(; degree = 2, bounded = false)))
    geom = get_rectangular_domain(T, (0.,0.,0.), (1.,1.,1.); bc_map = BCD(t => bc for t in 1:6))
    pg = ParticleGrid(geom, (dx,dx,dx), ExponentialWeightFunction(T(1), T(2.5)*dx), 1; randomness = (T(0.2)*dx, T(0.2)*dx, T(0.2)*dx), rng = MersenneTwister(1))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(T(f(Float64.(pg.core.positions[i])))); end
    eq = LinearAdvection(((1.0,), (1.0,), (1.0,)), T)
    out = String[]
    for (bn, be) in (("Polyester", PolyesterBackend()), ("Threads", ThreadsBackend()), ("KA CPU", KABackend(H.KA.CPU(); workgroupsize = 64)))
        set_backend!(be)
        buf = copy(pg.rhos); for i in ghosts(pg); buf[i] = State{1,T}(99); end
        for t in 1:6; bc(pg, buf, t, nothing, eq, 0.0); end
        set_backend!(PolyesterBackend())
        xt(i) = bc isa ExtrapolationBC ? Float64.(pg.core.positions[i]) : Float64.(mirrorpt(pg, i))
        e = [abs(buf[i][1] - f(xt(i))) for i in ghosts(pg)]
        push!(out, @sprintf("%s: max err %.2e, #>1e-2: %d", bn, maximum(e), count(>(1e-2), e)))
    end
    println("Float32 3-D $name (", length(ghosts(pg)), " ghosts): ", join(out, "; "))
end
