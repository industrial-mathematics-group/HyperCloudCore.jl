include("common.jl")
const H = HyperCloudCore
negate(eq, U, w) = -U
function mk(T, bcs; seed = 2)
    D = 3; dx = T(1/8)
    geom = get_rectangular_domain(T, (0.,0.,0.), (1.,1.,1.); bc_map = bcs)
    pg = ParticleGrid(geom, (dx,dx,dx), ExponentialWeightFunction(T(1), T(3.0) * dx), 2; randomness = (T(0.2)*dx, T(0.2)*dx, T(0.2)*dx), rng = MersenneTwister(seed))
    for i in 1:pg.meta.N; x = pg.core.positions[i]; pg.rhos[i] = State{2,T}(1 + sin(3x[1]) + x[end]^2, cos(2sum(x))); end
    pg
end
function ap1(pg, eq, tag, bc)
    buf = copy(pg.rhos)
    for i in ghosts(pg); buf[i] = buf[i] .* 0 .+ 99; end
    bc(pg, buf, tag, nothing, eq, 0.0)
    buf
end
for T in (Float64, Float32)
    bcs = BCD(1 => MirrorBC(negate; degree = 2), 2 => ExtrapolationBC(2), 3 => MirrorBC(; degree = 1, bounded = false), 4 => ExtrapolationBC(1), 5 => MirrorBC(; degree = 2), 6 => OutflowBC())
    pg = mk(T, bcs)
    pg64 = mk(Float64, bcs)   # same seed: same positions up to rounding of the jitter
    eq = LinearAdvection(ntuple(_ -> (1.0, 0.5), 3), T); eq64 = LinearAdvection(ntuple(_ -> (1.0, 0.5), 3), Float64)
    for tag in 1:6
        bc = bcs[tag]
        r64 = ap1(pg64, eq64, tag, bc)
        out = String[]
        for (name, be) in (("Poly", PolyesterBackend()), ("Threads", ThreadsBackend()), ("KA", KABackend(H.KA.CPU(); workgroupsize = 64)))
            set_backend!(be); r = ap1(pg, eq, tag, bc); set_backend!(PolyesterBackend())
            sel = [i for i in ghosts(pg) if pg.core.tags[i] == tag]
            push!(out, @sprintf("%s: max|.-F64| %.1e", name, maximum(maximum(abs, Float64.(r[i]) - r64[i]) for i in sel)))
        end
        println("$T tag $tag $(typeof(bc).name.name): ", join(out, ", "))
    end
end
