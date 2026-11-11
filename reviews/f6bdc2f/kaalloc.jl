include("common.jl")
const H = HyperCloudCore
negate(eq, U, w) = -U
function run()
    T = Float64; dx = 1/20
    for (name, bc) in (("FixedDirichlet", FixedDirichlet()), ("OutflowBC", OutflowBC()), ("ExtrapolationBC(0)", ExtrapolationBC(0)), ("ExtrapolationBC(1)", ExtrapolationBC(1)),
                       ("MirrorBC(0)", MirrorBC(; degree = 0)), ("MirrorBC(1)", MirrorBC(; degree = 1)), ("MirrorBC(negate;2)", MirrorBC(negate; degree = 2)))
        for M in (1, 2)
            geom = get_rectangular_domain(T, (0.0, 0.0), (1.0, 1.0); bc_map = BCD(t => bc for t in 1:4))
            pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(1.0, 2.5dx), M; randomness = (0.2dx, 0.2dx), rng = MersenneTwister(2))
            eq = LinearAdvection(ntuple(_ -> ntuple(_ -> 1.0, M), 2), T)
            buf = copy(pg.rhos)
            set_backend!(KABackend(H.KA.CPU(); workgroupsize = 64))
            bc(pg, buf, 1, nothing, eq, 0.0); bc(pg, buf, 1, nothing, eq, 0.0)
            a = @allocated bc(pg, buf, 1, nothing, eq, 0.0)
            set_backend!(PolyesterBackend())
            @printf("KA CPU %-20s M=%d N=%d: %d bytes per launch\n", name, M, pg.meta.N, a)
        end
    end
end
run()
