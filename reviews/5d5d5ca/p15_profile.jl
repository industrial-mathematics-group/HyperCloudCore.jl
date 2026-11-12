include("common.jl")
SEQ = get(ENV, "SEQ", "0") == "1"
SEQ && include("seq_sweep.jl")
function run2d(N, w; tend = 0.5)
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), w === nothing ? (t -> RigidMotion(SVector(0.5, 0.0), SMatrix{2,2}(1.0,0,0,1), SVector(0.0,0.0), zero(SMatrix{2,2,Float64}))) : translating((w, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5))
    geom = GeometricDomain(intersect(box, piston); mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => ExtrapolationBC(1)))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    eq = LinearAdvection(((1.0,), (0.0,)), Float64)
    u0(x, t) = 1 + 0.5 * sin(2π * (x[1] - t)) * cos(2π * x[2])
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(u0(pg.core.positions[i], 0.0)); end
    ts = GeneralRKTimeStepper(eq, MUSCL(Float64, 2, 1, 3, NoLimiter(), RusanovFlux()), MOOD(), RK3_SSP_Tableau(Float64))
    dt = 0.4 * h; t = 0.0
    while t < tend - 1e-12; d = min(dt, tend - t); ts(eq, pg, t, d); t += d; end
    face = w === nothing ? 1.5 : 1 + w * t
    inner = [i for i in interior(pg) if pg.core.positions[i][1] > t + 0.2]
    prof = Dict{Int, Float64}()
    for i in inner
        k = floor(Int, (face - pg.core.positions[i][1]) / h)
        prof[k] = max(get(prof, k, 0.0), abs(pg.rhos[i][1] - u0(pg.core.positions[i], t)))
    end
    @printf("  %s N=%d w=%s: max err by distance to the face (in h): %s\n", SEQ ? "seq" : "orig", N, string(w),
            join([@sprintf("%d:%.3f", k, prof[k]) for k in sort(collect(keys(prof)))[1:min(end, 12)]], " "))
end
for N in (20, 40); run2d(N, nothing); run2d(N, 1.0); end
