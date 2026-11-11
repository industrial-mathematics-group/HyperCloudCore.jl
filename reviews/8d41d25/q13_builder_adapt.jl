include("common.jl")
h = 1/20
plate = Moving(Tagged(Cuboid((-0.2, -0.05), (0.2, 0.05)), t -> t + 4), translating((1.0, 0.0)))
chan  = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
geom  = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => MirrorBC(; degree = 1) for t in 3:8))
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
ad = H.build_adaptivity(Dict{Symbol, Any}(:enabled => true), Dict{Symbol, Any}(:T => Float64, :D => 2, :Grid => pg))
println("build_adaptivity on a static grid with a moving piece: volume_estimator = ", typeof(ad.volume_estimator))
try
    adapt_particles!(pg, ad, 3); println("adapt_particles!: ok")
catch e
    println("adapt_particles!: ", sprint(showerror, e)[1:min(end, 160)])
end
