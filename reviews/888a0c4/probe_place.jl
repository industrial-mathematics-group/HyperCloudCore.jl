# place_ghosts! on a 3-D box with a static tag, a translating ball (tag 40), a rotating cuboid (tag 7) and a second
# piece sharing nothing: Polyester and KA CPU; dumps positions, normals and curvatures for a cross-commit comparison
using HyperCloudCore, StaticArrays, LinearAlgebra, Serialization
const H = HyperCloudCore
T = Float64
ax = normalize(SVector(1.0, 2.0, 2.0))
sh = Cuboid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)) - Moving(Tagged(Ball((0.4, 0.0, 0.0), 0.25), 40), translating((-0.3, 0.1, 0.0))) -
     Moving(Tagged(Cuboid((-0.6, -0.2, -0.2), (-0.2, 0.2, 0.2)), 7), rotating(Tuple(ax), 1.5; center = (-0.4, 0.0, 0.0)))
BCD = Dict{Int, AbstractBoundaryCondition}
geom = GeometricDomain(sh; bc_map = BCD(t => OutflowBC() for t in (1, 2, 3, 4, 5, 6, 7, 40)))
out = Dict{String, Any}()
for (bn, be) in (("polyester", PolyesterBackend()), ("ka", KABackend(H.KA.CPU(); workgroupsize = 32)))
    set_backend!(be)
    pg = ParticleGrid(geom, (0.1, 0.1, 0.1), ExponentialWeightFunction(1.0, 0.25), 1)
    X = copy(pg.core.positions)
    H.place_ghosts!(pg, X, 0.0, 0.1); H.place_ghosts!(pg, X, 0.1, 0.25)
    out[bn] = (Array(X), Array(pg.core.normals), Array(pg.core.wall_curvatures), Array(pg.core.tags))
    # exactness: moved ghosts of each piece vs the pose at 0.25
    for (tag, k) in ((40, 1), (7, 2))
        m = geom.pieces[k]; P0, P1 = H.pose(m, 0.0), H.pose(m, 0.25)
        idx = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == tag]
        e = maximum(i -> norm(X[i] - (P1.c + P1.R * (P0.R' * (pg.core.positions[i] - P0.c)))), idx)
        println("$bn tag $tag: $(length(idx)) ghosts, max position error $e")
    end
    st = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] <= 6]
    println("$bn static ghosts unchanged: ", all(i -> X[i] == pg.core.positions[i], st))
end
set_backend!(PolyesterBackend())
a, b = out["polyester"], out["ka"]
println("KA vs Polyester: positions ", maximum(norm.(a[1] - b[1])), ", normals ", maximum(norm.(a[2] - b[2])), ", curvatures ", maximum(norm.(a[3] - b[3])))
serialize(ARGS[1], a)
