# The changed kernels (_place_ghost! with the curvature rotation, _ghost_fit_body! with gm.R * H * gm.R') on KA CPU
# (PoCL) vs Polyester, 3-D: a ball rotating about an off-centre axis in a box; Float64 and Float32
using HyperCloudCore, StaticArrays, LinearAlgebra, Printf
const H = HyperCloudCore
struct RecH end      # records the Hessian (9) of the WallPoint
(::RecH)(eq, U, w) = State{9, eltype(w.distance)}(Tuple(w.hessian))
BCD = Dict{Int, AbstractBoundaryCondition}
for T in (Float64, Float32)
    ax = normalize(SVector{3, T}(1, 2, 2))
    sh = Cuboid((T(-1), T(-1), T(-1)), (T(1), T(1), T(1))) - Moving(Tagged(Ball((T(0.3), T(0), T(0)), T(0.3)), 7), rotating(Tuple(ax), T(2); center = (T(0), T(0), T(0))))
    geom = GeometricDomain(sh; bc_map = BCD(merge(Dict(t => MirrorBC(RecH()) for t in 1:6), Dict(7 => MirrorBC(RecH())))))
    res = Dict{String, Any}()
    for (bn, be) in (("polyester", PolyesterBackend()), ("ka-cpu", KABackend(H.KA.CPU(); workgroupsize = 32)))
        set_backend!(be)
        pg = ParticleGrid(geom, (T(0.1), T(0.1), T(0.1)), ExponentialWeightFunction(one(T), T(0.25)), 9)
        X = copy(pg.core.positions)
        H.place_ghosts!(pg, X, zero(T), T(0.2))     # rotates normals + curvatures to t = 0.2
        buf = copy(pg.rhos)
        MirrorBC(RecH())(pg, buf, 7, nothing, nothing, T(0.35))   # fit kernel at a later stage time
        idx = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 7]
        # exact: Hessian of the ball (complement) at the wall point at t = 0.35
        shp = at_time(sh, T(0.35))
        err = maximum(idx) do i
            gm = H.tag_motion(pg.geometry, 7, pg.meta.boundary_time, T(0.35))
            x = H.move(gm, X[i]); n = H.rotate(gm, pg.core.normals[i]); xw = x - pg.core.wall_distances[i] * n
            maximum(abs, SMatrix{3, 3, T}(Tuple(buf[i])) - H._sdf_hessian(shp, xw))
        end
        res[bn] = (Array(buf[idx]), Array(pg.core.wall_curvatures[idx]))
        @printf("%-8s %-10s %d ball ghosts: max |H_fit - H_exact(t)| = %.2e\n", T, bn, length(idx), err)
    end
    set_backend!(PolyesterBackend())
    a, b = res["polyester"], res["ka-cpu"]
    @printf("%-8s KA vs Polyester: hessians %.2e, stored curvatures %.2e\n", T, maximum(maximum(abs, x - y) for (x, y) in zip(a[1], b[1])), maximum(maximum(abs, x - y) for (x, y) in zip(a[2], b[2])))
end
