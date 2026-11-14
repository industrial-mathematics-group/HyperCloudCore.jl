# SlipWall ghosts around user-polygon obstacles (exact SDF, FD Hessian) in a uniform flow u = (0.5, 0.2): a triangle
# and a square rotated by 30°, F64 / F32, several spacings. Every ghost should be the uniform state.
using HyperCloud, StaticArrays, LinearAlgebra, Printf
const HCC = HyperCloud.HyperCloudCore
const HC = HCC
include(joinpath(@__DIR__, "upoly.jl"))
HC.bounding_box(s::UPoly) = (Tuple(minimum(v[i] for v in s.v) for i in 1:2), Tuple(maximum(v[i] for v in s.v) for i in 1:2))
const γ = 1.4
for T in (Float64, Float32)
    eq = HyperCloud.EulerEquation(Val(2), T, T(γ))
    BCD = Dict{Int, HCC.AbstractBoundaryCondition}
    rot(θ) = SMatrix{2, 2, T}(cosd(θ), sind(θ), -sind(θ), cosd(θ))
    shapes = (("triangle", UPoly(Tuple(SVector{2, T}(v) for v in ((-0.3, -0.2), (0.35, -0.15), (0.0, 0.4))))),
              ("square rot 30", UPoly(Tuple(rot(30) * SVector{2, T}(v) for v in ((-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25))))))
    for (label, P) in shapes, dx in T.((0.02, 0.0125, 0.01, 0.005))
        sh = Cuboid((T(-1), T(-1)), (T(1), T(1))) - Tagged(P, 5)
        geom = GeometricDomain(sh; bc_map = BCD(t => slip_wall() for t in 1:5))
        pg = ParticleGrid(geom, (dx, dx), ExponentialWeightFunction(one(T), T(0.05)), 4)
        U0 = State{4, T}(1, T(0.5), T(0.2), T(1 / (γ - 1) + 0.5 * (0.25 + 0.04)))
        fill!(pg.rhos, U0); buf = copy(pg.rhos)
        slip_wall()(pg, buf, 5, nothing, eq, zero(T))
        idx = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] == 5]
        P2 = [cons2prim(eq, buf[i]) for i in idx]
        sp = [abs(norm(SVector(v[2], v[3])) - norm(SVector{2, T}(0.5, 0.2))) for v in P2]
        bad = count(>(1e-3), sp) + count(v -> abs(v[4] - 1) > 1e-3, P2)
        κ = [maximum(abs, pg.core.wall_curvatures[i]) for i in idx]
        bad > 0 && for i in idx[findall(j -> sp[j] > 1e-3 || abs(P2[j][4] - 1) > 1e-3, eachindex(idx))][1:min(3, end)]; println("     ghost ", pg.core.positions[i], " d = ", pg.core.wall_distances[i], " n = ", pg.core.normals[i], " curv ", pg.core.wall_curvatures[i]); end
        @printf("%-8s %-14s dx=%-7.4g %4d ghosts: max ||u|-|U|| %.3e, p in [%.3f, %.3f], ghosts with ||u|-|U|| or |p-1| > 1e-3: %d, max stored |curv| %.3e\n",
                T, label, dx, length(idx), maximum(sp), minimum(v -> v[4], P2), maximum(v -> v[4], P2), bad, maximum(κ))
    end
end
