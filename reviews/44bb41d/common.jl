using HyperCloudCore, StaticArrays, Random, LinearAlgebra, Printf
import HyperCloudCore: flux, velocity, max_eigenvalue, prim2cons, cons2prim, fluid_velocity, param2vel
include("la.jl")
const BCD = Dict{Int, AbstractBoundaryCondition}
# monomial exponents up to degree q in D dims
function exps(D, q)
    out = NTuple{D,Int}[]
    for I in Iterators.product(ntuple(_ -> 0:q, D)...)
        sum(I) <= q && push!(out, I)
    end
    out
end
function randpoly(D, q, rng)
    E = exps(D, q); c = randn(rng, length(E))
    x -> sum(c[k] * prod(x[d]^E[k][d] for d in 1:D; init = 1.0) for k in eachindex(E))
end
ghosts(pg) = findall(view(pg.core.is_boundary, 1:pg.meta.N))
interior(pg) = findall(i -> !pg.core.is_boundary[i], 1:pg.meta.N)
mirrorpt(pg, i) = pg.core.positions[i] - 2 * pg.core.wall_distances[i] * pg.core.normals[i]
function applybc(pg, M, T)
    buf = copy(pg.rhos)
    for i in ghosts(pg); buf[i] = State{M,T}(ntuple(_->T(99), M)); end
    for (tag, bc) in pg.geometry.bc_map
        bc(pg, buf, tag, nothing, nothing, 0.0)
    end
    buf
end
