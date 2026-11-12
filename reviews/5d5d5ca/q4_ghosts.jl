# H3: static ghosts and wall rows behind a receding piston, compared with a cloud generated afresh at the same pose
include("common.jl")
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
function mindist(P)
    m = Inf
    for a in 1:length(P), b in a+1:length(P)
        m = min(m, sum(abs2, P[a] - P[b]))
    end
    sqrt(m)
end
function compare(mkshape, mins, maxs, h, w, nsteps, bcs; label = "", per = (false, false), xcut = x -> true, dt = 0.2h)
    println("== ", label)
    geom = GeometricDomain(mkshape(true); is_periodic = per, mins = mins, maxs = maxs, bc_map = bcs)
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
    ng0 = count(i -> pg.core.is_boundary[i], 1:pg.meta.N)
    t = 0.0
    for _ in 1:nsteps
        t += dt
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        sweep_boundaries!(pg, 0.0, t, 3)
    end
    # fresh cloud at the pose t: the static shape with the piece at t
    geomF = GeometricDomain(at_time(geom.shape, t); is_periodic = per, mins = mins, maxs = maxs, bc_map = bcs)
    pF = ParticleGrid(geomF, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    key(x) = round.(Tuple(x) ./ (h / 1000)) 
    function sets(p, static_only)
        S = Dict{Any, Int}()
        for i in 1:p.meta.N
            x = p.core.positions[i]
            xcut(x) || continue
            if p.core.is_boundary[i]
                p.core.tags[i] <= 4 || continue
            end
            S[(key(x), p.core.is_boundary[i])] = i
        end
        S
    end
    A, B = sets(pg, true), sets(pF, true)
    onlyA = setdiff(keys(A), keys(B)); onlyB = setdiff(keys(B), keys(A))
    @printf("  t=%.3f: compared region: swept %d, fresh %d; only in swept %d (ghosts %d), only in fresh %d (ghosts %d)\n", t, length(A), length(B),
            length(onlyA), count(k -> k[2], onlyA), length(onlyB), count(k -> k[2], onlyB))
    for k in collect(onlyA)[1:min(end, 4)]; println("    swept only: ", pg.core.positions[A[k]], " ghost=", k[2], " tag=", pg.core.tags[A[k]]); end
    for k in collect(onlyB)[1:min(end, 4)]; println("    fresh only: ", pF.core.positions[B[k]], " ghost=", k[2], " tag=", pF.core.tags[B[k]]); end
    # common ghosts: tags, normals, wall distances
    dn, dd, dtag = 0.0, 0.0, 0
    for k in intersect(keys(A), keys(B))
        k[2] || continue
        i, j = A[k], B[k]
        dtag += pg.core.tags[i] != pF.core.tags[j]
        dn = max(dn, norm(pg.core.normals[i] - pF.core.normals[j])); dd = max(dd, abs(pg.core.wall_distances[i] - pF.core.wall_distances[j]))
    end
    @printf("  common static ghosts: tag mismatches %d, max |Δnormal| %.2e, max |Δwall distance| %.2e\n", dtag, dn, dd)
    G = [pg.core.positions[i] for i in 1:pg.meta.N if pg.core.is_boundary[i]]
    @printf("  ghosts %d -> %d; min distance between ghosts %.3f h; min distance between all particles %.3f h\n", ng0, length(G), mindist(G) / h,
            mindist(pg.core.positions[1:pg.meta.N]) / h)
    return pg, t
end
h = 1/20
bcs = BCD(1 => FixedDirichlet(), 2 => FixedDirichlet(), 3 => MirrorBC(; degree = 1), 4 => MirrorBC(; degree = 1), 5 => ExtrapolationBC(1))
pist(w) = mov -> (p = Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5); intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), mov ? Moving(p, translating((w, 0.0))) : p))
pg, t = compare(pist(1.0), (0.0, 0.0), (2.0, 0.5), h, 1.0, 30, bcs; label = "receding plane piston, walls y = 0, 0.5 (h = 1/20, t = 30·0.2h)",
                xcut = x -> x[1] < 1 + 6h - 3h)
# FixedDirichlet data along the bottom wall: new ghosts copy the nearest ghost
bcsD = BCD(t => FixedDirichlet() for t in 1:5)
geom = GeometricDomain(pist(1.0)(true); mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = bcsD)
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
g(x) = 1 + x[1]
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(g(pg.core.positions[i])); end
t = 0.0
for _ in 1:30; global t; t += 0.2h; place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg); sweep_boundaries!(pg, 0.0, t, 3); end
newg = [i for i in 1:pg.meta.N if pg.core.is_boundary[i] && pg.core.tags[i] in (3, 4) && pg.core.positions[i][1] > 1 + 2h]
@printf("FixedDirichlet data g = 1 + x on the walls: %d new wall ghosts, max |U - g| %.3f (h = %.3f)\n", length(newg),
        maximum(abs(pg.rhos[i][1] - g(pg.core.positions[i])) for i in newg; init = 0.0), h)
# curved static wall: a disc cut by a receding piston face
discp(w) = mov -> (p = Tagged(HalfSpace((-0.3, 0.0), (1.0, 0.0)), 5); intersect(Tagged(Ball((0.0, 0.0), 0.8), 1), mov ? Moving(p, translating((w, 0.0))) : p))
compare(discp(1.0), (-0.8, -0.8), (0.8, 0.8), h, 1.0, 20, BCD(1 => MirrorBC(; degree = 1), 5 => ExtrapolationBC(1)); label = "disc (curved static wall) with a receding piston face",
        xcut = x -> x[1] < -0.3 + 4h - 3h)
# oscillating piston: does the ghost count grow without bound?
geom = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5)), Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), t -> RigidMotion(SVector(0.3sin(2π * t), 0.0), SMatrix{2,2,Float64,4}(1, 0, 0, 1), SVector(0.6π * cos(2π * t), 0.0), zero(SMatrix{2,2,Float64,4})))));
                       mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = bcs)
pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0); end
t = 0.0; dt = 0.1h / (0.6π)
counts = Int[]
for k in 1:round(Int, 3 / dt)
    global t; t += dt
    place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg); sweep_boundaries!(pg, 0.0, t, 3)
    abs(t - round(t)) < dt / 2 && push!(counts, count(i -> pg.core.is_boundary[i], 1:pg.meta.N))
end
println("oscillating piston (amplitude 0.3): ghosts at t = 1, 2, 3: ", counts, "; interior ", count(i -> !pg.core.is_boundary[i], 1:pg.meta.N))
