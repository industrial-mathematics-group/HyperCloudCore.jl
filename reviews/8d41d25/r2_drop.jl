# _drop_empty!: which particles does the region pass zero, where are they (piece distance), and is the sweep conservative?
include("common.jl")
using Random
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
function regionpass(pg, t)
    pc = deepcopy(pg)
    cd, R, dx = pc.domain, pc.meta.R, minimum(pc.meta.dx)
    boxes = map(pc.geometry.pieces) do piece
        lo0, hi0 = bounding_box(at_time(piece, pc.meta.sweep_time)); lo1, hi1 = bounding_box(at_time(piece, t))
        (max.(min.(lo0, lo1) .- (R + 2dx), cd.canvas_mins), min.(max.(hi0, hi1) .+ (R + 2dx), cd.canvas_maxs))
    end
    m0 = sum(pc.core.volumes[i] * pc.rhos[i][1] for i in interior(pc))
    H._update_region!(pc, pc.geometry, pc.meta.sweep_time, t, boxes, 2)
    m1 = sum(pc.core.volumes[i] * pc.rhos[i][1] for i in interior(pc))
    return pc, m1 - m0
end
function run(label, geom, h, steps, dtw; jit = 0.0, nsw = 8)
    pg = jit == 0 ? ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1) :
                    ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1; randomness = (jit * h, jit * h), rng = MersenneTwister(3))
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
    println("== ", label)
    t = 0.0
    tot_drop, far_drop, mind = 0, 0, Inf
    for k in 1:nsw
        t += dtw
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        pc, dm_region = regionpass(pg, t)
        posed = map(p -> at_time(p, t), pg.geometry.pieces)
        zs = [i for i in 1:pc.meta.N if !pc.core.is_boundary[i] && pc.core.volumes[i] == 0]
        # piece distance of the zeroed particles (negative = inside the region, i.e. not covered)
        ds = [minimum(p -> signed_distance(p, pc.core.positions[i]), posed) for i in zs]   # piece sd: >0 outside the piece body? (sign per piece)
        inr = [H._is_interior(pg.geometry, pc.core.positions[i], t) for i in zs]
        tiny = count(i -> !pc.core.is_boundary[i] && 0 < pc.core.volumes[i] < 1e-3 * h^2, 1:pc.meta.N)
        tot_drop += length(zs); far_drop += count(inr)
        isempty(zs) || (mind = min(mind, minimum(abs.(ds)) / h))
        m0 = sum(pg.core.volumes[i] * pg.rhos[i][1] for i in interior(pg)); V0 = sumV(pg)
        r = sweep_boundaries!(pg, 0.0, t, 3)
        m1 = sum(pg.core.volumes[i] * pg.rhos[i][1] for i in interior(pg))
        nan = any(i -> isnan(pg.rhos[i][1]) || isnan(pg.core.volumes[i]), 1:pg.meta.N)
        @printf("  sweep %d: zeroed %d (still interior at t: %d; |piece sd|/h of zeroed in [%s]), tiny V %d, -%d +%d, mass_residual %.3e = Δm %.3e; region-pass Δm %.3e, remaps %.1e; NaN %s\n",
                k, length(zs), count(inr), isempty(ds) ? "-" : join(string.(round.(extrema(ds) ./ h, digits = 2)), ", "), tiny, r.removed, r.inserted,
                r.mass_residual[1], m1 - m0, dm_region, (m1 - m0) - dm_region, nan)
    end
    @printf("  total zeroed %d, of which still in the region %d\n", tot_drop, far_drop)
end
h = 1/40
pist(w; per = false) = GeometricDomain(intersect(Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, per)), Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((w, 0.0))));
                                       is_periodic = (false, per), mins = (0.0, 0.0), maxs = (2.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
for s in (1.5, 3.0)
    run("advancing piston, walls, |w|dt = $s h", pist(-1.0), h, 0, s * h)
end
run("advancing piston, 45% jitter, |w|dt = 0.2 h", pist(-1.0), h, 0, 0.2h; jit = 0.45, nsw = 12)
run("advancing piston, 45% jitter, |w|dt = 1.5 h", pist(-1.0), h, 0, 1.5h; jit = 0.45, nsw = 6)
# rotating plate (fast tip) in a periodic channel
plate = Moving(Tagged(Cuboid((-0.3, -0.03), (0.3, 0.03)), 5), rotating(1.0; center = (0.0, 0.0)))
chan = Cuboid((-1.0, -0.5), (1.0, 0.5); open = (true, false)) - plate
gpl = GeometricDomain(chan; is_periodic = (true, false), mins = (-1.0, -0.5), maxs = (1.0, 0.5), bc_map = BCD(t => ExtrapolationBC(1) for t in 3:5))
run("rotating plate, tip speed 0.3, Δθ per sweep giving 1.5 h at the tip", gpl, h, 0, 1.5h / 0.3)
run("rotating plate, 45% jitter, tip 0.2 h per sweep", gpl, h, 0, 0.2h / 0.3; jit = 0.45, nsw = 12)
