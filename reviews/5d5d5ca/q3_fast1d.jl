# 1-D fast oscillating piston (|w| dt up to 1 dx): Σ V vs the region, sweeps only (no PDE), step by step
include("common.jl")
oscill(A, ω) = t -> RigidMotion(SVector(A * sin(ω * t)), SMatrix{1,1}(1.0), SVector(A * ω * cos(ω * t)), SMatrix{1,1}(0.0))
for (N, cfl) in ((50, 0.4), (100, 0.4), (200, 0.4), (100, 0.1))
    h = 1 / N
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), oscill(0.1, 8π))
    geom   = GeometricDomain(intersect(Tagged(Cuboid((0.0,), (2.0,)), 1), piston); mins = (0.0,), maxs = (2.0,),
                             bc_map = BCD(1 => FixedDirichlet(), 2 => ExtrapolationBC(1)))
    pg = ParticleGrid(geom, (h,), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(1.0 + pg.core.positions[i][1]); end
    dt = cfl * h; t = 0.0; worst = 0.0; first_bad = -1.0; nan = false
    face(t) = 1 + 0.1sin(8π * t)
    while t < 0.8 - 1e-12
        t += dt
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        sweep_boundaries!(pg, 0.0, t, 3)
        dev = sumV(pg) - face(t)
        abs(dev) > 0.1h && first_bad < 0 && (first_bad = t)
        worst = max(worst, abs(dev))
        nan |= any(i -> isnan(pg.rhos[i][1]) || isnan(pg.core.volumes[i]), 1:pg.meta.N)
    end
    xs = sort([pg.core.positions[i][1] for i in interior(pg)])
    @printf("N=%d cfl=%.1f (max |w|dt = %.2f dx): final ΣV-|Ω| = %.4f (%.2f h), worst %.2f h, first > 0.1h at t=%.4f; max gap %.2fh, NaN %s, V=0 interior %d\n",
            N, cfl, 2.5 * cfl, sumV(pg) - face(t), (sumV(pg) - face(t)) / h, worst / h, first_bad, maximum(diff(xs)) / h, nan,
            count(i -> pg.core.volumes[i] == 0, interior(pg)))
end
