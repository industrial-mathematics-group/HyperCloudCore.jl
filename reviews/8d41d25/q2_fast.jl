# A piece that moves more than the clearance per sweep: zeroed volumes (L3), NaN in the remaps?
include("common.jl")
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
for (D, per) in ((2, true), (2, false))
  for step in (0.5, 1.0, 1.5, 3.0)
    h = 1/40
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((-1.0, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, per))
    geom = GeometricDomain(intersect(box, piston); is_periodic = (false, per), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
    t = 0.0; nan = false; zeroV = 0; err = 0.0; ex = (Inf, -Inf)
    for k in 1:6
        t += step * h
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        try
            sweep_boundaries!(pg, 0.0, t, 3)
        catch e
            println("  step=$step: sweep threw ", sprint(showerror, e)[1:min(end, 200)]); break
        end
        inner = interior(pg)
        nan |= any(i -> any(isnan, pg.rhos[i]) || isnan(pg.core.volumes[i]), 1:pg.meta.N)
        zeroV = count(i -> pg.core.volumes[i] <= 0, inner)
        err = maximum(abs(pg.rhos[i][1] - f2(pg.core.positions[i])) for i in inner)
        ex = extrema(pg.rhos[i][1] for i in inner)
    end
    @printf("  periodic=%s |w|dt=%.1f h: NaN %s, interior with V<=0 %d, max |U-f| %.3g, U in [%.4f, %.4f], ΣV %.6f exact %.6f\n",
            per, step, nan, zeroV, err, ex..., sumV(pg), 0.5 * (1 - t))
  end
end
