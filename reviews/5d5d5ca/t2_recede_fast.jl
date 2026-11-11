# A receding piece at 0.4 … 3 spacings per sweep (fluid at rest, no PDE: only the sweeps act; the smooth data
# should stay): insertions of one or several columns in one remap round. Error against the data, volumes, ΣV.
include("common.jl")
f2(x) = 1 + 0.5 * sin(2π * x[1]) * cos(4π * x[2])
for per in (false, true), step in (0.4, 1.0, 1.5, 3.0)
    h = 1/40
    piston = Moving(Tagged(HalfSpace((1.0, 0.0), (1.0, 0.0)), 5), translating((1.0, 0.0)))
    box = Cuboid((0.0, 0.0), (2.0, 0.5); open = (false, per))
    geom = GeometricDomain(intersect(box, piston); is_periodic = (false, per), mins = (0.0, 0.0), maxs = (2.0, 0.5),
                           bc_map = BCD(t => ExtrapolationBC(1) for t in 1:5))
    pg = ParticleGrid(geom, (h, h), ExponentialWeightFunction(1.0, 2.5h), 1)
    for i in 1:pg.meta.N; pg.rhos[i] = State{1, Float64}(f2(pg.core.positions[i])); end
    t = 0.0; nan = false; nins = 0
    K = round(Int, 12 / step)
    for k in 1:K
        t += step * h
        place_ghosts!(pg, pg.core.positions, pg.meta.boundary_time, t); pg.neighbor(pg)
        r = sweep_boundaries!(pg, 0.0, t, 3)
        nins += r.inserted
        nan |= any(i -> any(isnan, pg.rhos[i]) || isnan(pg.core.volumes[i]), 1:pg.meta.N)
    end
    inner = interior(pg); face = 1 + t
    unc = [i for i in inner if pg.core.positions[i][1] > 1.0 + 1e-9]
    errs = [abs(pg.rhos[i][1] - f2(pg.core.positions[i])) for i in inner]
    Vu = [pg.core.volumes[i] / h^2 for i in unc]
    Vall = [pg.core.volumes[i] / h^2 for i in inner if 0 < pg.core.positions[i][2] < 0.5]
    @printf("periodic=%s |w|dt=%.1f h, %d sweeps: NaN %s, inserted %d, interior %d, max|U-f| %.3g (uncovered %.3g), V/h² off-wall in [%.3g, %.3g], uncovered [%.3g, %.3g], ΣV %.6f exact %.6f\n",
            per, step, K, nan, nins, length(inner), maximum(errs), maximum(abs(pg.rhos[i][1] - f2(pg.core.positions[i])) for i in unc),
            extrema(Vall)..., extrema(Vu)..., sumV(pg), 0.5 * face)
end
