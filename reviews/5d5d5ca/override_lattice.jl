# PROBE-ONLY override (this process only; the repository is untouched): `_sweep_lattice` with a round-off tolerance
# on the upper end of the box index range, to test whether the dropped outermost ghost row causes the wall-row error.
@eval HyperCloudCore function _sweep_lattice(f, hg::ParticleGrid{D, M, T}, geom, box) where {D, M, T}
    cd, dx = hg.domain, hg.meta.dx
    wall_lo = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.mins[d], cd.canvas_mins[d], dx[d]), Val(D))
    wall_hi = ntuple(d -> cd.is_periodic[d] ? -1 : _wall_index(geom.maxs[d], cd.canvas_mins[d], dx[d]), Val(D))
    lo, hi = box
    rng = ntuple(d -> ceil(Int, (lo[d] - cd.canvas_mins[d]) / dx[d] - T(1e-6)):floor(Int, (hi[d] - cd.canvas_mins[d]) / dx[d] + T(1e-6)), Val(D))
    for I in CartesianIndices(rng)
        x = Space{D, T}(ntuple(Val(D)) do d
            j = I[d]
            j == wall_lo[d] ? geom.mins[d] : j == wall_hi[d] ? geom.maxs[d] : cd.canvas_mins[d] + j * dx[d]
        end)
        x = _wrap_point(hg, x)
        all(cd.canvas_mins .- T(1e-6) * dx .<= x .<= cd.canvas_maxs .+ T(1e-6) * dx) && f(x)
    end
    return nothing
end
