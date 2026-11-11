# mins/maxs smaller than the shape's bounding box: silently cropped region?
using HyperCloudCore, StaticArrays
g = GeometricDomain(Ball((0.0, 0.0), 1.0); mins = (0.0, 0.0), maxs = (1.0, 1.0))
pg = ParticleGrid(g, (0.05, 0.05), ExponentialWeightFunction(1.0, 0.125), 1)
P = view(pg.core.positions, 1:pg.meta.N)
int = [P[i] for i in 1:pg.meta.N if !pg.core.is_boundary[i]]
println("accepted; interior particles with x < 0: ", count(p -> p[1] < 0, int), ", min x of interior: ", minimum(p -> p[1], int),
        ", ghosts with x < 0 near y=0.5: ", count(i -> pg.core.is_boundary[i] && P[i][1] < 0 && abs(P[i][2] - 0.5) < 0.1, 1:pg.meta.N),
        ", domain_volume ", domain_volume(pg), " (full disc; region generated is a quarter plus a band)")
