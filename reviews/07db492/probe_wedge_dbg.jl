# Trace the repeated projection of wall_geometry for one wedge point.
using HyperCloudCore, StaticArrays, LinearAlgebra
const HC = HyperCloudCore
for (α, x) in ((120, SVector(-0.0430284162628018, -0.0744851078750326)), (15, SVector(-0.227446326115545, -0.011352029915703943)), (45, SVector(-0.046188308319805714, -0.018255729001285293)))
    a = deg2rad(α); nrm = SVector(-sin(a), cos(a))
    shape = intersect(Cuboid((-5.0, 0.0), (5.0, 5.0)), HalfSpace((0.0, 0.0), Tuple(nrm)))
    φ, n, _ = HC._probe(shape, x); p = x
    println("α=$α x=$x exact dist to corner ", norm(x), "; wall_geometry ", HC._wall_geometry(shape, x))
    for k in 1:9
        φ, n, _ = HC._probe(shape, p)
        println("   step $k: p=$p φ=$φ n=$n")
        p = p - φ * n
    end
end
