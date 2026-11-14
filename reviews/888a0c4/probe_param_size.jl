using HyperCloudCore, StaticArrays
const H = HyperCloudCore
gm = H.GhostMotion(Val(3), Float64)
println("old: 2 × NTuple{8, GhostMotion{3}} = ", 2 * 8 * sizeof(gm), " B (a tag-7 ball); per extra tag ", 2 * sizeof(gm), " B; 64 tags: ", 2 * 64 * sizeof(gm), " B")
pl = sizeof(SMatrix{3, 3, Float64, 9}) + 2 * sizeof(SVector{3, Float64}); rt = sizeof(SMatrix{3, 3, Float64, 9})
println("new: table 64 B + per piece ", pl + rt, " B -> 1 piece ", 64 + pl + rt, " B, 20 pieces ", 64 + 20 * (pl + rt), " B, 21 pieces ", 64 + 21 * (pl + rt), " B")
println("ghost fit: one GhostMotion{3} = ", sizeof(gm), " B")
