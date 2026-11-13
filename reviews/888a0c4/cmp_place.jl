using Serialization, LinearAlgebra, StaticArrays
a = deserialize(ARGS[1]); b = deserialize(ARGS[2])
for (k, n) in ((1, "positions"), (2, "normals"), (3, "curvatures"))
    println("888a0c4 vs 025b0df $n: ", a[k] == b[k] ? "bitwise equal" : maximum(norm.(a[k] .- b[k])))
end
