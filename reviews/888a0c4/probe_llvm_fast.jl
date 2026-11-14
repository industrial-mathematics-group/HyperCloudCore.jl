# A CPU-checkable regression test for d916089: no `afn`/`fast` square root in the LLVM IR of the fits
using HyperCloudCore, StaticArrays, LinearAlgebra, InteractiveUtils
const H = HyperCloudCore
sa(A) = cholesky(Symmetric(A), check = false).factors
hc(A) = H._cholesky_sym(A).factors
mo(A) = H._cholesky_moment(A)[1].factors
for (n, f) in (("StaticArrays", sa), ("_cholesky_sym", hc), ("_cholesky_moment", mo))
    ir = sprint(code_llvm, f, (SMatrix{4, 4, Float64, 16},))
    println(rpad(n, 18), " fast sqrt calls in the IR: ", length(collect(eachmatch(r"call (fast|afn)[^\n]*@llvm\.sqrt", ir))),
            ", plain sqrt calls: ", length(collect(eachmatch(r"call double @llvm\.sqrt", ir))))
end
