# _cholesky_sym vs StaticArrays' cholesky: bitwise equality on the CPU, failure semantics, allocation, large B
using HyperCloudCore, StaticArrays, LinearAlgebra, Random
const H = HyperCloudCore
rng = Xoshiro(3)
for T in (Float64, Float32), B in (1, 2, 3, 4, 6, 10, 15, 20, 24)
    nbit = 0; nf = 0
    for trial in 1:200
        X = @SMatrix randn(rng, T, B + 3, B)
        A = X' * X
        trial % 4 == 0 && (A = A + T(1e-30) * I)     # nearly singular cases
        trial % 7 == 0 && (A = SMatrix{B, B, T}(Matrix(A) .* (10 .^ (T.(rand(rng, -4:4, B)) * T(1)))) )  # unscaled
        A = (A + A') / 2
        C1 = cholesky(Symmetric(A), check = false); C2 = H._cholesky_sym(A)
        nf += issuccess(C1) != issuccess(C2)
        issuccess(C1) && issuccess(C2) && C1.factors != C2.factors && (nbit += 1)
    end
    println("T=$T B=$B: factor differs bitwise in $nbit/200, success flag differs in $nf/200")
end
# failure cases
for (name, A) in (("zero pivot", SMatrix{2, 2}(0.0, 0.0, 0.0, 1.0)), ("singular 2x2", SMatrix{2, 2}(1.0, 1.0, 1.0, 1.0)),
                  ("NaN", SMatrix{2, 2}(NaN, 0.0, 0.0, 1.0)), ("neg", SMatrix{2, 2}(-1.0, 0.0, 0.0, 1.0)),
                  ("NaN offdiag", SMatrix{2, 2}(1.0, NaN, NaN, 1.0)), ("Inf", SMatrix{2, 2}(Inf, 0.0, 0.0, 1.0)))
    C1 = cholesky(Symmetric(A), check = false); C2 = H._cholesky_sym(A)
    println(rpad(name, 14), " SA: info=$(C1.info) factors=$(C1.factors) | sym: info=$(C2.info) factors=$(C2.factors) pivots_ok=$(H._pivots_ok(C2, A))")
end
# _cholesky_moment on a singular matrix: retry
A = SMatrix{3, 3}(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
C, ok = H._cholesky_moment(A); println("moment of rank-1 3x3: ok=$ok info=$(C.info)")
A0 = zero(SMatrix{3, 3, Float64}); C, ok = H._cholesky_moment(A0); println("moment of zero 3x3: ok=$ok info=$(C.info)")
# allocation
A = let X = @SMatrix randn(rng, 12, 10); X' * X end
f(A) = H._cholesky_sym(A).factors[end]
f(A); println("allocs B=10: ", @allocated f(A))
# large B (3-D degree 4 with intercept: 35)
for B in (28, 35)
    X = @SMatrix randn(rng, B + 5, B); A = X' * X
    t = @elapsed C2 = H._cholesky_sym(A)
    C3 = cholesky(Symmetric(Matrix(A)))
    println("B=$B: first call (compile) $(round(t; digits = 1)) s, rel diff to LAPACK $(maximum(abs, Matrix(C2.factors) - C3.U) / maximum(abs, C3.U)), allocs $(@allocated H._cholesky_sym(A))")
    t1 = @elapsed cholesky(Symmetric(A), check = false); println("   StaticArrays B=$B (heap path) first call $(round(t1; digits = 1)) s, allocs $(@allocated cholesky(Symmetric(A), check = false))")
end
