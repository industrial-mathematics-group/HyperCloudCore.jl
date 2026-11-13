# Does the new typed-code test of _cholesky_sym catch a fast-math square root? Copies of the generator with
# (a) Base.FastMath.sqrt_fast, (b) @fastmath sqrt, (c) StaticArrays' cholesky, run through the same check.
using HyperCloudCore, StaticArrays, LinearAlgebra
const HC = HyperCloudCore
check(f, M) = (s = string(Base.code_typed(f, (SMatrix{M, M, Float64, M * M},))[1].first); (occursin("sqrt", s) && !occursin("fast", s), s))
for (name, rootexpr) in (("sqrt", :sqrt), ("sqrt_fast", :(Base.FastMath.sqrt_fast)), ("fastmath", :fm))
    fname = Symbol(:chol_, replace(name, "_" => ""))
    @eval @generated function $fname(A::SMatrix{M, M, T}) where {M, T}
        q = Expr(:block)
        for n in 1:M
            for m in n:M; push!(q.args, :($(Symbol(:L_, m, :_, n)) = @inbounds A[$n, $m])); end
            for k in 1:n-1, m in n:M
                Lmn, Lmk, Lnk = Symbol(:L_, m, :_, n), Symbol(:L_, m, :_, k), Symbol(:L_, n, :_, k)
                push!(q.args, :($Lmn = muladd(-$Lmk, $Lnk, $Lmn)))
            end
            Lnn = Symbol(:L_, n, :_, n)
            push!(q.args, :($Lnn > zero(T) || return Cholesky(zero(SMatrix{$M, $M, T, $(M * M)}), 'U', $n)))
            push!(q.args, $(QuoteNode(rootexpr)) === :fm ? :($Lnn = @fastmath sqrt($Lnn)) : :($Lnn = $($(QuoteNode(rootexpr)))($Lnn)))
            Linv = Symbol(:Linv_, n); push!(q.args, :($Linv = inv($Lnn)))
            for m in n+1:M; push!(q.args, :($(Symbol(:L_, m, :_, n)) *= $Linv)); end
        end
        ret = Expr(:tuple)
        for n in 1:M, m in 1:M; push!(ret.args, m <= n ? Symbol(:L_, n, :_, m) : :(zero(T))); end
        push!(q.args, :(Cholesky(SMatrix{$M, $M, T, $(M * M)}($ret), 'U', 0)))
        return Expr(:block, Expr(:meta, :inline), q)
    end
end
A = SMatrix{3, 3}(4.0, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2)
for (n, f) in (("HyperCloudCore._cholesky_sym", HC._cholesky_sym), ("copy with sqrt", chol_sqrt), ("copy with sqrt_fast", chol_sqrtfast), ("copy with @fastmath sqrt", chol_fastmath),
               ("StaticArrays cholesky (Symmetric not passed)", cholesky))
    pass, s = check(f, 3)
    println(rpad(n, 46), " test passes: ", pass, "   fast-tokens: ", [m.match for m in eachmatch(r"\w*fast\w*", s)] |> unique)
end
println("factors equal (sqrt vs real): ", chol_sqrt(A).factors == HC._cholesky_sym(A).factors, ", sqrt_fast copy: ", chol_sqrtfast(A).factors ≈ HC._cholesky_sym(A).factors)
# the device path is _cholesky_moment: its typed code
s = string(Base.code_typed(HC._cholesky_moment, (SMatrix{4, 4, Float64, 16},))[1].first)
println("_cholesky_moment typed code: sqrt ", occursin("sqrt", s), ", fast ", occursin("fast", s))
