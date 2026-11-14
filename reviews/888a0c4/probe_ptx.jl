# Compile the Cholesky factorisations to PTX (no GPU needed) and list the square-root instructions
using NVPTX_LLVM_Backend_jll, GPUCompiler, LLVM, StaticArrays, LinearAlgebra, HyperCloudCore
module TestRuntime
    signal_exception() = return
    malloc(sz) = Ptr{Cvoid}(Int(0xDEADBEEF))
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end
struct P <: GPUCompiler.AbstractCompilerParams end
GPUCompiler.runtime_module(::GPUCompiler.CompilerJob{<:Any, P}) = TestRuntime
function ptx(f, tt)
    src = GPUCompiler.methodinstance(typeof(f), Base.to_tuple_type(tt), Base.get_world_counter())
    target = PTXCompilerTarget(; cap = v"7.0")
    job = GPUCompiler.CompilerJob(src, GPUCompiler.CompilerConfig(target, P(); kernel = false))
    io = IOBuffer()
    GPUCompiler.code_native(io, job)
    String(take!(io))
end
sa(A)  = cholesky(Symmetric(A), check = false).factors
hc(A)  = HyperCloudCore._cholesky_sym(A).factors
fsq(x) = Base.FastMath.sqrt_fast(x)
for (name, f, tt) in (("StaticArrays cholesky 4x4 F64", sa, (SMatrix{4, 4, Float64, 16},)),
                      ("_cholesky_sym 4x4 F64", hc, (SMatrix{4, 4, Float64, 16},)),
                      ("StaticArrays cholesky 4x4 F32", sa, (SMatrix{4, 4, Float32, 16},)),
                      ("_cholesky_sym 4x4 F32", hc, (SMatrix{4, 4, Float32, 16},)),
                      ("sqrt_fast F64", fsq, (Float64,)))
    s = ptx(f, tt)
    ins = [m.match for m in eachmatch(r"(sqrt|rsqrt|rcp|div)\.[a-z0-9.]+", s)]
    println(rpad(name, 34), " => ", join(sort(unique(ins)), ", "), "  (counts: ", join(["$(u)×$(count(==(u), ins))" for u in sort(unique(ins))], ", "), ")")
end
