using HyperCloudCore, StaticArrays, LinearAlgebra, Serialization
const H = HyperCloudCore
xs, ys, x, invL, wf = deserialize("/tmp/brev_b/repro/data.jls")
# exactly the library's _fit_degree_b shape, with plain vectors
function fitA(x, xs, ys, wf, invL)
    N_s = zero(SMatrix{10, 10, Float32, 100}); b_s = zero(SMatrix{10, 1, Float32, 10})
    @inbounds for m in eachindex(xs)
        r = xs[m] - x
        p = H._intercept_basis(Val(2), r * invL)
        w = wf(sum(abs2, r))
        N_s += w * (p * p')
        b_s += w * (p * ys[m]')
    end
    C, ok = H._cholesky_moment(N_s)
    c = C \ b_s
    return c[1, 1], maximum(abs, N_s - N_s')
end
# same, Cholesky without the pivot check / regularisation
function fitB(x, xs, ys, wf, invL)
    N_s = zero(SMatrix{10, 10, Float32, 100}); b_s = zero(SMatrix{10, 1, Float32, 10})
    @inbounds for m in eachindex(xs)
        r = xs[m] - x
        p = H._intercept_basis(Val(2), r * invL)
        w = wf(sum(abs2, r))
        N_s += w * (p * p')
        b_s += w * (p * ys[m]')
    end
    C = cholesky(Symmetric(N_s), check = false)
    c = C \ b_s
    return c[1, 1], maximum(abs, N_s - N_s')
end
function fitN(x, xs, ys, wf, invL)
    N_s = zero(SMatrix{10, 10, Float32, 100})
    @inbounds for m in eachindex(xs)
        r = xs[m] - x
        p = H._intercept_basis(Val(2), r * invL); w = wf(sum(abs2, r))
        N_s += w * (p * p')
    end
    N_s
end
N = fitN(x, xs, ys, wf, invL)
println("plain accumulation symmetric: ", N == N', "  F64 solve of it: ", (Float64.(Matrix(N)) \ Float64.(Matrix(sum(wf(sum(abs2, xs[m]-x)) * (H._intercept_basis(Val(2), (xs[m]-x)*invL) * ys[m]') for m in eachindex(xs)))))[1])
println("fitA (with _cholesky_moment): ", fitA(x, xs, ys, wf, invL))
println("fitB (plain cholesky):        ", fitB(x, xs, ys, wf, invL))
