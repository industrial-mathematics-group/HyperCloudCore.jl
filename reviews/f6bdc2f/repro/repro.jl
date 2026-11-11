using StaticArrays, LinearAlgebra, Random
@inline basis(r::SVector{3,T}) where {T} = SVector{10,T}(one(T), r[1], r[2], r[3], r[1]^2/2, r[2]^2/2, r[3]^2/2, r[1]*r[2], r[1]*r[3], r[2]*r[3])
function fit(xs::Vector{SVector{3,T}}, ys::Vector{T}) where {T}
    N_s = zero(SMatrix{10,10,T,100}); b_s = zero(SMatrix{10,1,T,10})
    for k in eachindex(xs)
        p = basis(xs[k]); w = one(T) / (one(T) + sum(abs2, xs[k]))
        N_s += w * (p * p'); b_s += w * (p * SVector(ys[k])')
    end
    C = cholesky(Symmetric(N_s), check = false)
    c = C \ b_s
    return c[1, 1], N_s
end
function fitN(xs::Vector{SVector{3,T}}) where {T}
    N_s = zero(SMatrix{10,10,T,100})
    for k in eachindex(xs)
        p = basis(xs[k]); w = one(T) / (one(T) + sum(abs2, xs[k]))
        N_s += w * (p * p')
    end
    return N_s
end
for T in (Float32, Float64)
    rng = MersenneTwister(1); nbad = 0; maxasym = zero(T); maxerr = 0.0
    for trial in 1:1000
        xs = [SVector{3,T}(randn(rng, 3)) .+ SVector{3,T}(1, 1, 1) for _ in 1:16]
        f(x) = 1 + x[1] - 2x[2] + x[3] + x[1]^2 - x[2] * x[3]
        ys = T[f(x) for x in xs]
        v, N = fit(xs, ys)
        Nref = fitN(xs)
        a = maximum(abs, N - N')
        maxasym = max(maxasym, a); maxerr = max(maxerr, abs(v - 1))
        (a > 0 || maximum(abs, N - Nref) > 0) && (nbad += 1)
    end
    println("$T: trials with N_s differing from the plain accumulation: $nbad / 1000, max asym $maxasym, max |value - exact| $maxerr")
end
