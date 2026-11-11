include("common.jl")
const H = HyperCloudCore
T = Float32; q = 2; n = 10; dx = T(1/n)
geom = get_rectangular_domain(T, (0.,0.,0.), (1.,1.,1.); bc_map = BCD(t => ExtrapolationBC(q) for t in 1:6))
pg = ParticleGrid(geom, (dx,dx,dx), ExponentialWeightFunction(T(1), T(2.5)*dx), 1; randomness = (T(0.2)*dx, T(0.2)*dx, T(0.2)*dx), rng = MersenneTwister(1))
f(x) = 1 + x[1] + x[2] + x[3] + x[1]^2 - x[2]*x[3]
for i in 1:pg.meta.N; pg.rhos[i] = State{1,T}(T(f(Float64.(pg.core.positions[i])))); end
nb = pg.neighbor; isb = pg.core.is_boundary; pos = pg.core.positions
L, Li = pg.domain.L_wrap, pg.domain.invL_wrap
i = 713; x = pos[i]
j, best = H._nearest_interior(i, x, nb.ranges, nb.indices, isb, pos, L, Li)
@show i j
function mine(j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)
    N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
    r = H._min_image(pos[j] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
    N_s += w*(p*p'); b_s += w*(p*buf[j]')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[m]')
    end
    C, ok = H._cholesky_moment(N_s)
    (C \ b_s)[1], N_s, b_s
end
invL = one(T)/minimum(pg.meta.dx)
u1, N1, b1 = mine(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
U, ok = H._fit_degree_b(Val(9), Val(2), j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
@show u1 U ok
C, ok2 = H._cholesky_moment(N1)
@show (C \ b1)[1]
x64 = Float64.(Matrix(N1)) \ Float64.(Matrix(b1)); @show x64[1]
for _ in 1:3
    j2, b2_ = H._nearest_interior(j, x, nb.ranges, nb.indices, isb, pos, L, Li)
    (j2 != 0 && b2_ < best) || break; global j, best = j2, b2_
end
@show j
u1, N1, b1 = mine(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
U, ok = H._fit_degree_b(Val(9), Val(2), j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
@show u1 U ok cond(Float64.(Matrix(N1)))
@show (Float64.(Matrix(N1)) \ Float64.(Matrix(b1)))[1]
@show issymmetric(Matrix(N1))
@show LinearAlgebra.issuccess(cholesky(Symmetric(Matrix(N1)), check=false))
@show LinearAlgebra.issuccess(cholesky(Symmetric(N1), check=false))
@show LinearAlgebra.issuccess(cholesky(Symmetric(SMatrix{10,10,Float64}(N1)), check=false))
@show eigvals(Symmetric(Float64.(Matrix(N1))))
@show eigvals(Symmetric(Matrix(N1)))
@show diag(N1)
A = Matrix(N1); @show maximum(abs, A - A')
pp = H._target_basis(Val(2), SVector{3,Float32}(0.3f0, -1.7f0, 0.9f0)); O = pp*pp'; @show maximum(abs, O - O')
@show pp
function chk(j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)
    N_s = zero(SMatrix{10,10,Float32,100})
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
        O = p*p'
        a1 = maximum(abs, O - O')
        N_s += w*O
        a2 = maximum(abs, N_s - N_s')
        println("m=$m w=$w asymO=$a1 asymN=$a2 r=$r")
    end
end
chk(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
println("--- asym of mine (no prints): ", maximum(abs, Matrix(N1) - Matrix(N1)'))
function mine2(j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)
    N_s = zero(SMatrix{10,10,Float32,100})
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
        N_s += w*(p*p')
    end
    N_s
end
N2 = mine2(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
println("mine2 asym: ", maximum(abs, Matrix(N2) - Matrix(N2)'))
function mine3(j, x, rg, indices, pos, isb, L, Li, wf, invL)
    N_s = zero(SMatrix{10,10,Float32,100})
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL)
        N_s += p*p'
    end
    N_s
end
N3 = mine3(j, x, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
println("mine3 (no weight) asym: ", maximum(abs, Matrix(N3) - Matrix(N3)'))
function mine4(ps)
    N_s = zero(SMatrix{10,10,Float32,100})
    for p in ps
        N_s += p*p'
    end
    N_s
end
ps = [H._target_basis(Val(2), randn(SVector{3,Float32})) for _ in 1:15]
N4 = mine4(ps); println("mine4 random asym: ", maximum(abs, Matrix(N4) - Matrix(N4)'))
ps6 = [H._target_basis(Val(2), randn(SVector{3,Float64})) for _ in 1:15]
function mine5(ps)
    N_s = zero(SMatrix{10,10,Float64,100})
    for p in ps; N_s += p*p'; end
    N_s
end
N5 = mine5(ps6); println("mine5 F64 random asym: ", maximum(abs, Matrix(N5) - Matrix(N5)'))
q1 = [randn(SVector{10,Float32}) for _ in 1:15]; N6 = mine4(q1); println("plain SVector{10,F32} outer sum asym: ", maximum(abs, Matrix(N6) - Matrix(N6)'))
q2 = [randn(SVector{6,Float32}) for _ in 1:15]
function mine6(ps); N_s = zero(SMatrix{6,6,Float32,36}); for p in ps; N_s += p*p'; end; N_s; end
N7 = mine6(q2); println("SVector{6,F32} outer sum asym: ", maximum(abs, Matrix(N7) - Matrix(N7)'))
@show VERSION
function v_a(j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)   # donor + loop, no b
    N_s = zero(SMatrix{10,10,Float32,100})
    r = H._min_image(pos[j] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
    N_s += w*(p*p')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
        N_s += w*(p*p')
    end
    N_s
end
Na = v_a(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
println("donor+loop no b asym: ", maximum(abs, Matrix(Na) - Matrix(Na)'))
function v_b(j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)   # loop + b
    N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[m]')
    end
    N_s
end
Nb = v_b(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
println("loop + b asym: ", maximum(abs, Matrix(Nb) - Matrix(Nb)'))
Nref = zeros(Float64,10,10)
for m in vcat([j],[nb.indices[k] for k in nb.ranges[j] if !isb[nb.indices[k]]])
    r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = nb.weight_func(sum(abs2, r))
    Nref .+= Float64.(Matrix(w*(p*p')))
end
println("max |Na - ref|: ", maximum(abs, Matrix(Na) - Nref), "  max|N1 - ref|: ", maximum(abs, Matrix(N1) - Nref))
println("entries differing (N1 vs ref) > 1e-3: ", findall(abs.(Matrix(N1) - Nref) .> 1e-3))
function v_c(j, x, buf, rg, indices, pos, isb, L, Li, wf, invL)
    N_s = zero(SMatrix{10,10,Float32,100}); b_s = zero(SMatrix{10,1,Float32,10})
    r = H._min_image(pos[j] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
    N_s += w*(p*p'); b_s += w*(p*buf[j]')
    for k in rg
        m = indices[k]; isb[m] && continue
        r = H._min_image(pos[m] - x, L, Li); p = H._target_basis(Val(2), r*invL); w = wf(sum(abs2, r))
        N_s += w*(p*p'); b_s += w*(p*buf[m]')
    end
    N_s, b_s
end
Nc, bc_ = v_c(j, x, pg.rhos, nb.ranges[j], nb.indices, pos, isb, L, Li, nb.weight_func, invL)
println("v_c asym: ", maximum(abs, Matrix(Nc) - Matrix(Nc)'), " vs ref ", maximum(abs, Matrix(Nc) - Nref))
println("b ref diff: ", maximum(abs, Float64.(Matrix(bc_)) - sum(Float64.(Matrix(nb.weight_func(sum(abs2, H._min_image(pos[m]-x,L,Li))) * (H._target_basis(Val(2), H._min_image(pos[m]-x,L,Li)*invL) * pg.rhos[m]'))) for m in vcat([j],[nb.indices[k] for k in nb.ranges[j] if !isb[nb.indices[k]]]))))
