# Shared setup of the boundary tests (experiments/Boundary, reports/2026-11-12-boundary-tests.md): physical tests of the
# wall treatment of HyperCloudCore/HyperCloud main_dev (shape domains, MirrorBC/ExtrapolationBC ghosts, moving pieces,
# SlipWall/DiffuseWall operators), run in env_bnd / env_bnd_cuda. Included by every config of this folder with
# `isdefined(@__MODULE__, :BND_METHODS) || Base.include(@__MODULE__, joinpath(@__DIR__, "boundary_common.jl"))`.
#
# Methods (labels of experiments/methods_review.jl): RK2MUSCL2(VK,AR,MLS2), RK3MUSCL3(EPD1,AR,MLS2),
# RK3MUSCL3(EPD2,AR,MLS2) for the problems with shocks; unlimited RK3MUSCL3(none,AR,MLS2) for the smooth ones.
# Exact solutions are factories `params -> (xt -> State)` (xt = (x..., t)), named exact_* so that PDEStudioCore treats
# them as references; tools/boundary_check.jl evaluates them on the particles.
isdefined(@__MODULE__, :add_review_methods!) || Base.include(@__MODULE__, joinpath(@__DIR__, "..", "methods_review.jl"))
using HyperCloud, StaticArrays

const BND_BCD = Dict{Int, AbstractBoundaryCondition}
const BND_METHODS = ["RK2MUSCL2(VK,AR,MLS2)", "RK3MUSCL3(EPD1,AR,MLS2)", "RK3MUSCL3(EPD2,AR,MLS2)"]
const BND_METHODS_MD = ["RK2MUSCL2(VK,AR,MLS2)", "RK3MUSCL3(EPD1,AR,MLS2)"]          # 2-D / 3-D (cost)
const BND_SMOOTH_METHODS = ["RK3MUSCL3(none,AR,MLS2)", "RK2MUSCL2(none,AR,MLS2)"]
const BND_GAMMA = 1.4

bnd_methods() = add_review_methods!(get_master_method_dict(); orders = 2:3)

"Conserved Euler state (ρ, ρu…, E) of density, velocity (tuple) and pressure."
bnd_cons(ρ, u, p; γ = BND_GAMMA) = (Float64(ρ), (Float64(ρ * ui) for ui in u)..., Float64(p / (γ - 1) + ρ * sum(abs2, u) / 2))

"Shared parameters of a D-dimensional Euler boundary test; pairs override or add entries."
function bnd_shared(D::Int, overrides::Pair...)
    shared = Dict{Symbol, Any}(
        :PDE_name => :euler, :PDE_D => D, :PDE_gamma => BND_GAMMA, :T => Float64,
        :sim_func_name => :run_direct_simulation, :snapshots => 10,
        :Grid_periodic => false, :Grid_randomness_factor => ntuple(_ -> 0.2, D), :Grid_seed => 7, :Grid_mover => :none,
        :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
        :Time_CFL => 0.2,
    )
    for (k, v) in overrides
        shared[k] = v
    end
    return shared
end

"A uniform state as the `:riemann` IC (both sides equal)."
bnd_uniform(D, U) = (:IC_name => :riemann, :IC_uL => U, :IC_uR => U, :IC_p0 => ntuple(_ -> 0.0, D),
                     :IC_n => ntuple(d -> d == 1 ? 1.0 : 0.0, D))

bnd_config(shared, methods_list; varied = Dict{Symbol, Vector}(), reference = nothing, sources = String[]) =
    SimulationConfig(shared, bnd_methods(), [Symbol(m) for m in methods_list]; varied = varied, reference = reference,
                     sources = vcat(sources, joinpath(@__DIR__, "boundary_common.jl")))

# -------------------------------------------------------------------------------------------------------------------
# Exact solutions
# -------------------------------------------------------------------------------------------------------------------
_bnd_state(D, ρ, u1, p) = State{D + 2, Float64}(ρ, ρ * u1, ntuple(_ -> 0.0, D - 1)..., p / (BND_GAMMA - 1) + ρ * u1^2 / 2)
_bnd_prim1(U) = (U[1], U[2] / U[1], (BND_GAMMA - 1) * (U[end] - sum(abs2, U[2:end-1]) / (2U[1])))

"""
    exact_wall_reflection(params)

A gas `(ρ₁, u₀ > 0, p₁)` (the `:IC_uL` state) moving along x against a slip wall at `x = :wall_x`: a shock reflected
from the wall at t = 0 leaves the gas at rest behind it, at the Rankine–Hugoniot state of a piston moving at u₀ into a
gas at rest (in the wall frame the shock runs at S − u₀ away from the wall). Planar, so valid in 1-, 2- and 3-D.
"""
function exact_wall_reflection(params::Dict)
    D = params[:PDE_D]
    ρ₁, u₀, p₁ = _bnd_prim1(Float64.(collect(params[:IC_uL])))
    xw = Float64(params[:wall_x])
    ρ₂, p₂, S = HyperCloud.piston_state(ρ₁, p₁, u₀, BND_GAMMA)
    U₁, U₂ = _bnd_state(D, ρ₁, u₀, p₁), _bnd_state(D, ρ₂, 0.0, p₂)
    return xt -> xt[1] < xw - (S - u₀) * xt[end] ? U₁ : U₂
end

"""
    exact_piston_planar(params)

A piston face starting at `:piston_face` and moving at `−:piston_speed` along x into a gas at rest (`:IC_uL`):
the shock at `face − S t`, behind it the Rankine–Hugoniot state moving with the piston. HyperCloud's `exact_piston` in D
dimensions.
"""
function exact_piston_planar(params::Dict)
    D = params[:PDE_D]
    ρ₁, _, p₁ = _bnd_prim1(Float64.(collect(params[:IC_uL])))
    w, xf = Float64(params[:piston_speed]), Float64(params[:piston_face])
    ρ₂, p₂, S = HyperCloud.piston_state(ρ₁, p₁, w, BND_GAMMA)
    U₁, U₂ = _bnd_state(D, ρ₁, 0.0, p₁), _bnd_state(D, ρ₂, -w, p₂)
    return xt -> xt[1] < xf - S * xt[end] ? U₁ : U₂
end

"""
    exact_receding_piston(params)

A piston face starting at `:piston_face` and receding at `+:piston_speed` from a gas at rest on its left: a centred
rarefaction from (face, 0) into the gas, head at face − c₀t, tail at face + (w − c_t)t with c_t = c₀ − (γ−1)w/2, the gas
between tail and face moving with the piston (u = w, c = c_t, isentropic). Needs w < 2c₀/(γ−1).
"""
function exact_receding_piston(params::Dict)
    D, γ = params[:PDE_D], BND_GAMMA
    ρ₀, _, p₀ = _bnd_prim1(Float64.(collect(params[:IC_uL])))
    w, xf = Float64(params[:piston_speed]), Float64(params[:piston_face])
    c₀ = sqrt(γ * p₀ / ρ₀)
    w < 2c₀ / (γ - 1) || error("exact_receding_piston: cavitation (w ≥ 2c₀/(γ−1))")
    function state(u, c)
        ρ = ρ₀ * (c / c₀)^(2 / (γ - 1))
        return _bnd_state(D, ρ, u, p₀ * (ρ / ρ₀)^γ)
    end
    ct = c₀ - (γ - 1) * w / 2
    return xt -> begin
        x, t = xt[1], xt[end]
        t <= 0 && return state(0.0, c₀)
        ξ = (x - xf) / t
        ξ <= -c₀ && return state(0.0, c₀)
        ξ >= w - ct && return state(w, ct)
        u = 2 * (ξ + c₀) / (γ + 1)
        return state(u, c₀ - (γ - 1) * u / 2)
    end
end

# Acoustic pulse: p' = ε exp(−((x − x₀)/σ)²), right-going (u' = p'/(ρ₀c₀), ρ' = p'/c₀²), reflected by a slip wall at
# x = :wall_x (image pulse left-going). Linear acoustics: exact up to O(ε²).
_bnd_pulse(params, x) = Float64(params[:IC_eps]) * exp(-((x - Float64(params[:IC_x0])) / Float64(params[:IC_sigma]))^2)

"""
    exact_acoustic_wall(params)

The right-going acoustic pulse of the `:bnd_acoustic` IC (gas `ρ₀ = 1`, `p₀ = 1`) and its image reflected by the slip
wall at `:wall_x`, superposed (linear acoustics).
"""
function exact_acoustic_wall(params::Dict)
    D, γ = params[:PDE_D], BND_GAMMA
    c₀ = sqrt(γ)
    xw = Float64(params[:wall_x])
    return xt -> begin
        x, t = xt[1], xt[end]
        pr = _bnd_pulse(params, x - c₀ * t)              # incident, right-going
        pl = _bnd_pulse(params, 2xw - x - c₀ * t)        # image, left-going
        p′ = pr + pl; u′ = (pr - pl) / c₀
        ρ = 1 + p′ / c₀^2
        return _bnd_state(D, ρ, u′, 1 + p′)
    end
end

struct BndAcousticIC{D, T} <: HyperCloud.InitialCondition
    eps::T; x0::T; sigma::T
end
function (ic::BndAcousticIC{D, T})(pos) where {D, T}
    γ = T(BND_GAMMA); c₀ = sqrt(γ)
    p′ = ic.eps * exp(-((pos[1] - ic.x0) / ic.sigma)^2)
    ρ = 1 + p′ / c₀^2; u = p′ / c₀
    return State{D + 2, T}(ρ, ρ * u, ntuple(_ -> zero(T), D - 1)..., (1 + p′) / (γ - 1) + ρ * u^2 / 2)
end
HyperCloud.build_ic(::Val{:bnd_acoustic}, conf::Dict, ctx::Dict) =
    BndAcousticIC{ctx[:D], ctx[:T]}(ctx[:T](conf[:eps]), ctx[:T](conf[:x0]), ctx[:T](conf[:sigma]))

"""
    oscillating_from_rest(a, ω, T) -> motion

The 1-D motion `x(t) = −a (1 − cos ωt)` (velocity `−aω sin ωt`, starting from rest, so the radiated wave has no jump at
its front) in the float type `T`, for `Moving` pieces in shape strings.
"""
function oscillating_from_rest(a, ω, ::Type{T}) where {T}
    A, W = T(a), T(ω)
    I1, Z1 = SMatrix{1, 1, T, 1}(one(T)), zero(SMatrix{1, 1, T, 1})
    return t -> RigidMotion(Space{1, T}(-A * (1 - cos(W * T(t)))), I1, Space{1, T}(-A * W * sin(W * T(t))), Z1)
end

"""
    exact_oscillating_piston(params)

Linear acoustics of a piston face at `x_f − a (1 − cos ωt)` (`:piston_face`, `:osc_a`, `:osc_omega`) on a gas at rest
(ρ₀ = p₀ = 1) on its left: the left-going wave u(x, t) = v_p(t − (x_f − x)/c₀), p' = −ρ₀c₀u, ρ' = p'/c₀², zero ahead of
the front. Exact up to O(a ω/c₀) (amplitude) and O(a ω/c₀) (the face's displacement, linearised at x_f).
"""
function exact_oscillating_piston(params::Dict)
    D, γ = params[:PDE_D], BND_GAMMA
    c₀ = sqrt(γ)
    xf, a, ω = Float64(params[:piston_face]), Float64(params[:osc_a]), Float64(params[:osc_omega])
    return xt -> begin
        x, t = xt[1], xt[end]
        τ = t - (xf - x) / c₀
        u = τ > 0 ? -a * ω * sin(ω * τ) : 0.0
        p′ = -c₀ * u
        return _bnd_state(D, 1 + p′ / c₀^2, u, 1 + p′)
    end
end

# Oblique shock of a supersonic flow (Mach M along x) at a compression corner of angle θ: the weak solution of the
# θ-β-M relation and the Rankine–Hugoniot state behind the shock (flow parallel to the ramp).
function bnd_oblique_shock(M, θ; γ = BND_GAMMA)
    f(β) = 2cot(β) * (M^2 * sin(β)^2 - 1) / (M^2 * (γ + cos(2β)) + 2) - tan(θ)
    lo, hi = asin(1 / M) + 1e-9, asin(1 / M) + 1e-9
    βs = range(asin(1 / M) + 1e-6, π / 2 - 1e-6, length = 20001)
    k = findfirst(i -> f(βs[i]) <= 0 < f(βs[i + 1]) || f(βs[i]) >= 0 > f(βs[i + 1]), 1:length(βs) - 1)
    k === nothing && error("no attached oblique shock for M = $M, θ = $θ")
    lo, hi = βs[k], βs[k + 1]
    for _ in 1:100
        mid = (lo + hi) / 2
        sign(f(mid)) == sign(f(lo)) ? (lo = mid) : (hi = mid)
    end
    β = (lo + hi) / 2
    Mn1 = M * sin(β)
    pr = 1 + 2γ / (γ + 1) * (Mn1^2 - 1)
    ρr = (γ + 1) * Mn1^2 / ((γ - 1) * Mn1^2 + 2)
    Mn2 = sqrt((1 + (γ - 1) / 2 * Mn1^2) / (γ * Mn1^2 - (γ - 1) / 2))
    M2 = Mn2 / sin(β - θ)
    return (; β, pr, ρr, M2)
end

"""
    exact_wedge(params)

Steady supersonic flow (`:IC_uL`, along x) over a ramp of angle `:wedge_angle` (degrees) starting at x = `:wedge_x` on
the floor y = 0: free stream ahead of the attached oblique shock from the corner, the Rankine–Hugoniot state parallel
to the ramp behind it (valid away from the outflow boundaries, once the start-up has left the domain).
"""
function exact_wedge(params::Dict)
    ρ₁, u₁, p₁ = _bnd_prim1(Float64.(collect(params[:IC_uL])))
    θ = deg2rad(Float64(params[:wedge_angle])); xc = Float64(params[:wedge_x])
    M = u₁ / sqrt(BND_GAMMA * p₁ / ρ₁)
    s = bnd_oblique_shock(M, θ)
    ρ₂, p₂ = ρ₁ * s.ρr, p₁ * s.pr
    q₂ = s.M2 * sqrt(BND_GAMMA * p₂ / ρ₂)
    U₁ = State{4, Float64}(bnd_cons(ρ₁, (u₁, 0.0), p₁)...)
    U₂ = State{4, Float64}(bnd_cons(ρ₂, (q₂ * cos(θ), q₂ * sin(θ)), p₂)...)
    return xt -> (xt[1] <= xc || xt[2] > (xt[1] - xc) * tan(s.β)) ? U₁ : U₂
end

# -------------------------------------------------------------------------------------------------------------------
# Tests added 2026-11-13 (retest of HyperCloudCore #45 / HyperCloud #20)
# -------------------------------------------------------------------------------------------------------------------
"""
    bnd_channel_reflection(M₁, θ) -> (; s1, M2, s2, ρ2r, p2r, ρ3r, p3r)

Regular reflection of the oblique shock of a θ ramp at a parallel wall: the incident shock turns the flow by θ (region 2),
the reflected shock turns it back by θ (region 3, parallel to the wall again). Density and pressure ratios to region 1.
"""
function bnd_channel_reflection(M₁, θ; γ = BND_GAMMA)
    s1 = bnd_oblique_shock(M₁, θ; γ)
    s2 = bnd_oblique_shock(s1.M2, θ; γ)
    return (; s1, M2 = s1.M2, s2, ρ2r = s1.ρr, p2r = s1.pr, ρ3r = s1.ρr * s2.ρr, p3r = s1.pr * s2.pr)
end

"""
    exact_channel_reflection(params)

Supersonic flow (`:IC_uL`) in a channel of height `:channel_height` over a ramp of `:wedge_angle` degrees from
x = `:wedge_x`: the incident shock from the corner reflects regularly at the top slip wall (impingement at
x_i = x_c + H / tan β₁); region 1 ahead of the incident shock, region 2 between the shocks, region 3 behind the
reflected shock (which leaves the wall at β₂ − θ). Valid upstream of the reflected shock's impact on the ramp.
"""
function exact_channel_reflection(params::Dict)
    ρ₁, u₁, p₁ = _bnd_prim1(Float64.(collect(params[:IC_uL])))
    θ = deg2rad(Float64(params[:wedge_angle])); xc = Float64(params[:wedge_x]); H = Float64(params[:channel_height])
    M = u₁ / sqrt(BND_GAMMA * p₁ / ρ₁)
    r = bnd_channel_reflection(M, θ)
    β1, β2 = r.s1.β, r.s2.β
    xi = xc + H / tan(β1)
    ρ₂, p₂ = ρ₁ * r.ρ2r, p₁ * r.p2r; q₂ = r.M2 * sqrt(BND_GAMMA * p₂ / ρ₂)
    ρ₃, p₃ = ρ₁ * r.ρ3r, p₁ * r.p3r; q₃ = r.s2.M2 * sqrt(BND_GAMMA * p₃ / ρ₃)
    U₁ = State{4, Float64}(bnd_cons(ρ₁, (u₁, 0.0), p₁)...)
    U₂ = State{4, Float64}(bnd_cons(ρ₂, (q₂ * cos(θ), q₂ * sin(θ)), p₂)...)
    U₃ = State{4, Float64}(bnd_cons(ρ₃, (q₃, 0.0), p₃)...)
    return xt -> begin
        x, y = xt[1], xt[2]
        (x <= xc || y > (x - xc) * tan(β1)) && return U₁
        (x <= xi || y < H - (x - xi) * tan(β2 - θ)) && return U₂
        return U₃
    end
end

# Potential vortex between two concentric static walls: u_θ = Γ/r, isentropic (p = ρ^γ/γ, c = 1 and ρ = 1 at
# r = :vortex_rout), the pressure from Bernoulli. A steady solution of the Euler equations; slip walls at both radii.
function _bnd_vortex(params, x, y)
    Γ, Ro = Float64(params[:vortex_gamma]), Float64(params[:vortex_rout]); γ = BND_GAMMA
    r = hypot(x, y); uθ = Γ / r
    Hb = 1 / (γ - 1) + (Γ / Ro)^2 / 2                  # c² /(γ−1) + u²/2 at the outer wall (c = 1)
    c2 = (γ - 1) * (Hb - uθ^2 / 2)
    ρ = c2^(1 / (γ - 1)); p = ρ * c2 / γ
    return ρ, (-uθ * y / r, uθ * x / r), p
end
"""
    exact_annulus_vortex(params)

The steady potential vortex `u_θ = :vortex_gamma / r` between static walls (isentropic, Bernoulli); the run starts from it.
"""
exact_annulus_vortex(params::Dict) = xt -> begin
    ρ, u, p = _bnd_vortex(params, xt[1], xt[2])
    State{4, Float64}(bnd_cons(ρ, u, p)...)
end
struct BndVortexIC{T} <: HyperCloud.InitialCondition
    gamma::Float64; rout::Float64
end
function (ic::BndVortexIC{T})(pos) where {T}
    ρ, u, p = _bnd_vortex(Dict(:vortex_gamma => ic.gamma, :vortex_rout => ic.rout), Float64(pos[1]), Float64(pos[2]))
    return State{4, T}(T.(bnd_cons(ρ, u, p))...)
end
HyperCloud.build_ic(::Val{:bnd_vortex}, conf::Dict, ctx::Dict) = BndVortexIC{ctx[:T]}(Float64(conf[:gamma]), Float64(conf[:rout]))

"Complementary error function (Numerical Recipes Chebyshev fit, relative error < 1.2e-7; no extra dependency)."
function bnd_erfc(x::Real)
    z = abs(Float64(x)); t = 1 / (1 + z / 2)
    r = t * exp(-z^2 - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return x >= 0 ? r : 2 - r
end

"""
    exact_stokes_layer(params)

Stokes' first problem: a gas at rest (ρ = 1, θ = 1) above a wall that starts sliding at `:wall_speed` along x at t = 0.
Near the Euler limit the Navier–Stokes solution u(y, t) = U erfc(y / (2√(νt))) with the BGK viscosity ν = τ p/ρ = τθ
(`:PDE_tau`); density and temperature stay constant to O(U²). The BGK gas has a velocity slip at the wall of O(Kn).
"""
function exact_stokes_layer(params::Dict)
    U, ν = Float64(params[:wall_speed]), Float64(params[:PDE_tau])
    return xt -> begin
        y, t = xt[2], xt[end]
        u = t > 0 ? U * bnd_erfc(y / (2sqrt(ν * t))) : 0.0
        State{4, Float64}(bnd_cons(1.0, (u, 0.0), 1.0)...)
    end
end
