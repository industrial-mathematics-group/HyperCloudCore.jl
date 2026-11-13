# Max deviation from the uniform state (ρ = p = 1, u = (0.5, 0)) of the X1/X4 runs: julia eval_uniform.jl SAVE
using Logging, Printf
using HyperCloud, StaticArrays
global_logger(ConsoleLogger(stderr, Logging.Error))
const ROOT = "/home/agent/.claude/jobs/a4b4be7c/tmp/rr7/probes/simroot"
set_target_module!(Main); HyperCloud.CONFIG_ROOT_OVERWRITE[] = ROOT
set_save_path!(ARGS[1])
const γ = 1.4
prim(U) = (ρ = U[1]; u = SVector(U[2], U[3]) / ρ; (ρ, u, (γ - 1) * (U[4] - ρ * sum(abs2, u) / 2)))
for name in ("x_disc_comoving_2d", "x_disc_comoving_spin_2d")
    c = Base.include(Main, HyperCloud.resolve_config_path(Main, name)); c = Base.invokelatest(Main.get_config)
    for m in c.defaults, Ns in c.varied[:Grid_Ns]
        p = HyperCloud.PDEStudioCore.assemble_params(c.shared, c.methods, m); p[:Grid_Ns] = Tuple(Ns); haskey(ENV, "BND_T") && (p[:T] = Float32)
        does_sim_data_exist(p, Val(:raw)) || (println("$name $m $Ns: no data"); continue)
        d = load_sim_data(p, :raw); P = [prim(u) for u in d.u[end]]
        du = maximum(q -> sqrt(sum(abs2, q[2] - SVector(0.5, 0.0))), P); dr = maximum(q -> abs(q[1] - 1), P); dp = maximum(q -> abs(q[3] - 1), P)
        @printf("%-26s %-26s %-10s t=%.2f  max |u - U| %.2e  max |ρ - 1| %.2e  max |p - 1| %.2e\n", name, m, join(Ns, "x"), d.t[end], du, dr, dp)
    end
end
