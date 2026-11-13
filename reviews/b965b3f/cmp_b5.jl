# B5: max |Δ| of the final states between two save dirs (same params, same particle count?)
using Logging; using HyperCloud
global_logger(ConsoleLogger(stderr, Logging.Error))
set_target_module!(Main); HyperCloud.CONFIG_ROOT_OVERWRITE[] = "/workspace/julia/Simulations"
c = Base.include(Main, HyperCloud.resolve_config_path(Main, "piston_channel_2d")); c = Base.invokelatest(Main.get_config)
for m in c.defaults, Ns in c.varied[:Grid_Ns]
    p = HyperCloud.PDEStudioCore.assemble_params(c.shared, c.methods, m); p[:Grid_Ns] = Tuple(Ns)
    r = map(ARGS) do s; set_save_path!(s); load_sim_data(p, :raw); end
    a, b, z = r[1].u[end], r[2].u[end], r[3].u[end]
    f(x, y) = length(x) == length(y) ? string(maximum(i -> maximum(abs.(x[i] - y[i])), eachindex(x))) : "N differs $(length(x)) vs $(length(y))"
    println(rpad(string(m), 26), " ", Ns, "  new vs prev: ", f(a, b), "   new vs parent: ", f(a, z))
end
