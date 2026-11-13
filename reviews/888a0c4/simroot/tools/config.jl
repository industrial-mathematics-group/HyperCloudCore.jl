# Shared by run_sim.jl and plot_sim.jl: loads experiment ARGS[1] and applies SIM_T / SIM_SET / SIM_METHODS.
const NAME = ARGS[1]
path = HyperCloud.resolve_config_path(Main, NAME)
config = Base.include(Main, path)
config isa Function && (config = Base.invokelatest(config))
config isa SimulationConfig || (config = Base.invokelatest(Main.get_config))

T = eval(Meta.parse(get(ENV, "SIM_T", "Float32")))
config.shared[:T] = T
for assignment in split(get(ENV, "SIM_SET", ""), ';'; keepempty = false)
    k, v = strip.(split(assignment, '='; limit = 2))
    config.shared[Symbol(k)] = Core.eval(Main, Meta.parse(v))
end
if haskey(ENV, "SIM_METHODS")
    empty!(config.defaults)
    append!(config.defaults, Symbol.(strip.(split(ENV["SIM_METHODS"], ';'; keepempty = false))))
end

function label(p)
    l = "$(get(p, :Time_stepper, "?"))$(get(p, :Scheme_name, ""))$(get(p, :Scheme_order, ""))"
    get(p, :Limiter_name, :none) == :none || (l *= "($(p[:Limiter_name]))")
    get(p, :MOOD_criterion, :none) == :none || (l *= "($(p[:MOOD_strategy]),$(p[:MOOD_criterion]))")
    haskey(p, :Scheme_MLS_order) && (l *= " MLS=$(p[:Scheme_MLS_order])")
    return l * " N=$(join(p[:Grid_Ns], "x"))" * " CFL=$(get(p, :Time_CFL, "-"))" * " R=$(get(p, :Weight_range, "-"))"
end

# SIM_VARY = "Time_CFL=[0.1, 0.2]; Grid_Ns=[(100, 100)]" replaces the sweep of the config
if haskey(ENV, "SIM_VARY")
    empty!(config.varied)
    for assignment in split(ENV["SIM_VARY"], ';'; keepempty = false)
        k, v = strip.(split(assignment, '='; limit = 2))
        config.varied[Symbol(k)] = collect(Core.eval(Main, Meta.parse(v)))
    end
end

# SIM_MSET = "Time_stepper=:RK3" overrides parameters of the selected methods (method entries win over shared ones)
for assignment in split(get(ENV, "SIM_MSET", ""), ';'; keepempty = false)
    k, v = strip.(split(assignment, '='; limit = 2))
    for m in config.defaults
        haskey(config.methods, m) && (config.methods[m][Symbol(k)] = Core.eval(Main, Meta.parse(v)))
    end
end
