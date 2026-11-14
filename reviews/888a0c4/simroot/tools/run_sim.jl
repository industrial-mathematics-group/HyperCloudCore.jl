# Runs one experiment of experiments/ (like `@sim NAME`) on a chosen backend / precision, then prints
# a health summary of every run (final time, NaNs, min density / pressure, mass drift, runtime).
#
#     julia -t 6 --project=env_gpu tools/run_sim.jl NAME                       (AMD GPU)
#     SIM_BACKEND=cuda julia -t 6 --project=env_cuda tools/run_sim.jl NAME   (NVIDIA GPU)
#
# Environment variables:
#   SIM_BACKEND = gpu | cuda | cpu   (default gpu: KABackend(ROCBackend()), cuda: KABackend(CUDABackend()) via CUDACore,
#                                     cpu: PolyesterBackend())
#   SIM_T       = Float32 | Float64  (default Float32; stored as :T in the run parameters)
#   SIM_METHODS = "m1;m2"            replaces the config's default methods (';' since names contain ',')
#   SIM_VARY    = "Time_CFL=[0.1, 0.2]; Grid_Ns=[(100,100)]"  replaces the sweep of the config
#   SIM_SET     = "Time_CFL=0.4; Grid_Ns=(200,200)"   overrides of shared parameters (Julia literals)
#   SIM_SAVE    = directory for the result files (default results/)
#   SIM_FORCE   = 1 reruns cached runs
#   SIM_SKIP_RUN = 1 only prints the summary of the cached runs
#   SIM_PATCHES = "tools/patches/x.jl;…"  test patches loaded after HyperCloud (paths relative to the repo). The run
#                 parameters do not record them, so use a separate SIM_SAVE for patched runs.
using Logging, Printf
using HyperCloud, StaticArrays
const BACKEND = get(ENV, "SIM_BACKEND", "gpu")
BACKEND in ("gpu", "cuda", "cpu") || error("SIM_BACKEND must be gpu, cuda or cpu, got $BACKEND")
const GPU = BACKEND != "cpu"
BACKEND == "gpu"  && @eval using AMDGPU
BACKEND == "cuda" && @eval using CUDACore

# Flushes after every message, so progress shows up in a log file (stderr to a file is block-buffered)
struct FlushingLogger <: AbstractLogger
    inner::ConsoleLogger
end
Logging.min_enabled_level(l::FlushingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::FlushingLogger, args...) = Logging.shouldlog(l.inner, args...)
Logging.catch_exceptions(l::FlushingLogger) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::FlushingLogger, args...; kwargs...)
    Logging.handle_message(l.inner, args...; kwargs...)
    flush(stderr)
end
global_logger(FlushingLogger(ConsoleLogger(stderr, Logging.Info)))
set_target_module!(Main)
set_save_path!(get(ENV, "SIM_SAVE", joinpath(@__DIR__, "..", "results")))
set_stat_preset!("hyperbolic")
HyperCloud.CONFIG_ROOT_OVERWRITE[] = joinpath(@__DIR__, "..")
set_progress_interval!(parse(Float64, get(ENV, "SIM_PROGRESS", "60")))
enable_progress_bar!(false)

if BACKEND == "gpu"
    AMDGPU.allowscalar(false)
    set_backend!(KABackend(ROCBackend()))
elseif BACKEND == "cuda"
    CUDACore.allowscalar(false)
    set_backend!(KABackend(CUDABackend()))
else
    set_backend!(PolyesterBackend())
    # Float32 neighbour sums underflow to subnormals, which x86 cores process very slowly
    for _ in 1:20
        HyperCloud.HyperCloudCore.@batch per=core for i in 1:10_000; set_zero_subnormals(true); end
    end
    set_zero_subnormals(true)
end

for f in split(get(ENV, "SIM_PATCHES", ""), ';'; keepempty = false)
    @warn "Loading test patch $f (not recorded in the run parameters)"
    include(joinpath(@__DIR__, "..", strip(f)))
end

include(joinpath(@__DIR__, "config.jl"))
@info "Running $NAME on $(GPU ? "GPU" : "CPU") with T = $T, methods $(config.defaults)"

get(ENV, "SIM_SKIP_RUN", "0") == "1" ||
    Base.invokelatest(() -> run_all_simulations(config; force_overwrite = get(ENV, "SIM_FORCE", "0") == "1",
                                                calculate_stats = true))

# ---- health summary ----------------------------------------------------------------------------
# Returns (ok, line): a run is ok if it reached tmax without NaNs or huge values (Euler: positive density and pressure).
function summary_line(params)
    does_sim_data_exist(params, Val(:raw)) || return (false, @sprintf("FAIL %-40s NO DATA (run failed)", label(params)))
    d   = load_sim_data(params, :raw)
    u   = d.u[end]
    nan = count(s -> any(isnan, s), u)
    rho = minimum(s -> s[1], u)
    ok   = nan == 0 && d.t[end] >= params[:tmax] * (1 - 1e-4) && (params[:PDE_name] != :euler || rho > 0)
    # blown up without NaNs: relative to the largest value of the initial data (an absolute bound flags e.g. the
    # energy density 1.6e6 of the Mach 2000 jet)
    ok  &= maximum(s -> maximum(abs, s), u) < 1e4 * max(1, maximum(s -> maximum(abs, s), d.u[1]))
    line = @sprintf("%-40s t=%.4g/%.4g  NaN=%d  min(u1)=%.3e", label(params), d.t[end], params[:tmax], nan, rho)
    if params[:PDE_name] == :euler
        γ = params[:PDE_gamma]
        p(s) = (γ - 1) * (s[end] - sum(abs2, s[2:end-1]) / (2s[1]))
        pmin, ip = findmin(p, u)
        ok &= pmin > 0
        line *= @sprintf("  min(p)=%.3e at x=(%s)", pmin, join((@sprintf("%.3f", c) for c in d.x[end][ip]), ","))
    end
    st = d.stats
    if haskey(st, :mass)
        m = st[:mass]
        m0, m1 = first(m)[1], last(m)[1]
        line *= @sprintf("  mass drift=%.2e", (m1 - m0) / abs(m0))
    end
    haskey(st, :k_step)  && (line *= @sprintf("  steps=%d", st[:k_step][1]))
    haskey(st, :Runtime) && (line *= @sprintf("  runtime=%.1fs", st[:Runtime][1]))
    haskey(st, :l1error) && (line *= @sprintf("  L1err(u1)=%.3e", last(st[:l1error])[1]))
    return ok, (ok ? "OK   " : "FAIL ") * line
end

println("\n==== SUMMARY $NAME (T=$T, $(GPU ? "GPU" : "CPU")) ====")
all_ok = true
for task in generate_tasks(config)
    try
        ok, line = summary_line(task)
        global all_ok &= ok
        println(line)
    catch e
        global all_ok = false
        println("FAIL ", rpad(label(task), 40), " summary failed: ", sprint(showerror, e))
    end
end
exit(all_ok ? 0 : 2)
