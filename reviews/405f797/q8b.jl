using HyperCloudCore; const H = HyperCloudCore
ctx = Dict{Symbol, Any}(:T => Float32, :D => 2)
for s in ("r = 0.5f0; Ball((0f0, 0f0), r)", "r = T(0.5); Ball((T(0), T(0)), r)")
    try
        g = H.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => s), ctx); println(s, ": ok")
    catch e
        println(s, ": ", first(split(sprint(showerror, e), '\n')))
    end
end
