# String :shape parsing after 8d41d25
include("common.jl")
ctx(T) = Dict{Symbol, Any}(:T => T, :D => 2)
for (lbl, s) in (("one expression", "Ball((T(0), T(0)), T(0.5))"),
                 ("two statements ;", "r = T(0.5); Ball((T(0), T(0)), r)"),
                 ("trailing ;", "Ball((T(0), T(0)), T(0.5));"),
                 ("newline separated", "r = T(0.5)\nBall((T(0), T(0)), r)"),
                 ("begin block", "begin r = T(0.5); Ball((T(0), T(0)), r) end"),
                 ("comment", "Ball((T(0), T(0)), T(0.5))  # a disc"),
                 ("function + Moving", "f(t) = t; Moving(Tagged(Ball((T(0), T(0)), T(0.1)), 5), translating((T(1), T(0))))"),
                 ("incomplete", "Ball((T(0), T(0)), "),
                 ("syntax error", "Ball((T(0), T(0)),, T(1))"),
                 ("empty", ""),
                 ("only a comment", "# nothing"),
                 ("using inside", "using LinearAlgebra; Ball((T(0), T(0)), T(0.5))"),
                 ("wrong float type", "Ball((0.0, 0.0), 0.5)"))
    for T in (Float32, Float64)
        msg = try
            g = H.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => s), ctx(T))
            "ok: $(typeof(g.shape).name.name){$(eltype(g.mins))}"
        catch e
            "$(typeof(e)): " * first(replace(sprint(showerror, e), "\n" => " | "), 160)
        end
        @printf("%-18s %-8s %s\n", lbl, T, msg)
    end
end
println("Meta.parse of incomplete: ", Meta.parse("Ball((T(0), T(0)), "; raise = false) |> x -> (x isa Expr ? x.head : typeof(x)))
println("Meta.parse of syntax error (raise=false): ", Meta.parse("Ball((T(0), T(0)),, T(1))"; raise = false) |> x -> (x isa Expr ? x.head : typeof(x)))
