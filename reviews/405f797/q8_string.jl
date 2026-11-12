# 405f797: `let T = ...` around a string :shape
using HyperCloudCore
const H = HyperCloudCore
ctx64 = Dict{Symbol, Any}(:T => Float64, :D => 2)
ctx32 = Dict{Symbol, Any}(:T => Float32, :D => 2)
function try_shape(s, ctx)
    try
        g = H.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => s), ctx)
        println(rpad(repr(s), 75), " T=", ctx[:T], ": ok ", typeof(g.shape).name.name, "{", eltype(g.shape), "}")
    catch e
        msg = sprint(showerror, e)
        println(rpad(repr(s), 75), " T=", ctx[:T], ": ERROR ", first(split(msg, '\n'))[1:min(end, 160)])
    end
end
for s in ("Ball((T(0), T(0)), T(1))",
          "Ball((0.0, 0.0), 1.0)",
          "Ball((0f0, 0f0), 1f0)",
          "r = T(0.5); Ball((T(0), T(0)), r)",                               # two statements
          "begin r = T(0.5); Ball((T(0), T(0)), r) end",
          "let T = Float64; Ball((T(0), T(0)), T(1)) end",                    # the user's own T
          "Cuboid((T(-1), T(-1)), (T(1), T(1))) - Moving(Tagged(Ball((T(0), T(0)), T(0.2)), 5), translating((T(1), T(0))))",
          "Cuboid((-1.0, -1.0), (1.0, 1.0)) - Moving(Tagged(Ball((0.0, 0.0), 0.2), 5), rotating(1.0))",
          "Ball((T(0), T(0)), T(1)",                                           # syntax error
          "UndefinedShape(1)")
    for ctx in (ctx64, ctx32)
        try_shape(s, ctx)
    end
end
# does evaluating the string leak globals into Main now / before?
H.build_domain(Dict{Symbol, Any}(:domain => :shape, :shape => "begin myr = 0.5; Ball((0.0, 0.0), myr) end"), ctx64)
println("isdefined(Main, :myr) after a string with an assignment: ", isdefined(Main, :myr))
# a user global named T in Main
@eval Main T = 3
try_shape("Ball((0.0, 0.0), 1.0 * T)", ctx64)
