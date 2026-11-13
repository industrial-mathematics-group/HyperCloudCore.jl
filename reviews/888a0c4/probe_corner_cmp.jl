using Serialization, Printf
a, b = deserialize(ARGS[1]), deserialize(ARGS[2]); h = parse(Float64, ARGS[3])
sa, sb = Set(a), Set(b)
println("parent only: ", length(setdiff(sa, sb)), ", review only: ", length(setdiff(sb, sa)))
for (lab, s) in (("parent only", setdiff(sa, sb)), ("review only", setdiff(sb, sa)))
    for (x, isb, tag) in sort(collect(s))
        @printf("  %-12s x = (%.5f, %.5f)  boundary %s tag %d  (x - 1)/h = %.3f\n", lab, x[1], x[2], isb, tag, (x[1] - 1) / h)
    end
end
