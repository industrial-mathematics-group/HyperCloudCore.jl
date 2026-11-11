using Serialization, LinearAlgebra
o = deserialize(ARGS[1]); n = deserialize(ARGS[2])
function mind(P)
    m = Inf
    for i in 1:length(P), j in i+1:length(P)
        m = min(m, norm(P[i] - P[j]))
    end
    m
end
for k in sort(collect(keys(n)))
    haskey(o, k) || (println(rpad(k, 34), " (new only) Nint=", count(!, n[k].isb), " minghostφ=", minimum(n[k].wd[n[k].isb]; init=Inf)); continue)
    a, b = o[k], n[k]
    so = Set(a.pos); sn = Set(b.pos)
    added = [i for i in 1:length(b.pos) if !(b.pos[i] in so)]
    removed = [i for i in 1:length(a.pos) if !(a.pos[i] in sn)]
    idx_o = Dict(p => i for (i, p) in enumerate(a.pos))
    # retagged: same position, boundary flag changed
    flip = count(i -> haskey(idx_o, b.pos[i]) && a.isb[idx_o[b.pos[i]]] != b.isb[i], 1:length(b.pos))
    println(rpad(k, 34), " N ", length(a.pos), "->", length(b.pos), " Nint ", count(!, a.isb), "->", count(!, b.isb),
            " new interior ", count(i -> !b.isb[i], added), " new ghosts ", count(i -> b.isb[i], added), " removed ", length(removed),
            " (old ghosts: ", all(i -> a.isb[i], removed), ", old φ max ", isempty(removed) ? 0 : round(maximum(a.wd[removed]), sigdigits=3), ")",
            " flips ", flip,
            " min ghost φ ", round(minimum(a.wd[a.isb]; init=Inf), sigdigits=3), "->", round(minimum(b.wd[b.isb]; init=Inf), sigdigits=3),
            " min spacing ", round(mind(a.pos), sigdigits=3), "->", round(mind(b.pos), sigdigits=3))
end
