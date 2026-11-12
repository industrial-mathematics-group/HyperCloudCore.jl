using Markdown
for f in ARGS
    md = Markdown.parse(read(f, String))
    for b in md.content
        if b isa Markdown.List
            items = [first(Markdown.plain(Markdown.MD(it)), 50) for it in b.items]
            any(s -> occursin("No ghosts close", s) || occursin("The fit is centred", s), items) && println(basename(dirname(dirname(dirname(dirname(f))))), ": list start ", b.ordered, " items: ", items)
        elseif b isa Markdown.Paragraph
            s = Markdown.plain(b); occursin("The fit is centred", s) && println("  'The fit is centred' inside a paragraph: ", first(s, 60))
        end
    end
end
