using Markdown
src = read(ARGS[1], String)
md = Markdown.parse(src)
for (k, b) in enumerate(md.content)
    b isa Markdown.Table || continue
    println("table $k: rows ", length(b.rows), " cells per row ", length.(b.rows))
    last = b.rows[end]
    println("  last row, cell count ", length(last), "; cell 6+ present: ", length(last) >= 6)
    length(last) >= 6 && println("  cell 6 starts: ", first(Markdown.plain(Markdown.Paragraph(last[6])), 80))
end
# is the "(Up to 0.8" text a paragraph anywhere outside a table?
println("paragraph with 'Up to 0.8 the ghosts got volumes': ", any(b -> b isa Markdown.Paragraph && occursin("Up to 0.8 the ghosts got volumes", Markdown.plain(b)), md.content))
