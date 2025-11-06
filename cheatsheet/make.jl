using Documenter, ReactionDiffusion

makedocs(
    sitename="ReactionDiffusion.jl",
    format = Documenter.HTML(),
    doctest =false,
    clean=true,
    modules=[ReactionDiffusion],
    checkdocs=:none,
    pages=["Cheatsheet" => "index.md"]
)