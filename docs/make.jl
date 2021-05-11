using Documenter, RandomExtensions

makedocs(
    sitename="RandomExtensions.jl",
    modules=[RandomExtensions],
    authors="Rafael Fourquet",
)

deploydocs(
    repo = "github.com/JuliaRandom/RandomExtensions.jl.git",
)
