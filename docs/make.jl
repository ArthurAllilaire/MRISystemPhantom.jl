using Documenter
using Literate
using MRISystemPhantom

# Generate example pages from scripts in examples/
examples_src = joinpath(@__DIR__, "..", "examples")
examples_out = joinpath(@__DIR__, "src", "examples")

Literate.markdown(joinpath(examples_src, "t1_mapping.jl"),      examples_out;
                  execute = false, flavor = Literate.CommonMarkFlavor())
Literate.markdown(joinpath(examples_src, "snr_calibration.jl"), examples_out;
                  execute = false, flavor = Literate.CommonMarkFlavor())

makedocs(
    sitename = "MRISystemPhantom.jl",
    authors  = "Arthur Allilaire",
    modules  = [MRISystemPhantom],
    format   = Documenter.HTML(
        prettyurls       = get(ENV, "CI", nothing) == "true",
        canonical        = "https://arthurallilaire.github.io/MRISystemPhantom.jl",
        edit_link        = "main",
        assets           = String[],
    ),
    pages = [
        "Home"            => "index.md",
        "Getting Started" => "getting_started.md",
        "Guides" => [
            "Phantom Construction" => "phantom.md",
            "Pulse Sequences"      => "sequences.md",
            "Parameter Fitting"    => "fitting.md",
            "SNR Diagnostics"      => "diagnostics.md",
        ],
        "Examples" => [
            "T1 Mapping"        => "examples/t1_mapping.md",
            "SNR Calibration"   => "examples/snr_calibration.md",
        ],
        "API Reference"   => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo   = "github.com/ArthurAllilaire/MRISystemPhantom.jl.git",
    branch = "gh-pages",
    push_preview = true,
)
