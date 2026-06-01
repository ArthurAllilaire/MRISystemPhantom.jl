using KomaMRI
using MRISystemPhantom

# Grab PlotlyJS from KomaMRI's namespace (same trick as plot_phantom.jl)
const PlotlyJS = parentmodule(typeof(plot_phantom_map(
    Phantom(x = [0.0]), :T1; height = 10)))

const OUT = joinpath(@__DIR__, "..", "src", "assets")
isdir(OUT) || mkpath(OUT)

# Build the sequence with typical E2 parameters
TI = 400e-3   # [s]
TE = 80e-3    # [s]
TR = 1500e-3  # [s]

seq = ir_se_2d_sequence(TI, TE, TR;
    FOV   = 0.2,
    Nfe   = 16,
    Npe   = 8,
    amp_T = 20e-6)

p = plot_seq(seq; show_adc=true, slider=true)

out_path = joinpath(OUT, "ir_se_2d_sequence.html")
PlotlyJS.savefig(p, out_path; format = "html")
@info "Saved" path = out_path
