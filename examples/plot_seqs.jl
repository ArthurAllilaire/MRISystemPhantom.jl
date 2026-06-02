# Plots the 2D Cartesian sequence blocks shipped in `sequences/blocks.jl`,
# writing one interactive Plotly HTML per sequence into
# `src/assets/sequences/`. Run with:
#
#   julia --project=. examples/plot_seqs.jl
#
# Each plot shows the RF, gradient and ADC waveforms (use the slider to scrub
# through the shots). The four sequences differ in how they form and weight the
# echo — see `src/sequences/blocks.jl` for the full gradient/k-space convention.

using KomaMRI
using MRISystemPhantom

# Grab PlotlyJS from KomaMRI's namespace (same trick as plot_phantom.jl)
const PlotlyJS = parentmodule(typeof(plot_phantom_map(
    Phantom(x = [0.0]), :T1; height = 10)))

const OUT = joinpath(@__DIR__, "..", "src", "assets", "sequences")
isdir(OUT) || mkpath(OUT)

# Shared imaging geometry: small matrix so the multi-shot structure is easy to
# read in the waveform plot (each shot is one phase-encode line).
FOV = 0.2    # [m]
Nfe = 16     # frequency-encode (readout) samples
Npe = 8      # phase-encode lines / shots
amp = 20e-6  # [T] hard RF pulses

# A spoiler config to demonstrate the crusher/TR-spoiler variant.
spoiler = SpoilerConfig(enabled = true, amp_T = 30e-3, dur = 2e-3, axis = :z)

# (filename, human title, builder) for each sequence to plot.
sequences = [
    ("ir_se_2d",
     "IR-SE 2D (TI/TE/TR = 400/80/1500 ms)",
     ir_se_2d_sequence(400e-3, 80e-3, 1500e-3;
        FOV = FOV, Nfe = Nfe, Npe = Npe, amp_T = amp)),

    ("se_2d",
     "SE 2D — T2 mapping (TE/TR = 80/2000 ms)",
     se_2d_sequence(80e-3, 2000e-3;
        FOV = FOV, Nfe = Nfe, Npe = Npe, amp_T = amp)),

    ("ir_tse_2d",
     "IR-TSE 2D, ETL=4 (TI/ESP/TR = 400/20/1500 ms)",
     ir_tse_2d_sequence(400e-3, 20e-3, 1500e-3;
        etl = 4, FOV = FOV, Nfe = Nfe, Npe = Npe, amp_T = amp)),

    ("gre_2d",
     "GRE 2D, α=15° (TE/TR = 10/100 ms)",
     gre_2d_sequence(10e-3, 100e-3;
        α = deg2rad(15), FOV = FOV, Nfe = Nfe, Npe = Npe, amp_T = amp)),

    ("ir_se_2d_spoiled",
     "IR-SE 2D with Gz crushers + TR spoiler",
     ir_se_2d_sequence(400e-3, 80e-3, 1500e-3;
        FOV = FOV, Nfe = Nfe, Npe = Npe, amp_T = amp, spoiler = spoiler)),
]

for (name, title, seq) in sequences
    p = plot_seq(seq; show_adc = true, slider = true)
    out_path = joinpath(OUT, "$(name).html")
    PlotlyJS.savefig(p, out_path; format = "html")
    @info "Saved" title path = out_path
end
