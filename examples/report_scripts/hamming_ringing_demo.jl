# Report figure: the 1-D mechanism behind apodisation. Companion to
# hamming_window_response.jl (window weights + frequency response) and
# hamming_apodisation.jl (effect on the phantom). Reconstructs a single point
# source and a step edge from a finite (N-line) k-space, box vs Hamming.
#
# The left panel IS the PSF (delta * PSF = PSF); the right panel is the step edge
# convolved with that same PSF — so panel (a) explains panel (b):
#   * Point source (left): box reconstructs a Dirichlet/sinc PSF whose nulls fall
#     exactly on integer pixel offsets — markers show neighbours = 0 (no leakage at
#     sample points). Hamming's wider main lobe is nonzero at ±1 px ⇒ it blurs.
#   * Step edge (right): box side-lobes fail to cancel across the discontinuity ⇒
#     Gibbs ringing. Hamming's −43 dB side-lobes suppress it, at the cost of a
#     softer (wider) edge.
#
# Writes one PNG to src/assets/report/.
# Run:  julia --project=. examples/report_scripts/hamming_ringing_demo.jl

using MRISystemPhantom, KomaMRI, FFTW

const PlotlyJS = parentmodule(typeof(plot_phantom_map(Phantom(x = [0.0]), :T1; height = 10)))
const OUT = joinpath(@__DIR__, "..", "..", "src", "assets", "report")
isdir(OUT) || mkpath(OUT)

const N   = 32                      # acquired k-space lines (matches the other scripts)
const OS  = 16                      # oversampling, to draw the continuous reconstruction
const NF  = N * OS                  # fine grid length
const C_BOX, C_HAM, C_TRUE = "#1f77b4", "#d62728", "#444444"

xfine = (0:NF-1) ./ OS              # pixel coordinate 0 .. N (step 1/OS)
ham_w = 0.54 .- 0.46 .* cos.(2π .* (0:N-1) ./ (N - 1))

# Reconstruct a fine-grid object from only its central N k-space coefficients
# (box truncation = acquiring N lines), optionally apodised. This is just a
# frequency mask + standard inverse, so the amplitude of the input object is
# preserved (a unit step reconstructs to a unit plateau, with Gibbs overshoot).
function reconstruct(o; window = nothing)
    O  = fftshift(fft(o))
    c  = div(NF, 2) + 1
    lo = c - div(N, 2); hi = lo + N - 1
    acq = O[lo:hi]
    window === nothing || (acq = acq .* window)
    K = zeros(ComplexF64, NF)
    K[lo:hi] = acq
    real.(ifft(ifftshift(K)))
end

# ---- Point source on a pixel centre (x = N/2) ----
# A single-sample delta gives the exact sinc PSF (nulls land on integer pixels).
# Its absolute amplitude is arbitrary, so normalise both curves to the box peak.
o_pt   = zeros(NF); o_pt[div(NF, 2) + 1] = 1.0
pt_box = reconstruct(o_pt)
pt_ham = reconstruct(o_pt; window = ham_w)
pk     = maximum(pt_box); pt_box ./= pk; pt_ham ./= pk
xoff   = xfine .- N / 2                       # offset from the source, in pixels
midx   = 1:OS:NF                              # integer-pixel sample points

# ---- Step edge at x = N/2 ----
o_step  = Float64.(xfine .>= N / 2)
st_box  = reconstruct(o_step)
st_ham  = reconstruct(o_step; window = ham_w)

fig = PlotlyJS.make_subplots(rows = 1, cols = 2,
    subplot_titles = ["(a) Point source" "(b) Step edge"], horizontal_spacing = 0.12)

# Panel (a): PSFs, with integer-pixel markers on the box curve (= the nulls).
PlotlyJS.add_trace!(fig, PlotlyJS.scatter(; x = xoff, y = pt_box, mode = "lines",
    name = "Box (no window)", legendgroup = "box", line = Dict(:width => 2, :color => C_BOX)),
    row = 1, col = 1)
PlotlyJS.add_trace!(fig, PlotlyJS.scatter(; x = xoff, y = pt_ham, mode = "lines",
    name = "Hamming", legendgroup = "ham", line = Dict(:width => 2, :color => C_HAM)),
    row = 1, col = 1)
PlotlyJS.add_trace!(fig, PlotlyJS.scatter(; x = xoff[midx], y = pt_box[midx], mode = "markers",
    name = "pixel samples (box)", legendgroup = "mark",
    marker = Dict(:size => 6, :color => C_BOX, :symbol => "circle-open")), row = 1, col = 1)

# Panel (b): the edge, true vs both reconstructions.
PlotlyJS.add_trace!(fig, PlotlyJS.scatter(; x = xfine, y = o_step, mode = "lines",
    name = "true edge", legendgroup = "true",
    line = Dict(:width => 1.5, :dash => "dash", :color => C_TRUE)), row = 1, col = 2)
PlotlyJS.add_trace!(fig, PlotlyJS.scatter(; x = xfine, y = st_box, mode = "lines",
    legendgroup = "box", showlegend = false, line = Dict(:width => 2, :color => C_BOX)),
    row = 1, col = 2)
PlotlyJS.add_trace!(fig, PlotlyJS.scatter(; x = xfine, y = st_ham, mode = "lines",
    legendgroup = "ham", showlegend = false, line = Dict(:width => 2, :color => C_HAM)),
    row = 1, col = 2)

PlotlyJS.relayout!(fig; width = 980, height = 440,
    margin = Dict(:t => 60, :l => 70, :r => 30, :b => 95),
    title = Dict(:text => "Why edges ring: finite-kspace reconstruction (N = $N), box vs Hamming",
                 :x => 0.5, :xanchor => "center", :font => Dict(:size => 17)),
    legend = Dict(:orientation => "h", :x => 0.5, :y => -0.22,
                  :xanchor => "center", :yanchor => "top"),
    xaxis  = Dict(:title => Dict(:text => "pixel offset from source"),
                  :range => [-6.5, 6.5], :tick0 => 0, :dtick => 1, :zeroline => false),
    yaxis  = Dict(:title => Dict(:text => "amplitude (norm.)"), :zeroline => true),
    xaxis2 = Dict(:title => Dict(:text => "pixel position"),
                  :range => [N / 2 - 7, N / 2 + 7], :zeroline => false),
    yaxis2 = Dict(:title => Dict(:text => "amplitude (norm.)"), :zeroline => false))

out = joinpath(OUT, "hamming_ringing_demo.png")
PlotlyJS.savefig(fig, out)
@info "Saved" path = out
