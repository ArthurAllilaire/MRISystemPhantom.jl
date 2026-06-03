# Frequency response of the Hamming window vs a plain box (rectangular) window.
#
# Companion to examples/hamming_apodisation.jl: that script shows the *effect* of
# apodisation on a reconstructed phantom; this one shows *why* it works. A finite
# Cartesian acquisition with no window = a box window in k-space, whose transform
# (Dirichlet/sinc PSF) has −13 dB first side-lobes that ring across edges. The
# Hamming window trades a ~2× wider main lobe for −43 dB side-lobes.
#
# We FFT each window (heavily zero-padded for a smooth curve) and plot the
# magnitude response in dB, normalised to 0 dB at the peak. Window-only: no
# phantom, no simulation.
#
# Writes one vector PDF to src/assets/ (drop straight into LaTeX). If Kaleido
# misbehaves on your platform, change FORMAT to "svg" or "png".
#
# Run with:  julia --project=. examples/hamming_window_response.jl

using MRISystemPhantom, KomaMRI, FFTW

const PlotlyJS = parentmodule(typeof(plot_phantom_map(Phantom(x = [0.0]), :T1; height = 10)))

const OUT = joinpath(@__DIR__, "..", "..", "src", "assets", "report")
isdir(OUT) || mkpath(OUT)
const FORMAT = "png"

const N    = 32     # realistic phase-encode line count (matches hamming_apodisation.jl Npe)
const NPAD = 8192   # zero-pad length: dense frequency sampling => smooth side-lobes

# The two windows. hamming_window_2d is separable, so a 1-D slice is the column of
# hamming_window_2d(N, 1) — but spell it out here so the figure is self-contained.
box = ones(Float64, N)
ham = 0.54 .- 0.46 .* cos.(2π .* (0:N-1) ./ (N - 1))

# Magnitude response in dB, normalised to peak = 0 dB, centred at DC.
function response_db(w)
    padded = vcat(w, zeros(eltype(w), NPAD - length(w)))  # zero-pad to NPAD
    W = fftshift(abs.(fft(padded)))                       # DC to centre
    20 .* log10.(W ./ maximum(W) .+ eps())
end

# Frequency axis in cycles/sample over [-0.5, 0.5), then re-expressed as image
# pixel offset: one cycle/sample == N pixels, so xpix = freq * N. In pixels the
# point-spread structure is literal — box nulls land on integer pixels, each
# side-lobe spans ~1 pixel, and Hamming's wider main lobe is the resolution cost.
freq = range(-0.5, 0.5; length = NPAD + 1)[1:end-1]
xpix = freq .* N
XLIM_PX = 6.5                # zoom to the informative core, in pixels

fig = PlotlyJS.plot(
    [
        PlotlyJS.scatter(; x = xpix, y = response_db(box), mode = "lines",
                         name = "Box (no window): −13 dB",
                         line = Dict(:width => 2)),
        PlotlyJS.scatter(; x = xpix, y = response_db(ham), mode = "lines",
                         name = "Hamming: −43 dB",
                         line = Dict(:width => 2)),
    ],
    PlotlyJS.Layout(
        width = 640, height = 440,
        margin = Dict(:t => 60, :l => 70, :r => 30, :b => 60),
        title = Dict(:text => "Window frequency response (N = $N)",
                     :x => 0.5, :xanchor => "center", :font => Dict(:size => 18)),
        xaxis = Dict(:title => Dict(:text => "image pixel offset"),
                     :range => [-XLIM_PX, XLIM_PX], :zeroline => false,
                     :tick0 => 0, :dtick => 1),
        yaxis = Dict(:title => Dict(:text => "magnitude [dB]"),
                     :range => [-80, 3]),
        legend = Dict(:x => 0.98, :y => 0.98, :xanchor => "right", :yanchor => "top"),
        # Reference lines at the two side-lobe levels the docstring quotes.
        shapes = [
            Dict(:type => "line", :x0 => -XLIM_PX, :x1 => XLIM_PX, :y0 => -13, :y1 => -13,
                 :line => Dict(:dash => "dot", :width => 1, :color => "gray")),
            Dict(:type => "line", :x0 => -XLIM_PX, :x1 => XLIM_PX, :y0 => -43, :y1 => -43,
                 :line => Dict(:dash => "dot", :width => 1, :color => "gray")),
        ],
    ),
)

out_path = joinpath(OUT, "hamming_window_response.$FORMAT")
PlotlyJS.savefig(fig, out_path)
@info "Saved" path = out_path

# ---------------------------------------------------------------------------
# Second figure: the window WEIGHTS themselves — flat box vs the Hamming taper
# that decreases towards the k-space edges. This is what multiplies k-space
# before the IFFT (the response above is the transform of these curves).
samp = 0:N-1
fig_w = PlotlyJS.plot(
    [
        PlotlyJS.scatter(; x = samp, y = box, mode = "lines+markers",
                         name = "Box (no window)",
                         line = Dict(:width => 2), marker = Dict(:size => 5)),
        PlotlyJS.scatter(; x = samp, y = ham, mode = "lines+markers",
                         name = "Hamming",
                         line = Dict(:width => 2), marker = Dict(:size => 5)),
    ],
    PlotlyJS.Layout(
        width = 640, height = 440,
        margin = Dict(:t => 60, :l => 70, :r => 30, :b => 60),
        title = Dict(:text => "Window weights (N = $N)",
                     :x => 0.5, :xanchor => "center", :font => Dict(:size => 18)),
        xaxis = Dict(:title => Dict(:text => "k-space sample index"),
                     :zeroline => false),
        yaxis = Dict(:title => Dict(:text => "weight"),
                     :range => [0, 1.08], :zeroline => false),
        legend = Dict(:x => 0.5, :y => 0.02, :xanchor => "center", :yanchor => "bottom"),
    ),
)

out_path_w = joinpath(OUT, "hamming_window_weights.$FORMAT")
PlotlyJS.savefig(fig_w, out_path_w)
@info "Saved" path = out_path_w

# ---------------------------------------------------------------------------
# Combined side-by-side panel: (a) weights, (b) frequency response. This is the
# report-ready pair — the taper you apply and the side-lobe suppression it buys.
const C_BOX, C_HAM = "#1f77b4", "#d62728"   # shared colours across both panels
fig_p = PlotlyJS.make_subplots(rows = 1, cols = 2,
    subplot_titles = ["(a) Window weights" "(b) Frequency response"],
    horizontal_spacing = 0.12)

# Panel (a): weights. Legend shown here only (legendgroups tie to panel b).
PlotlyJS.add_trace!(fig_p, PlotlyJS.scatter(; x = samp, y = box, mode = "lines+markers",
    name = "Box (no window)", legendgroup = "box", line = Dict(:width => 2, :color => C_BOX),
    marker = Dict(:size => 4)), row = 1, col = 1)
PlotlyJS.add_trace!(fig_p, PlotlyJS.scatter(; x = samp, y = ham, mode = "lines+markers",
    name = "Hamming", legendgroup = "ham", line = Dict(:width => 2, :color => C_HAM),
    marker = Dict(:size => 4)), row = 1, col = 1)

# Panel (b): frequency response. Same colours/groups, legend suppressed here.
PlotlyJS.add_trace!(fig_p, PlotlyJS.scatter(; x = xpix, y = response_db(box), mode = "lines",
    legendgroup = "box", showlegend = false, line = Dict(:width => 2, :color => C_BOX)),
    row = 1, col = 2)
PlotlyJS.add_trace!(fig_p, PlotlyJS.scatter(; x = xpix, y = response_db(ham), mode = "lines",
    legendgroup = "ham", showlegend = false, line = Dict(:width => 2, :color => C_HAM)),
    row = 1, col = 2)
# −13 / −43 dB reference lines (as traces so they live on panel b's axes).
for lvl in (-13, -43)
    PlotlyJS.add_trace!(fig_p, PlotlyJS.scatter(; x = [-XLIM_PX, XLIM_PX], y = [lvl, lvl],
        mode = "lines", showlegend = false,
        line = Dict(:width => 1, :dash => "dot", :color => "gray")), row = 1, col = 2)
end

PlotlyJS.relayout!(fig_p; width = 980, height = 440,
    margin = Dict(:t => 60, :l => 70, :r => 30, :b => 95),
    title = Dict(:text => "Hamming vs box window (N = $N)",
                 :x => 0.5, :xanchor => "center", :font => Dict(:size => 18)),
    legend = Dict(:orientation => "h", :x => 0.5, :y => -0.22,
                  :xanchor => "center", :yanchor => "top"),
    xaxis  = Dict(:title => Dict(:text => "k-space sample index"), :zeroline => false),
    yaxis  = Dict(:title => Dict(:text => "weight"), :range => [0, 1.08], :zeroline => false),
    xaxis2 = Dict(:title => Dict(:text => "image pixel offset"),
                  :range => [-XLIM_PX, XLIM_PX], :zeroline => false,
                  :tick0 => 0, :dtick => 1),
    yaxis2 = Dict(:title => Dict(:text => "magnitude [dB]"), :range => [-80, 3]))

out_path_p = joinpath(OUT, "hamming_window_panel.$FORMAT")
PlotlyJS.savefig(fig_p, out_path_p)
@info "Saved" path = out_path_p
