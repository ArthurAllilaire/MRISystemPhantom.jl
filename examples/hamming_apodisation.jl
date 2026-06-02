# Reconstruction "cleaning" with a Hamming window.
#
# A finite Cartesian acquisition reconstructs the object convolved with a sinc
# (Dirichlet) point-spread function, whose side-lobes ring across pixels near any
# sharp edge — most visibly the water/sphere boundaries. Apodising k-space with a
# 2-D Hamming window before the IFFT lowers those side-lobes (−13 dB → −43 dB),
# suppressing the ringing at the cost of a wider main lobe (lower resolution).
#
# This example reconstructs ONE coarse-water phantom two ways — plain and
# Hamming-apodised — and shows `coarse | coarse+Hamming | difference` for both
# the image and the k-space it is reconstructed from. The Hamming column's
# k-space is the *windowed* k-space (what the IFFT actually sees), so the
# apodisation is visible directly. The window is a reconstruction-side operation
# (an element-wise multiply), independent of and far cheaper than the simulation.
#
# Writes one interactive Plotly HTML to src/assets/.
#
# Run with:  julia --project=. examples/hamming_apodisation.jl

using MRISystemPhantom, KomaMRI

const PlotlyJS = parentmodule(typeof(plot_phantom_map(Phantom(x = [0.0]), :T1; height = 10)))

const OUT = joinpath(@__DIR__, "..", "src", "assets")
isdir(OUT) || mkpath(OUT)

const FOV, Nfe, Npe = 0.2, 64, 32

# Build a single coarse-water phantom and acquire one IR-SE-2D image of the T1 plate.
cfg = PhantomConfig(field = :T15, voxel_size_mm = 1.0,
                    water_voxel_size_mm = 3.0,            # coarse water (blocky edges)
                    include_plates = [:T1, :water],
                    slice_thickness_mm = 1.0,
                    slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
obj = build_phantom(cfg)
seq = ir_se_2d_sequence(0.4, 0.08, 1.5; FOV = FOV, Nfe = Nfe, Npe = Npe)
raw = simulate(obj, seq, scanner_for_field(cfg))
ksp = raw_to_kspace(raw, Npe, Nfe)

# Two reconstructions of the SAME k-space.
img_plain = kspace_to_image(ksp)
img_ham   = kspace_to_image(ksp; hamming = true)
ksp_plain = log10.(abs.(ksp) .+ eps(Float32))
ksp_ham   = log10.(abs.(ksp .* hamming_window_2d(Npe, Nfe)) .+ eps(Float32))

# coarse | coarse+Hamming | difference  (title, image_z, img_cs, ksp_z, ksp_cs, is_diff)
cols = [
    ("Coarse (plain)",   img_plain,            "Greys", ksp_plain,            "Viridis", false),
    ("Coarse + Hamming", img_ham,              "Greys", ksp_ham,              "Viridis", false),
    ("Difference",       img_plain .- img_ham, "RdBu",  ksp_plain .- ksp_ham, "RdBu",    true),
]
n = length(cols)

const HS, VS = 0.12, 0.14
fig = PlotlyJS.make_subplots(rows = 2, cols = n,
    subplot_titles = reshape(vcat([c[1] * "<br>image" for c in cols],
                                  [c[1] * "<br>k-space (log|·|)" for c in cols]), 1, :),
    horizontal_spacing = HS, vertical_spacing = VS)

# Place each colorbar just right of its own panel.
colw = (1 - HS * (n - 1)) / n
rowh = (1 - VS) / 2
col_x1(j) = (j - 1) * (colw + HS) + colw
row_yc(row) = row == 1 ? rowh + VS + rowh / 2 : rowh / 2
colorbar(j, row) = Dict(:x => col_x1(j) + 0.012, :xanchor => "left",
                        :y => row_yc(row), :yanchor => "middle",
                        :len => rowh, :thickness => 10)

for (j, (_t, iz, ics, kz, kcs, isdiff)) in enumerate(cols)
    extra = isdiff ? (; zmid = 0) : (;)   # diverging diff scale, white = 0
    PlotlyJS.add_trace!(fig, PlotlyJS.heatmap(; z = iz, colorscale = ics,
                        showscale = true, colorbar = colorbar(j, 1), extra...), row = 1, col = j)
    PlotlyJS.add_trace!(fig, PlotlyJS.heatmap(; z = kz, colorscale = kcs,
                        showscale = true, colorbar = colorbar(j, 2), extra...), row = 2, col = j)
end

PlotlyJS.relayout!(fig; height = 760, width = 380 * n,
    margin = Dict(:t => 90, :l => 70, :r => 80, :b => 60),
    title = Dict(:text => "Hamming apodisation on the coarse-water reconstruction (IR-SE-2D)",
                 :x => 0.5, :xanchor => "center", :y => 0.98, :yanchor => "top",
                 :font => Dict(:size => 18)))
for ann in fig.plot.layout.annotations
    ann[:font] = Dict(:size => 12)
end
# Square physical FOV (z is Npe×Nfe ⇒ scaleratio = Nfe/Npe) + axis labels.
for k in 1:2n
    row    = k <= n ? 1 : 2
    ya     = k == 1 ? "yaxis" : "yaxis$k"
    xa     = k == 1 ? "xaxis" : "xaxis$k"
    anchor = k == 1 ? "x" : "x$k"
    xtitle = row == 1 ? "freq-encode [px]" : "kx [px]"
    ytitle = row == 1 ? "phase-encode [px]" : "ky [px]"
    PlotlyJS.relayout!(fig;
        Symbol(ya) => Dict(:scaleanchor => anchor, :scaleratio => Nfe / Npe,
                           :title => Dict(:text => ytitle)),
        Symbol(xa) => Dict(:title => Dict(:text => xtitle)))
end

out_path = joinpath(OUT, "hamming_apodisation.html")
PlotlyJS.savefig(fig, out_path; format = "html")
@info "Saved" path = out_path
