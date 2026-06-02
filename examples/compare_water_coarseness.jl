# Compare reconstructed magnitude image and k-space across water-voxel
# coarseness for ONE sequence (IR-SE-2D). The only knob that changes between
# columns is `water_voxel_size_mm`:
#
#     nothing  → water at the sphere voxel size  (≡ "full"  / "cached")
#     3.0      → coarse 3 mm water               (≡ "full3" / "cached3")
#
# Same phantom geometry otherwise, same Bloch simulation through the same
# sequence and scanner, so any difference you see in image/k-space is purely
# the water voxelisation. Edit `WATER_VOXELS` to add/remove columns, or swap
# the `seq = ...` line to compare a different sequence.
#
# Writes one interactive Plotly HTML to src/assets/.

using MRISystemPhantom, KomaMRI, Printf, Statistics

const PlotlyJS = parentmodule(typeof(plot_phantom_map(
    Phantom(x = [0.0]), :T1; height = 10)))

const OUT = joinpath(@__DIR__, "..", "src", "assets")
isdir(OUT) || mkpath(OUT)

# ---- knobs you'll usually touch ------------------------------------------
const FIELD     = :T15
const VOXEL_MM  = 1.0          # sphere voxel size (held fixed)
const FOV       = 0.2          # [m]
const Nfe       = 64
const Npe       = 32
const NREPS     = 20           # timed simulate() repeats (after a warm-up) for mean±std

# water_voxel_size_mm => display label. `nothing` = water at the sphere voxel
# size; a number = coarser water grid (fewer spins, cheaper to simulate). The
# first entry is the high-fidelity reference; the rest are cheaper approximations.
const WATER_VOXELS = [
    nothing => "Fine water (1 mm)",
    3.0     => "Coarse water (3 mm)",
]

# One sequence, reused for every column. Change this to compare another seq.
seq = ir_se_2d_sequence(0.4, 0.08, 1.5;
    FOV = FOV, Nfe = Nfe, Npe = Npe)
# --------------------------------------------------------------------------

# Simulate one phantom and return (magnitude image, log-magnitude k-space).
function recon(water_vox)
    cfg = PhantomConfig(
        field               = FIELD,
        voxel_size_mm       = VOXEL_MM,
        water_voxel_size_mm = water_vox,
        include_plates      = [:T1, :water],
        slice_thickness_mm  = VOXEL_MM,
        slice_center_mm     = (0.0, 0.0, PLATE_Z_MM.T1),
    )
    obj     = build_phantom(cfg)
    scanner = scanner_for_field(cfg)
    nspins  = length(obj.x)
    @info "Simulating" water_vox spins = nspins

    raw = simulate(obj, seq, scanner)                # warm-up (JIT) + used for recon
    ksp = raw_to_kspace(raw, Npe, Nfe)               # complex k-space (DC centred)
    img = kspace_to_image(ksp)                       # magnitude image
    ksp_log = log10.(abs.(ksp) .+ eps(Float32))      # log-mag k-space (for display)

    # Wall-clock cost: time NREPS fresh simulate() calls and report mean ± std.
    times = [(@elapsed simulate(obj, seq, scanner)) for _ in 1:NREPS]
    img, ksp_log, ksp, nspins, mean(times), std(times)
end

# Reconstruct every config up front so we can diff them. The first entry is the
# reference; each other entry gets a "ref − this" difference column appended.
# Tuple layout per entry: (label, img, ksp_log, ksp_complex, nspins).
results = [(lbl, recon(water_vox)...) for (water_vox, lbl) in WATER_VOXELS]
ref_lbl, ref_img, ref_ksp = results[1]              # ref_ksp is log-mag (for diff plots)

# --- Overall sums ----------------------------------------------------------
# Σ|image| (L1) is NOT a Fourier invariant, so it need not match. The physically
# conserved quantity is the k-space DC point = ∫ signal = total magnetization,
# which should be ~equal across coarseness (same water volume), but not bit-exact
# because coarse voxels tile the boundary differently. Σ|k|² is image L2 energy
# (Parseval), reported for completeness.
dc = (Npe ÷ 2 + 1, Nfe ÷ 2 + 1)
ref_dc = results[1][4][dc...]                      # complex DC of the reference
@printf("\n%-20s  %8s  %16s  %14s  %14s  %14s\n",
        "config", "spins", "sim time [s]", "Σ|image|", "|k-space DC|", "Σ|k|²")
for (lbl, img, _ksp_log, ksp, nspins, tμ, tσ) in results
    @printf("%-20s  %8d  %7.3f ± %-6.3f  %14.6e  %14.6e  %14.6e\n",
            lbl, nspins, tμ, tσ, sum(img), abs(ksp[dc...]), sum(abs2, ksp))
end
for (lbl, _img, _ksp_log, ksp, _n, _tμ, _tσ) in results[2:end]
    rel = abs(ref_dc - ksp[dc...]) / abs(ref_dc)   # complex (mag+phase) DC mismatch
    @printf("DC mismatch  %s vs %s: %.4f %%\n", ref_lbl, lbl, 100rel)
end
println()

# --- Per-sphere ROI signal -------------------------------------------------
# The k-space DC is dominated by background water; what actually matters is how
# much the *sphere* signal (the measurement target) moves when the water grid is
# coarsened. Compare the reconstructed ROI magnitude at each T1-sphere centre,
# reference vs each cheaper config. Water voxelisation doesn't move the spheres,
# so a single cfg gives the pixel locations for all columns.
# r = 0 is a single centre pixel (maximally Gibbs-/rounding-sensitive); r = 1 is
# a 3×3 mean over the sphere, which is the more physically meaningful measure.
const ROI_RADII = [0, 1]
roi_cfg = PhantomConfig(field = FIELD, voxel_size_mm = VOXEL_MM,
                        include_plates = [:T1, :water],
                        slice_thickness_mm = VOXEL_MM,
                        slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
descs   = sphere_descriptors(:T1, roi_cfg)
px      = sphere_descriptor_pixels(descs, Npe, Nfe, FOV)

for (lbl, img, _ksp_log, _ksp, _n, _tμ, _tσ) in results[2:end]
    refs = Dict(r => [roi_mean(ref_img, p...; r = r) for p in px] for r in ROI_RADII)
    curs = Dict(r => [roi_mean(img,     p...; r = r) for p in px] for r in ROI_RADII)

    @printf("\nPer-sphere ROI signal — %s vs %s\n", ref_lbl, lbl)
    @printf("%-12s", "sphere")
    for r in ROI_RADII
        @printf("   ref(r%d)  this(r%d)  Δ%%(r%d)", r, r, r)
    end
    println()
    for (i, d) in enumerate(descs)
        @printf("%-12s", string(d.label))
        for r in ROI_RADII
            rr, cc = refs[r][i], curs[r][i]
            @printf("  %8.4f  %8.4f  %+6.2f", rr, cc, 100 * (cc - rr) / rr)
        end
        println()
    end
    # NRMSE over spheres is a null-robust headline (per-sphere Δ% blows up for
    # spheres sitting near their IR null, where ref ≈ 0).
    for r in ROI_RADII
        nrmse = sqrt(mean(abs2, curs[r] .- refs[r])) / sqrt(mean(abs2, refs[r]))
        @printf("ROI NRMSE (r=%d): %.2f %%\n", r, 100 * nrmse)
    end
end
println()

# Build the column list: raw columns first, then one diff column per non-ref.
# Each column is a NamedTuple with separate image / k-space titles + colorscales.
columns = NamedTuple[]
for (lbl, img, ksp_log, _cplx, nspins, tμ, _tσ) in results
    push!(columns, (img_title = "$(lbl)<br>$(nspins) spins · $(round(tμ, digits = 1)) s",
                    img_z = img, img_cs = "Greys",
                    ksp_title = "k-space (log|·|)", ksp_z = ksp_log, ksp_cs = "Viridis",
                    is_diff = false))
end
for (lbl, img, ksp_log, _cplx, _n, _tμ, _tσ) in results[2:end]
    push!(columns, (img_title = "Difference (fine − coarse)",
                    img_z = ref_img .- img, img_cs = "RdBu",
                    ksp_title = "k-space difference",
                    ksp_z = ref_ksp .- ksp_log, ksp_cs = "RdBu",
                    is_diff = true))
end

# 2 rows (image / k-space) × ncols (raw configs + diffs).
n = length(columns)
const HS = 0.12   # horizontal spacing (fraction) — leaves room for colorbars
const VS = 0.14   # vertical spacing
fig = PlotlyJS.make_subplots(
    rows = 2, cols = n,
    subplot_titles = reshape(
        vcat([c.img_title for c in columns],
             [c.ksp_title for c in columns]),
        1, :),
    horizontal_spacing = HS, vertical_spacing = VS)

# Subplot domain geometry (matches make_subplots' equal-column layout) so each
# colorbar sits just right of its own panel rather than stacking at the figure edge.
colw = (1 - HS * (n - 1)) / n
rowh = (1 - VS) / 2
col_x1(j) = (j - 1) * (colw + HS) + colw          # right edge of column j
row_yc(row) = row == 1 ? rowh + VS + rowh / 2 : rowh / 2

function colorbar(j, row)
    Dict(:x => col_x1(j) + 0.012, :xanchor => "left",
         :y => row_yc(row), :yanchor => "middle",
         :len => rowh, :thickness => 10)
end

for (j, c) in enumerate(columns)
    # Diff heatmaps use a diverging scale centred at zero so 0 = white.
    extra = c.is_diff ? (; zmid = 0) : (;)
    PlotlyJS.add_trace!(fig,
        PlotlyJS.heatmap(; z = c.img_z, colorscale = c.img_cs,
                         showscale = true, colorbar = colorbar(j, 1), extra...),
        row = 1, col = j)
    PlotlyJS.add_trace!(fig,
        PlotlyJS.heatmap(; z = c.ksp_z, colorscale = c.ksp_cs,
                         showscale = true, colorbar = colorbar(j, 2), extra...),
        row = 2, col = j)
end

PlotlyJS.relayout!(fig; height = 780, width = 380 * n,
    margin = Dict(:t => 90, :l => 70, :r => 80, :b => 60),
    title = Dict(:text => "Simulation fidelity vs cost: background-water voxelisation (IR-SE-2D)",
                 :x => 0.5, :xanchor => "center", :y => 0.98, :yanchor => "top",
                 :font => Dict(:size => 18)))
# Shrink subplot-title font so the two-line headers clear the figure title.
for ann in fig.plot.layout.annotations
    ann[:font] = Dict(:size => 12)
end
# Square physical FOV + axis labels (row 1 = image, row 2 = k-space). z is
# Npe×Nfe, so y spans Npe units and x spans Nfe units; scaleratio = Nfe/Npe
# makes the box square even when the matrix is anisotropic (e.g. 32×64).
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

out_path = joinpath(OUT, "water_coarseness_compare.html")
PlotlyJS.savefig(fig, out_path; format = "html")
@info "Saved" path = out_path
