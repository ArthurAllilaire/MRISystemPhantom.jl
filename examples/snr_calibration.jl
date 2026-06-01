# # SNR Calibration with NEMA MS-1 Dual-Acquisition
#
# This example shows how to calibrate the noise level `σ` on k-space so that
# the simulated phantom matches a target clinical SNR, following the NEMA MS-1
# dual-acquisition method.
#
# The dual-acquisition method acquires the same sequence twice with independent
# noise realisations, then computes:
#
# ```
# noise = std(A − B, pooled over signal ROIs) / √2
# SNR   = mean((A+B)/2 in ROI) / noise
# ```
#
# This eliminates Rayleigh bias and structured background artefacts, making it
# the gold standard for reproducible SNR measurement (NEMA MS-1 2014).

using MRISystemPhantom, KomaMRI, Random

# ## Build the phantom and sequence

cfg = PhantomConfig(
    field              = :T3,
    voxel_size_mm      = 2.0,
    water_voxel_size_mm = 4.0,
    include_plates     = [:T1, :water],
    slice_thickness_mm = 10.0,
    slice_center_mm    = (0.0, 0.0, PLATE_Z_MM.T1),
)
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)

seq = ir_se_2d_sequence(0.4, 0.012, 3.0;
    FOV = 0.2, Nfe = 64, Npe = 64,
    spoiler = SpoilerConfig(enabled = true))

# ## Simulate a noise-free acquisition

raw_clean = Suppressor.@suppress simulate(obj, seq, scanner)
ksp_clean = raw_to_kspace(raw_clean, 64, 64)

# ## Build background mask and sphere pixel map

descs  = sphere_descriptors(:T1, cfg)
px     = sphere_descriptor_pixels(descs, 64, 64, 0.2)
bg     = background_mask(obj, 64, 64, 0.2; erosion_px = 2)

# ## Sweep σ and report NEMA dual-acquisition SNR
#
# We add two independent noise realisations to the clean k-space and compute
# the full `SNRReport` via `snr_report_from_clean`.

σ_values = [0.002, 0.005, 0.01, 0.02, 0.05]
rng = MersenneTwister(42)

println("σ         SNR_dual_peak   SNR_ksp")
for σ in σ_values
    rep = snr_report_from_clean(ksp_clean, σ;
        sphere_px = px, bg_mask = bg, rng = rng)
    @printf "  σ=%.3f   %6.1f          %6.1f\n" σ rep.image.snr_dual_peak rep.snr_ksp
end

# ## Detailed report at a chosen σ
#
# `print_snr_report` gives the full breakdown: per-sphere SNR, temporal
# instability, and all three metrics side-by-side.

σ_target = 0.005
rep = snr_report_from_clean(ksp_clean, σ_target;
    sphere_px = px, bg_mask = bg, rng = MersenneTwister(0))
print_snr_report(rep; label = "T1-plate IR (TI=400 ms, σ=$(σ_target))")
