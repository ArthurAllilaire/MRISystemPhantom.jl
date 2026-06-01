# # T1 Mapping with Inversion Recovery
#
# This example builds the T1-plate slice of the QalibreMD Model 130 phantom,
# simulates a multi-TI inversion-recovery spin-echo acquisition at 3 T, and
# fits T1 per sphere — reproducing the core E0 pipeline.

using MRISystemPhantom, KomaMRI

# ## Build the phantom
#
# We isolate the T1 plate with a 10 mm axial slab, using a 2 mm voxel grid for
# the contrast spheres and a 4 mm grid for background water (fewer spins, same
# bulk signal power).

cfg = PhantomConfig(
    field                        = :T3,
    voxel_size_mm                = 2.0,
    water_voxel_size_mm          = 4.0,
    include_plates               = [:T1, :water],
    slice_thickness_mm           = 10.0,
    slice_center_mm              = (0.0, 0.0, PLATE_Z_MM.T1),
)
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)

println("Phantom spins: ", length(obj.x))

# ## Get sphere pixel positions
#
# `sphere_descriptor_pixels` maps each sphere's physical centre to its
# (phase-encode, frequency-encode) pixel index in a 64×64, 200 mm-FOV image.

descs  = sphere_descriptors(:T1, cfg)
px     = sphere_descriptor_pixels(descs, 64, 64, 0.2)

# ## Simulate a TI sweep
#
# Seven inversion times spanning the dynamic range of the T1 plate (85 ms –
# 1800 ms at 3 T). Each call to `ir_se_2d_sequence` produces a 64-shot
# multi-phase-encode sequence; `simulate` returns a `RawAcquisitionData`.

TIs = [0.05, 0.1, 0.25, 0.5, 0.8, 1.2, 2.0]

images = Matrix{Float64}[]
for TI in TIs
    seq = ir_se_2d_sequence(TI, 0.012, 3.0;
        FOV = 0.2, Nfe = 64, Npe = 64,
        spoiler = SpoilerConfig(enabled = true))
    raw = Suppressor.@suppress simulate(obj, seq, scanner)
    ksp = raw_to_kspace(raw, 64, 64)
    push!(images, abs.(kspace_to_image(ksp)))
end

# ## Fit T1 per sphere
#
# Extract the ROI-mean signal for each sphere at each TI, then call
# `fit_t1_generalized_ir`. We pass `Npe = 64` so the fitter uses the transient
# magnetisation ramp model that matches what `ir_se_2d_sequence` actually does.

T1_fits = Float64[]
T1_true = [d.T1 for d in descs]

for (i, p) in enumerate(px)
    mags = Float64[roi_mean(img, p[1], p[2]; radius = 1) for img in images]
    res  = fit_t1_generalized_ir(TIs, fill(π, length(TIs)), mags;
               Npe          = 64,
               sigma_method = :profile_likelihood)
    push!(T1_fits, res.T1)
end

# ## Results
#
# Mean absolute percentage error across the 14 T1-plate spheres:

mape = 100 * mean(abs.(T1_fits .- T1_true) ./ T1_true)
println("T1 MAPE: ", round(mape, digits = 2), " %")

for (i, (fit, true_)) in enumerate(zip(T1_fits, T1_true))
    @printf "  sphere %2d:  true = %5.0f ms   fit = %5.0f ms   err = %+.1f %%\n" i (1000true_) (1000fit) (100*(fit-true_)/true_)
end
