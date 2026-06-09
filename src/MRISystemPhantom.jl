"""
    MRISystemPhantom

Programmatic, parameterised digital twin of the **NIST/ISMRM System Standard
Model 130** MRI phantom (Caliber MRI). Builds a `KomaMRI.Phantom` from a
single `PhantomConfig` so simulations can sample field strength, rotation,
voxel size, per-property jitter, etc. without touching library code.

    using MRISystemPhantom
    obj = build_phantom(PhantomConfig(field = :T3, voxel_size_mm = 2.0))
"""
module MRISystemPhantom

using KomaMRI
using FFTW
using Random
using LinearAlgebra
using Rotations
import Suppressor
import Statistics
import Statistics: mean

# --- materials (pure data) ------------------------------------------------
include("materials/fiducial.jl")    # defines Relax, FIDUCIAL_PROPS
include("materials/background.jl")  # uses Relax; defines BACKGROUND_WATER
include("materials/t1_array.jl")
include("materials/t2_array.jl")
include("materials/pd_array.jl")    # uses BACKGROUND_WATER

# --- geometry primitives --------------------------------------------------
include("geometry/plane.jl")
include("geometry/sphere.jl")
include("geometry/plate_layouts.jl")
include("geometry/projection.jl")

# --- configs, builder, augmentations --------------------------------------
include("sphere_descriptor.jl")
include("config.jl")
include("augment.jl")
include("builder.jl")
include("random_phantom.jl")

# --- sequences, fitting, baseline experiments -----------------------------
include("sequences/blocks.jl")
include("fitting/fits.jl")
include("baselines/conventional.jl")
include("baselines/cr_optimal.jl")
include("baselines/cr_optimal_alpha.jl")

# --- imaging pipeline (k-space ↔ image, noise) ----------------------------
include("imaging.jl")

# --- analytical forward models (phantom → predicted image) ----------------
include("forward_model.jl")

# --- cached-water model (analytic background-water k-space) ----------------
include("water_cache.jl")

# --- diagnostics ----------------------------------------------------------
include("diagnostics/snr.jl")

# --- interactive 3D visualisation (implemented in the PlotlyJS extension) --
"""
    plot_phantom_html(cfg::PhantomConfig = PhantomConfig(); kwargs...)

Build an interactive 3D render of a phantom, with one Plotly `scatter3d` trace
per group (T1/T2/PD plates, fiducials, custom spheres, background water). Spheres
are coloured by a selectable property (`T1`, `T2`, `T2s`, `ρ`, `Δw`). Water is a
translucent constant-colour trace by default; pass `color_water = true` to put it
on the same property colour axis as the spheres. Every group can be toggled from
the legend.

This is implemented in a package extension that is only loaded when `PlotlyJS` is
available. Call it after `using PlotlyJS` (or any package that loads it, e.g.
`using KomaMRI`); otherwise this fallback tells you to load PlotlyJS.

Keyword arguments: `color_by`, `properties`, `max_water_points`,
`max_sphere_points`, `color_water`, `water_opacity`, `opacity_sliders`
(`:water`/`:all`/`:none`), `height`, and `file` (when set, the figure is also
written to that path as HTML).
"""
function plot_phantom_html(args...; kwargs...)
    error("plot_phantom_html requires PlotlyJS. Load PlotlyJS, or another package " *
          "that loads PlotlyJS, before calling plot_phantom_html.")
end

"""
    plot_random_phantom_explorer_html(rpcfg::RandomPhantomConfig; seeds = 1:10, kwargs...)

Pre-sample one episode per seed from `rpcfg` and bake them all into a single
self-contained interactive HTML. A Plotly slider (plus Play/Pause) scrubs through
the pre-sampled episodes — dragging it acts as a "resample" button that loads a
fresh random phantom, so rotation, subset selection, and material sampling can be
seen to vary. Because every episode is precomputed and embedded, the result needs
no running Julia and works offline (ideal for a presentation).

Each episode collapses to two traces: the randomised spheres (coloured by
`color_by`, on a shared colour axis so episodes are comparable) and a translucent
background-water trace. The per-episode title reports the seed, the active sphere
count, and a rotation summary.

This is implemented in the PlotlyJS extension; call it after `using PlotlyJS` (or
a package that loads it). Keyword arguments: `seeds`, `color_by`,
`max_water_points`, `max_sphere_points`, `water_opacity`, `sphere_size`,
`height`, `play_ms`, and `file` (when set, the figure is also written there).
"""
function plot_random_phantom_explorer_html(args...; kwargs...)
    error("plot_random_phantom_explorer_html requires PlotlyJS. Load PlotlyJS, or " *
          "another package that loads PlotlyJS, before calling it.")
end

export PhantomConfig, AugmentConfig, SphereDescriptor, scanner_for_field,
       build_phantom, build_plate, build_sphere, build_background_water,
       build_phantom_from_descriptors,
       sphere_descriptors, all_sphere_descriptors,
       with_sphere_relaxation,
       # random phantom (domain randomisation)
       RandomPhantomConfig, RandomPhantomEpisode,
       sample_phantom_config, sample_phantom, eval_episodes,
       FixedPose, InPlanePoseSampler, GaussianEulerPose, UniformSO3PoseSampler,
       E2SphereSelector, SphereCountPerPlate, RatioPreservingLogNormalT1,
       MaterialDistributionSampler,
       PerLabel, PerPlate, PerSphere, PreserveNominalRatio, ScaledFrom,
       transform_descriptor, transform_descriptors,
       sphere_descriptor_pixel, sphere_descriptor_pixels,
       voxelise_sphere, sphere_volume,
       Slab, slice_basis, signed_distance, voxelise_plane,
       contrast_plate_centres, fiducial_grid_centres,
       rotation_matrix, apply_transform!, apply_per_spin_noise!,
       T1_ARRAY, T2_OF_T1_ARRAY, T1_ARRAY_LEGACY,
       T2_ARRAY, T1_OF_T2_ARRAY,
       PD_FRACTIONS, pd_t1, pd_t2,
       FIDUCIAL_PROPS, BACKGROUND_WATER, Relax,
       PLATE_Z_MM, CONTRAST_RADIUS_M, FIDUCIAL_RADIUS_M, HOUSING_RADIUS_M,
       # sequences
       rf_duration, ir_sequence, se_sequence, mse_sequence, ir_se_2d_sequence,
       ir_tse_2d_sequence, se_2d_sequence, gre_2d_sequence,
       SpoilerConfig, apply_spoiler,
       mse_signal,
       single_spin_phantom,
       # fitting
       fit_t1_ir, fit_t2_se,
       # conventional-sequence baseline
       measure_ir_signal, measure_se_signal, measure_mse_signal,
       measure_t1, measure_t2,
       default_TI_schedule, default_TE_schedule,
       adaptive_TI_schedule, adaptive_TE_schedule,
       run_conventional_baseline,
       # CR-optimal baseline
       cr_T1_variance, cr_fleet_objective, cr_optimize, cr_optimize_sweep,
       block_time_s, schedule_time_s,
       # α-aware CR-optimal + Ernst-angle baseline
       ernst_angle, ernst_fixed_schedule,
       cr_T1_variance_alpha, cr_fleet_objective_alpha,
       cr_optimize_alpha, cr_optimize_sweep_alpha,
       # generalized IR
       generalized_ir_signal, fit_t1_generalized_ir, fit_t1_t2_generalized_ir,
       steady_state_mz_at_excite,
       transient_mz_at_excite_npe, transient_mz_per_shot,
       raw_to_kspace, kspace_to_image, hamming_window_2d, roi_mean, phantom_occupancy,
       phys_to_pixel, phys_to_pixel_wrap,
       phantom_parameter_map, image_to_kspace,
       # cached-water model
       CachedWaterModel, build_cached_water_model, cached_water_ksp,
       build_dry_and_water,
       # analytical forward models
       central_crop, bandlimit_image, ir_se_theory_image, ir_se_theory_image_binned,
       add_noise!, add_noise, add_gaussian_noise!,
       # SNR diagnostics
       SNRReport, ImageSNRReport, MultiBlockSNRReport,
       background_mask, nema_stats, dual_acq_stats,
       image_snr_report, snr_report, snr_report_from_clean,
       pooled_image_snr_report, multi_block_snr_report_to_dict,
       print_snr_report, snr_report_to_dict,
       # interactive 3D visualisation (PlotlyJS extension)
       plot_phantom_html, plot_random_phantom_explorer_html

end # module
