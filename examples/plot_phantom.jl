# Render interactive phantom views of the QalibreMD Model 130 digital twin.
# Writes Plotly HTML into `src/assets/`.
#
# `plot_phantom_html` creates one trace per logical group and includes a dropdown
# for T1/T2/T2s/rho/delta-w colouring. Water is normally a constant translucent
# trace; the material-map slices below pass `color_water = true` to put water on
# the same colour axis as the spheres.
#
# Run with:  julia --project=. examples/plot_phantom.jl

using MRISystemPhantom
using KomaMRI   # loads PlotlyJS transitively, which activates the viewer extension

const ASSETS = joinpath(@__DIR__, "..", "src", "assets")
isdir(ASSETS) || mkpath(ASSETS)

function save_render(cfg::PhantomConfig, name::AbstractString;
        color_by::Symbol = :T1, color_water::Bool = false, kwargs...)
    out = joinpath(ASSETS, name)
    plot_phantom_html(cfg; color_by, color_water, file = out, kwargs...)
    @info "Saved phantom render" file = out
end

# ---------- 1. full phantom ---------------------------------------------
cfg_full = PhantomConfig(field = :T3, voxel_size_mm = 1.0)
save_render(cfg_full, "phantom_3T_3d.html";
            color_by = :T1, height = 650,
            max_water_points = 250_000, max_sphere_points = 250_000)

# ---------- 2. per-plate axial slabs ------------------------------------
# These replace the old 2-D `plot_phantom_map` slice snapshots. The viewer is 3-D,
# but the phantom builder applies the same slab mask through PhantomConfig.
for (plate, z_mm) in pairs(PLATE_Z_MM)
    color_by = plate === :PD ? :ρ : plate
    cfg_slice = PhantomConfig(
        field              = :T3,
        voxel_size_mm      = 1.0,
        include_plates     = [plate, :water],
        slice_thickness_mm = 16.0,
        slice_center_mm    = (0.0, 0.0, z_mm),
    )
    save_render(cfg_slice, "plate_$(plate)_slice_3T.html";
                color_by, color_water = true, height = 650,
                max_water_points = 150_000, max_sphere_points = 150_000)
end

# ---------- 3. fiducial grid near z = 0 ---------------------------------
cfg_fid = PhantomConfig(
    field              = :T3,
    voxel_size_mm      = 1.0,
    include_plates     = [:fiducials],
    slice_thickness_mm = 11.0,
    slice_center_mm    = (0.0, 0.0, 0.0),
)
save_render(cfg_fid, "fiducials_z0_slice.html";
            color_by = :T1, height = 650, max_sphere_points = 150_000)

# ---------- 4. vertical slice through all three contrast plates ----------
# A thin y-normal slab exposes the x-z stack of T1, T2 and PD plates.
cfg_vert = PhantomConfig(
    field              = :T3,
    voxel_size_mm      = 1.0,
    include_plates     = [:T1, :T2, :PD, :water],
    slice_thickness_mm = 1.0,
    slice_center_mm    = (0.0, 0.0, 0.0),
    slice_normal       = (0.0, 1.0, 0.0),
)
save_render(cfg_vert, "vertical_slice_3T.html";
            color_by = :T1, color_water = true, height = 650,
            max_water_points = 150_000, max_sphere_points = 150_000)

# ---------- 5. augmented full phantom -----------------------------------
cfg_aug = PhantomConfig(
    field          = :T3,
    voxel_size_mm  = 1.5,
    include_plates = [:T1, :T2, :PD, :fiducials],
    rotation       = (0.0, 0.0, deg2rad(30)),
    translation_mm = (5.0, -3.0, 0.0),
    augment        = AugmentConfig(T1_sigma_rel = 0.03, B0_sigma_Hz = 3.0),
    rng_seed       = 42,
)
save_render(cfg_aug, "phantom_T1_augmented_3T.html";
            color_by = :T1, height = 650, max_sphere_points = 200_000)

@info "Saved phantom renders" dir = ASSETS
