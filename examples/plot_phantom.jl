# Render and save phantom-map snapshots of the QalibreMD Model 130 digital
# twin. Writes interactive HTML into `src/assets/`.
#
# Two render modes:
#   • view_2d = true  → scatter in the x-y plane, best for checking the
#     ring layout of each plate against the manual photos.
#   • view_2d = false → full 3D scatter (slider over z), good for a global
#     overview but sparser unless max_spins is large.
#
# `max_spins` caps how many spins are sent to Plotly. The default 20 000 is
# why earlier renders looked sparse; bumping it to 200 000+ makes the
# geometry readable.

using KomaMRI
using MRISystemPhantom

const PlotlyJS = parentmodule(typeof(plot_phantom_map(
    Phantom(x = [0.0]), :T1; height = 10)))

const ASSETS = joinpath(@__DIR__, "..", "src", "assets")
isdir(ASSETS) || mkpath(ASSETS)

save_html(p, name) = PlotlyJS.savefig(p, joinpath(ASSETS, name); format = "html")

# ---------- slicing helper ----------------------------------------------
"""
    slab(obj, z_mm; halfthick_mm)

Return a Phantom containing just the spins whose z lies within
`±halfthick_mm` of `z_mm`. Useful for rendering one plate at a time.
"""
function slab(obj::Phantom, z_mm::Real; halfthick_mm::Real = 8.0)
    z0 = z_mm * 1e-3
    dz = halfthick_mm * 1e-3
    mask = findall(z -> abs(z - z0) <= dz, obj.z)
    obj[mask]
end

# ---------- 1. full phantom, fine voxels, high spin cap ------------------
cfg_fine = PhantomConfig(field = :T3, voxel_size_mm = 1.0)
obj_fine = build_phantom(cfg_fine)
@info "Fine full phantom" spins = length(obj_fine.x)

for prop in (:T1, :T2, :ρ)
    p3d = plot_phantom_map(obj_fine, prop;
                           height = 650, max_spins = 250_000)
    save_html(p3d, "phantom_$(prop)_3T_3d.html")
end

# ---------- 2. per-plate 2D slices (ring layout) -------------------------
# Plates are held on plate_layouts.jl (PLATE_Z_MM).
for (plate, z_mm) in pairs(PLATE_Z_MM)
    sl  = slab(obj_fine, z_mm; halfthick_mm = 8.0)   # catches the sphere caps
    key = plate === :PD ? :ρ : plate                  # colour by the right prop
    p2d = plot_phantom_map(sl, key;
                           view_2d  = true,
                           height   = 650,
                           max_spins = 150_000)
    save_html(p2d, "plate_$(plate)_slice_3T.html")
end

# ---------- 3. fiducial grid, 2D top view (z ≈ 0) ------------------------
cfg_fid = PhantomConfig(field = :T3, voxel_size_mm = 1.0,
                        include_plates = [:fiducials])
obj_fid = build_phantom(cfg_fid)
p_fid = plot_phantom_map(slab(obj_fid, 0.0; halfthick_mm = 5.5), :T1;
                         view_2d = true, height = 650, max_spins = 150_000)
save_html(p_fid, "fiducials_z0_slice.html")

# ---------- 4. augmented full phantom ------------------------------------
cfg_aug = PhantomConfig(
    field          = :T3,
    voxel_size_mm  = 1.5,
    include_plates = [:T1, :T2, :PD, :fiducials],
    rotation       = (0.0, 0.0, deg2rad(30)),
    translation_mm = (5.0, -3.0, 0.0),
    augment        = AugmentConfig(T1_sigma_rel = 0.03, B0_sigma_Hz = 3.0),
    rng_seed       = 42,
)
obj_aug = build_phantom(cfg_aug)
p_aug = plot_phantom_map(obj_aug, :T1;
                         height = 650, max_spins = 200_000)
save_html(p_aug, "phantom_T1_augmented_3T.html")

@info "Saved phantom maps" dir = ASSETS files = readdir(ASSETS)
