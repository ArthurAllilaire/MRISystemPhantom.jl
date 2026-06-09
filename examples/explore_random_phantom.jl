# Interactive explorer for the random phantom sampler. Pre-samples a handful of
# episodes from one `RandomPhantomConfig` and bakes them into a single
# self-contained HTML: drag the slider (or hit ▶ Resample) to load a fresh random
# phantom and watch the rotation, the active sphere subset, and the sampled
# materials all change. It is fully static — no running Julia, works offline — so
# it is safe to open from a file during a presentation.
#
# Run with:  julia --project=. examples/explore_random_phantom.jl

using MRISystemPhantom
using KomaMRI   # loads PlotlyJS transitively, which activates the viewer extension

const ASSETS = joinpath(@__DIR__, "..", "src", "assets")
isdir(ASSETS) || mkpath(ASSETS)

# One distribution that exercises everything worth showing in the talk:
#   - pose:     in-plane rotation + x/y translation (varies orientation)
#   - subset:   4–10 of the 14 T1-plate spheres (varies which spheres appear)
#   - material: log-normal T1 jitter with ratio-preserving T2 (varies contrast)
# Coarse voxels keep each pre-sampled episode small and the HTML snappy.
rpcfg = RandomPhantomConfig(
    base = PhantomConfig(
        field               = :T15,
        voxel_size_mm       = 2.0,
        water_voxel_size_mm = 6.0,
        include_plates      = [:T1, :water],
    ),
    sphere_selector  = SphereCountPerPlate(:T1 => 4:10),
    material_sampler = RatioPreservingLogNormalT1(0.25),
    pose_sampler     = InPlanePoseSampler(rotation_sigma_rad = 0.25,
                                          translation_sigma_mm = 6.0),
)

out = joinpath(ASSETS, "random_phantom_explorer.html")
plot_random_phantom_explorer_html(rpcfg; seeds = 1:10, color_by = :T1, file = out)
@info "Saved random-phantom explorer" episodes = 10 file = out

# To show *uniform 3D orientation* instead of in-plane, swap the pose sampler:
#   pose_sampler = UniformSO3PoseSampler()
# (Volumetric view — tilts spheres out of a thin axial slab, so use the full
#  water volume rather than a single slice.)
