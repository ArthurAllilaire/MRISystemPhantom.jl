# Visualise the phantoms behind each fidelity stage of the multi-fidelity RL
# curriculum (icr/python/train_e2_mf.py `FIDELITIES`). Writes one interactive
# Plotly HTML per config into `src/assets/fidelity_phantoms/`.
#
# The Python `FIDELITIES` dict only flips the forward-model / water knobs; the
# trace into the env (icr/python/qalibremd_gym/env_e2.py → julia/rl/e2.jl
# `_e2_build_episode_phantom`) shows those knobs map onto a `PhantomConfig` as:
#
#     include_water        → include_plates contains :water (else dropped)
#     water_voxel_size_mm  → PhantomConfig.water_voxel_size_mm
#     forward_model        → :analytic builds NO Koma phantom (closed-form sim)
#     water_model          → :cached_perline splits spheres/water for the sim
#                            but builds the SAME geometry as :bloch
#
# So geometrically only three things matter here: is there water, and at what
# voxel size. `cached` ≡ `full` and `cached3` ≡ `full3` as phantoms — the
# difference between them is purely how the water signal is computed, not where
# the spins sit. The interesting axis to *see* is water voxelisation thickness:
# default (= sphere voxel size) vs the coarse 3 mm used by `cached3` / `full3`.
#
# Run with:  julia --project=. examples/plot_fidelity_phantoms.jl

using KomaMRI
using MRISystemPhantom

const PlotlyJS = parentmodule(typeof(plot_phantom_map(
    Phantom(x = [0.0]), :T1; height = 10)))

const ASSETS = joinpath(@__DIR__, "..", "src", "assets", "fidelity_phantoms")
isdir(ASSETS) || mkpath(ASSETS)

save_html(p, name) = PlotlyJS.savefig(p, joinpath(ASSETS, name); format = "html")

# Held fixed across the curriculum (env defaults: T1.5 plate, 1 mm spheres).
const FIELD          = :T15
const VOXEL_MM       = 1.0
const INCLUDE_PLATES = [:T1]          # T1 plate spheres are the scene of interest

# Mirror of train_e2_mf.py FIDELITIES — only the water-relevant knobs are kept,
# since forward_model/water_model do not change phantom geometry.
const FIDELITIES = [
    # name        include_water   water_voxel_size_mm   note
    ("analytic",  false,          nothing,  "no Koma phantom (closed-form sim)"),
    ("dry",       false,          nothing,  "spheres only, no background water"),
    ("cached",    true,           nothing,  "water at sphere voxel size"),
    ("full",      true,           nothing,  "water at sphere voxel size (≡ cached geometry)"),
    ("cached3",   true,           3.0,      "coarse 3 mm water"),
    ("full3",     true,           3.0,      "coarse 3 mm water (≡ cached3 geometry)"),
]

for (name, include_water, water_vox, note) in FIDELITIES
    if name == "analytic"
        @info "Skipping $name — $note"
        continue
    end

    plates = include_water ? vcat(INCLUDE_PLATES, :water) : INCLUDE_PLATES
    # Match the env: keep only the thin axial slab through the T1 plate that is
    # actually acquired (e2.jl uses slice_thickness_mm = voxel_size_mm).
    cfg = PhantomConfig(
        field               = FIELD,
        voxel_size_mm       = VOXEL_MM,
        water_voxel_size_mm = water_vox,
        include_plates      = plates,
        slice_thickness_mm  = VOXEL_MM,
        slice_center_mm     = (0.0, 0.0, PLATE_Z_MM.T1),
    )
    obj = build_phantom(cfg)
    @info "Built $name phantom" note spins = length(obj.x) water_vox

    # Colour by ρ so the coarse vs fine water grid is obvious, and by T1 to keep
    # the sphere contrast readable.
    for prop in (:ρ, :T1)
        p = plot_phantom_map(obj, prop;
                             view_2d = true, height = 650, max_spins = 200_000)
        save_html(p, "fidelity_$(name)_$(prop).html")
    end
end

@info "Saved fidelity phantom maps" dir = ASSETS
