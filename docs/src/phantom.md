# Phantom Construction

The central type is [`PhantomConfig`](@ref). Pass one to [`build_phantom`](@ref)
and you get a `KomaMRI.Phantom` ready for simulation. Everything — field strength,
voxel size, which plates to include, pose, noise — lives in the config.

## Interactive 3D Viewer

The full phantom can be inspected interactively. Use the legend to toggle plates,
the dropdown to colour spheres by material properties, and hover over points for
coordinates and tissue parameters.

```@raw html
<div class="phantom-viewer-frame">
  <iframe
    id="phantom-interactive-3d"
    title="Interactive 3D rendering of the MR system phantom"
    width="100%"
    height="760"
    loading="lazy"
    style="border:1px solid #d8dee4; border-radius:6px; background:white;"
    allowfullscreen>
  </iframe>
</div>
<p>
  <a id="phantom-interactive-3d-link" href="assets/phantom_interactive_3d.html" target="_blank" rel="noopener">
    Open the interactive phantom in a full page
  </a>
</p>
<script>
(function () {
  var frame = document.getElementById("phantom-interactive-3d");
  var link = document.getElementById("phantom-interactive-3d-link");
  if (!frame || !link) return;
  var asset = "phantom_interactive_3d.html";
  var base = window.location.pathname.endsWith("/") ? "../assets/" : "assets/";
  var src = base + asset;
  frame.src = src;
  link.href = src;
})();
</script>
```

## PhantomConfig fields

```julia
cfg = PhantomConfig(
    field            = :T3,         # :T3 (3.0 T) or :T15 (1.5 T)
    voxel_size_mm    = 2.0,         # isotropic voxel edge length
    include_plates   = [:T1, :T2, :PD, :fiducials, :water],
    serial_number_class = :new,     # :new (≥0042) or :legacy
    temperature_C    = 20.0,
    rotation         = (0.0, 0.0, 0.0),      # Euler XYZ, radians
    translation_mm   = (0.0, 0.0, 0.0),
    augment          = AugmentConfig(),
    rng_seed         = 0,
)
obj = build_phantom(cfg)
```

## Selecting plates

`include_plates` accepts any subset of `[:T1, :T2, :PD, :fiducials, :water]`.
Omitting `:water` removes the 100 mm background water sphere (faster simulation,
no bulk-water signal contribution).

```julia
# T1 plate only — fastest single-plate simulation
cfg = PhantomConfig(include_plates = [:T1])
```

## Slicing

Isolate a single MRI slice without altering the phantom coordinate frame.
Set `slice_thickness_mm` and `slice_center_mm`; spins outside the slab are
dropped before simulation.

```julia
using MRISystemPhantom: PLATE_Z_MM

# 10 mm slab centred on the T1 plate
cfg = PhantomConfig(
    include_plates     = [:T1, :water],
    slice_thickness_mm = 10.0,
    slice_center_mm    = (0.0, 0.0, PLATE_Z_MM.T1),
    slice_normal       = (0.0, 0.0, 1.0),   # default: axial
)
```

The water layer can use a coarser voxel grid (`water_voxel_size_mm`) to reduce
spin count without affecting the contrast spheres. The spin density is reweighted
so the total water signal (and the k-space DC) is conserved.

!!! note "Coarse water: prefer a Hamming window at reconstruction"
    Coarsening the water (`water_voxel_size_mm > voxel_size_mm`) blockifies the
    water edges, which adds Gibbs ringing near the sphere boundaries. Reconstruct
    with a Hamming window — [`kspace_to_image`](@ref)`(...; hamming = true)` — to
    largely remove it. The benefit is largest for single-pixel / edge ROIs and
    modest for 3×3-averaged ROIs, and it costs some spatial resolution (wider
    main lobe). The trade-off is quantified end-to-end in
    `examples/compare_water_coarseness.jl`.

## Domain randomisation (augmentation)

For RL training, enable augmentation to sample a distribution of phantoms:

```julia
aug = AugmentConfig(
    T1_sigma_rel       = 0.05,   # ±5% T1 jitter
    T2_sigma_rel       = 0.05,
    position_sigma_mm  = 0.5,    # sub-voxel position noise
    B0_sigma_Hz        = 20.0,   # B0 inhomogeneity
    drop_sphere_p      = 0.1,    # 10% chance each sphere is dropped
)
cfg = PhantomConfig(augment = aug, rng_seed = 42)
```

Each `build_phantom(cfg)` call with the same `rng_seed` produces the same
phantom. Change `rng_seed` per episode for different realisations.

## Pose randomisation

```julia
import Random
θ = 0.1 * randn(Random.MersenneTwister(0), 3)   # small random rotation
cfg = PhantomConfig(rotation = Tuple(θ))
```

## Accessing sphere descriptors

[`sphere_descriptors`](@ref) returns the list of [`SphereDescriptor`](@ref)
objects for one plate, useful for computing ROI pixel positions:

```julia
descs = sphere_descriptors(:T1, cfg)
pixels = sphere_descriptor_pixels(descs, 64, 64, 0.2)  # (Npe, Nfe, FOV)
```

## Customising individual spheres

Override specific spheres by label via `custom_sphere_map`:

```julia
cfg = PhantomConfig(
    custom_sphere_map = Dict(:T1_3 => with_sphere_relaxation(
        sphere_descriptors(:T1, PhantomConfig())[3], 0.5, 0.05)),
)
```

Or drop specific spheres:

```julia
cfg = PhantomConfig(drop_sphere_labels = [:T1_1, :T1_2])
```

## Material tables

The relaxation values are sourced from:

| Constant | Description |
|----------|-------------|
| `T1_ARRAY[:T3]` | T1 values for the 14 T1-plate spheres at 3 T |
| `T1_ARRAY[:T15]` | T1 values at 1.5 T |
| `T1_ARRAY_LEGACY` | Legacy serial class (pre-0042) values |
| `T2_ARRAY[:T3]` | T2 values for the 14 T2-plate spheres |
| `T2_OF_T1_ARRAY` | T2 companion values for each T1-plate sphere |
| `PD_FRACTIONS` | Proton density fractions for PD-plate spheres |

## Domain randomisation: random phantoms

For RL training "by the book" the agent should never see the true phantom
configuration: sample a *different* random phantom each episode and keep the true
values hidden for evaluation. [`RandomPhantomConfig`](@ref) describes a
distribution over deterministic `PhantomConfig`s, and
[`sample_phantom_config`](@ref) draws one episode.

!!! note "Why not `AugmentConfig`?"
    `AugmentConfig` applies *independent per-spin* jitter, so each sphere's mean
    relaxation stays at its nominal value — the true values still leak. Random
    phantoms instead draw **one** value per sphere per episode, before
    voxelisation, so the nominal values are never observable.

```julia
using MRISystemPhantom, Distributions

rp = RandomPhantomConfig(
    base = PhantomConfig(include_plates = [:T1, :water]),
    sphere_selector  = E2SphereSelector(subset_size = 5, forced_indices = [1, 14]),
    material_sampler = RatioPreservingLogNormalT1(0.2),     # jitter T1, keep T2/T1
    pose_sampler     = InPlanePoseSampler(rotation_sigma_rad = 0.05,
                                          translation_sigma_mm = 2.0),
)

episode = sample_phantom_config(rp; rng_seed = 7)
phantom = build_phantom(episode.cfg)   # safe to show the agent
# episode.truth holds the hidden ground truth — keep it out of the observation
```

The returned [`RandomPhantomEpisode`](@ref) carries `cfg` (an ordinary
deterministic config) and a hidden `truth` record (`descriptors_sampled`,
`active_labels`, `active_indices_by_plate`, pose, `episode_seed`, `build_seed`).
`truth` is descriptor-level and pre-augment.

Contrast plates (`:T1`, `:T2`, `:PD`) listed in `base.include_plates` are owned by
the random pipeline: after sampling they are either present as sampled
`custom_sphere_descriptors` or absent if the selector drops them. They are never
silently regenerated as deterministic plates. Non-contrast plates such as `:water`
and, by default, `:fiducials` stay deterministic, so the water cutout still works.

### Sampler contracts

Every sampler field is duck-typed — a `Distributions.jl` object, a constant, a
closure, or a callable struct all work.

- **Material sampler** `(rng, d, ctx) -> result`, where `result` is a
  `NamedTuple` of `T1`/`T2`/`T2s`/`ρ`/`delta_w` overrides, a full
  `SphereDescriptor`, or `nothing` (keep nominal). `ctx` carries
  `plate`, `index`, `label`, `field`, etc.
- **Sphere selector** `nothing` (all), an `Integer`/range (pooled count), a
  `Dict`/[`SphereCountPerPlate`](@ref) (per-plate counts), or a callable
  `(rng, descriptors_by_plate, base) -> selected_by_plate`.
- **Pose sampler** `(rng, base) -> (; rotation, translation_mm)`. Built-ins:
  [`FixedPose`](@ref), [`InPlanePoseSampler`](@ref), [`GaussianEulerPose`](@ref).

### Declarative material sampler

For experiments that need different distributions per plate / sphere / label,
[`MaterialDistributionSampler`](@ref) is a declarative alternative to a hand-written
closure. Property specs resolve recursively through [`PerLabel`](@ref),
[`PerPlate`](@ref), and [`PerSphere`](@ref) (indexed by the stable label index),
bottoming out at a constant, a distribution, a `(rng, d, ctx)` function, or a
derived helper ([`PreserveNominalRatio`](@ref), [`ScaledFrom`](@ref)). Properties
are evaluated in per-sphere dependency order, with cycle detection.

```julia
material = MaterialDistributionSampler(
    T1 = PerPlate(:T1 => Uniform(0.3, 2.5),
                  :T2 => ScaledFrom(:T2, Truncated(Normal(10.0, 1.0), 0.1, Inf))),
    T2 = PerPlate(:T1 => PreserveNominalRatio(:T2, :T1),
                  :T2 => Uniform(0.02, 0.10)),
    ρ  = Truncated(Normal(1.0, 0.02), 0.0, 1.0),
)
```

### Train/eval split

Keep training and evaluation seeds **disjoint** and reuse a fixed evaluation pool
so the agent is never evaluated on a configuration it trained on:

```julia
eval_pool = eval_episodes(rp, 1:200)     # held-out; draw training seeds >= 10_000
```
