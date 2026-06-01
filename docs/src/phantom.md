# Phantom Construction

The central type is [`PhantomConfig`](@ref). Pass one to [`build_phantom`](@ref)
and you get a `KomaMRI.Phantom` ready for simulation. Everything — field strength,
voxel size, which plates to include, pose, noise — lives in the config.

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
spin count without affecting the contrast spheres.

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
