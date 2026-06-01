# API Reference

## Configuration

```@docs
PhantomConfig
AugmentConfig
SphereDescriptor
scanner_for_field
```

## Building phantoms

```@docs
build_phantom
build_plate
build_background_water
build_phantom_from_descriptors
build_sphere
```

## Sphere descriptors

```@docs
sphere_descriptors
all_sphere_descriptors
with_sphere_relaxation
transform_descriptor
sphere_descriptor_pixel
sphere_descriptor_pixels
```

## Sequences

```@docs
ir_sequence
se_sequence
mse_sequence
ir_se_2d_sequence
ir_tse_2d_sequence
se_2d_sequence
gre_2d_sequence
SpoilerConfig
apply_spoiler
rf_duration
single_spin_phantom
```

## Analytical forward models

```@docs
generalized_ir_signal
mse_signal
steady_state_mz_at_excite
transient_mz_at_excite_npe
transient_mz_per_shot
```

## Parameter fitters

```@docs
fit_t2_se
fit_t1_ir
fit_t1_generalized_ir
fit_t1_t2_generalized_ir
```

## Imaging

```@docs
raw_to_kspace
kspace_to_image
add_noise!
add_noise
phys_to_pixel
phys_to_pixel_wrap
roi_mean
```

## SNR diagnostics

See the [SNR Diagnostics](@ref) page for the full API and usage guide.

## Material tables

```@docs
T1_ARRAY
T2_ARRAY
T2_OF_T1_ARRAY
T1_OF_T2_ARRAY
T1_ARRAY_LEGACY
PD_FRACTIONS
BACKGROUND_WATER
FIDUCIAL_PROPS
PLATE_Z_MM
```

## Geometry

```@docs
contrast_plate_centres
fiducial_grid_centres
voxelise_sphere
sphere_volume
rotation_matrix
apply_transform!
Slab
slice_basis
signed_distance
voxelise_plane
```
