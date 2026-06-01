# MRISystemPhantom.jl

A programmatic, parameterised **digital twin** of the
[QalibreMD NIST/ISMRM System Standard Model 130](https://www.qalibremd.com/)
MRI phantom, built on [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl).

The library turns a single [`PhantomConfig`](@ref) into a `KomaMRI.Phantom`
that can be fed directly to `KomaMRI.simulate` — handling voxelisation of all
contrast spheres, the background water housing, optional pose randomisation,
and per-spin noise in a single call.

## Quick example

```julia
using MRISystemPhantom, KomaMRI

# Build the phantom at 3 T, 2 mm isotropic voxels
cfg = PhantomConfig(field = :T3, voxel_size_mm = 2.0)
obj = build_phantom(cfg)

# Matching scanner
scanner = scanner_for_field(cfg)

# Single inversion-recovery shot (TI = 400 ms)
seq = ir_sequence(0.4)

# Simulate (returns RawAcquisitionData)
raw = simulate(obj, seq, scanner)
```

## Features

- **Faithful geometry** — T1, T2, and PD contrast plates at their physical
  z-positions; fiducial grid; 100 mm water housing with sphere cutouts.
- **Material tables** at 1.5 T and 3 T for serial classes ≥0042 and legacy.
- **Flexible slicing** — isolate a single plate or a custom slab without
  modifying the phantom frame.
- **Domain randomisation** — fractional T1/T2 jitter, per-spin position/B0
  noise, and sphere dropout via [`AugmentConfig`](@ref).
- **Sequence library** — IR, SE, MSE, 2D IR-SE, 2D IR-TSE, 2D SE, 2D GRE.
- **Analytical forward models** — closed-form `generalized_ir_signal` and
  `mse_signal` for fast training loops.
- **Parameter fitters** — `fit_t1_generalized_ir` (generalised IR, profile
  likelihood / bootstrap σ), `fit_t1_t2_generalized_ir` (joint T1+T2),
  `fit_t2_se` (log-linear SE).

## Installation

This package is not yet registered. Add it directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/ArthurAllilaire/MRISystemPhantom.jl")
```

Or, in the Julia REPL package mode (`]`):

```
pkg> add https://github.com/ArthurAllilaire/MRISystemPhantom.jl
```

## Navigation

| Section | Contents |
|---------|----------|
| [Getting Started](@ref) | Installation, first simulation, reconstruction |
| [Phantom Construction](@ref) | `PhantomConfig` options, plates, slicing, augmentation |
| [Pulse Sequences](@ref) | Sequence builders and analytical forward models |
| [Parameter Fitting](@ref) | T1 / T2 fitters, uncertainty estimates |
| [API Reference](@ref) | Full docstring index |
