# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ArthurAllilaire/MRISystemPhantom.jl")
```

The package depends on [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
which will be installed automatically.

## Your first simulation

The three-line path to a simulated IR signal:

```julia
using MRISystemPhantom, KomaMRI

# 1. Build the phantom (3 T, 2 mm voxels, T1 plate only)
cfg = PhantomConfig(
    field            = :T3,
    voxel_size_mm    = 2.0,
    include_plates   = [:T1, :water],
    slice_thickness_mm = 10.0,
    slice_center_mm  = (0.0, 0.0, PLATE_Z_MM.T1),
)
obj = build_phantom(cfg)

# 2. Build a matching scanner and a sequence
scanner = scanner_for_field(cfg)
seq     = ir_sequence(0.4)         # TI = 400 ms

# 3. Simulate
raw = simulate(obj, seq, scanner)
signal = abs(raw.profiles[1].data[1, 1])   # first ADC sample magnitude
```

## Sweeping TI to measure T1

```julia
TIs    = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]   # seconds
# single spin — fastest possible target
spin   = single_spin_phantom(T1 = 1.2, T2 = 0.1)
mags   = Float64[]
for TI in TIs
    raw = simulate(spin, ir_sequence(TI), scanner_for_field(:T3))
    push!(mags, abs(raw.profiles[1].data[1, 1]))
end

result = fit_t1_ir(TIs, mags)
@show result.T1   # ≈ 1.2 s
```

## 2D imaging: T1 mapping of the T1 plate

```julia
cfg = PhantomConfig(
    field              = :T3,
    voxel_size_mm      = 2.0,
    include_plates     = [:T1, :water],
    slice_thickness_mm = 10.0,
    slice_center_mm    = (0.0, 0.0, PLATE_Z_MM.T1),
)
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)

TIs   = [0.1, 0.4, 1.0, 2.5]
images = []
for TI in TIs
    seq = ir_se_2d_sequence(TI, 0.012, 3.0; FOV = 0.2, Nfe = 64, Npe = 64)
    raw = simulate(obj, seq, scanner)
    ksp = raw_to_kspace(raw, 64, 64)
    push!(images, abs.(kspace_to_image(ksp)))
end

# images[i] is a 64×64 magnitude image for TIs[i]
# fit pixel-wise with fit_t1_ir or fit_t1_generalized_ir
```

## Analytical forward model (fast training path)

For RL training loops where calling `KomaMRI.simulate` every step is too
slow, use the closed-form forward model instead:

```julia
# Orders of magnitude faster — exact for a single-spin phantom
sig = generalized_ir_signal(1.2, 0.1; TI = 0.4, α = π)
# Returns a vector of n_adc magnitude samples decaying with T2
```
