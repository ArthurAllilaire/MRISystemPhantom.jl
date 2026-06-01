# Pulse Sequences

All sequence builders return a `KomaMRI.Sequence` that can be passed directly
to `KomaMRI.simulate`. Analytical forward models (prefixed with the signal
they predict) are provided for use in fast training loops.

## Single-shot sequences

### Inversion Recovery

```julia
seq = ir_sequence(TI;
    amp_T    = 2e-6,   # RF amplitude (T)
    n_adc    = 16,     # ADC samples
    dur_adc  = 2e-3,   # ADC window (s)
)
```

180° inversion → delay `TI` → 90° excitation → ADC. The first ADC sample
magnitude tracks `|1 − 2·exp(−TI/T1)|` for a fully recovered spin.

**Analytical equivalent:**

```julia
sig = generalized_ir_signal(T1, T2; TI = 0.4, α = π)
```

`α = π` is canonical IR; `α = π/2` is saturation recovery; any angle works.

### Spin Echo

```julia
seq = se_sequence(TE;
    amp_T   = 20e-6,
    n_adc   = 32,
    dur_adc = min(2e-3, TE/4),
)
```

90° → delay TE/2 → 180° → delay TE/2 → ADC. The echo peak tracks
`S0·exp(−TE/T2)`.

### Multi-Echo Spin Echo (CPMG)

```julia
seq = mse_sequence(ESP, n_echoes;
    amp_T   = 20e-6,
    n_adc   = 1,
    dur_adc = min(1e-3, ESP/4),
)
```

One 90° excitation followed by `n_echoes` 180° refocusing pulses. Echo `k`
forms at `t = k·ESP` and has magnitude `S0·exp(−k·ESP/T2)`. Samples the full
T2 decay curve in a single TR.

**Analytical equivalent:**

```julia
sig = mse_signal(T2; ESP = 0.02, n_echoes = 8)
```

## 2D Cartesian sequences

All 2D sequences share the same k-space convention and reconstruction path:

```julia
ksp   = raw_to_kspace(raw, Npe, Nfe)
image = abs.(kspace_to_image(ksp))   # magnitude image
```

Sphere centre `(x, y)` in metres maps to pixel `(i_pe, i_fe)`:

```julia
i_fe = mod(round(Int, x * Nfe / FOV), Nfe) + 1
i_pe = mod(round(Int, y * Npe / FOV), Npe) + 1
```

Or use [`sphere_descriptor_pixel`](@ref) / [`sphere_descriptor_pixels`](@ref).

### 2D Spin-Echo IR (`ir_se_2d_sequence`)

```julia
seq = ir_se_2d_sequence(TI, TE, TR;
    α_exc   = π/2,
    FOV     = 0.2,     # m
    Nfe     = 64,      # frequency-encode steps
    Npe     = 64,      # phase-encode steps (= number of shots)
    amp_T   = 20e-6,
    spoiler = SpoilerConfig(),
)
```

Multi-shot 2D spin-echo with inversion prep. Each of the `Npe` shots
acquires one k-space row. Combine with `fit_t1_generalized_ir` (pass
`Npe = Npe`) to account for the transient magnetisation ramp.

Enable crushers around the 180° to suppress FID contamination at short TR:

```julia
spoiler = SpoilerConfig(enabled = true, amp_T = 30e-3, dur = 5e-3, axis = :z)
seq = ir_se_2d_sequence(TI, TE, TR; spoiler = spoiler)
```

### 2D Inversion-Recovery Turbo Spin Echo (`ir_tse_2d_sequence`)

```julia
seq = ir_tse_2d_sequence(TI, esp, TR;
    etl   = 4,    # echo train length — must divide Npe
    FOV   = 0.2,
    Nfe   = 64,
    Npe   = 64,
    amp_T = 20e-6,
    spoiler = SpoilerConfig(),
)
```

`etl` echoes per shot → `Npe ÷ etl` shots total (≈`etl`× faster scan).
Meiboom–Gill y-phase 180°s prevent alternating echo-sign artefacts.

### 2D Spin Echo (`se_2d_sequence`)

```julia
seq = se_2d_sequence(TE, TR; FOV = 0.2, Nfe = 64, Npe = 64, amp_T = 20e-6)
```

IR block dropped; use for T2 mapping by sweeping TE.

### 2D Gradient Echo (`gre_2d_sequence`)

```julia
seq = gre_2d_sequence(TE, TR;
    α   = deg2rad(15),   # Ernst angle for T1 contrast
    FOV = 0.2,
    Nfe = 64,
    Npe = 64,
    spoiler = SpoilerConfig(),
)
```

T2*-weighted (faster T2 decay than SE). Gradient spoiling only; add
quadratic RF phase increments externally for fully-spoiled FLASH behaviour.

## Spoiler configuration

```julia
SpoilerConfig(
    enabled = false,
    amp_T   = 30e-3,    # spoiler gradient amplitude (T/m)
    dur     = 5e-3,     # duration (s)
    axis    = :z,       # :x, :y, or :z
)
```

## Utility

```julia
# RF pulse duration for a given flip angle and B1 amplitude
dur = rf_duration(π/2; amp_T = 2e-6)

# Minimal single-spin phantom for fast non-spatial tests
spin = single_spin_phantom(T1 = 1.2, T2 = 0.1)
```
