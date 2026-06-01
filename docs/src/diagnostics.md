# SNR Diagnostics

The diagnostics module provides NEMA MS-1 compliant SNR measurement for
reconstructed magnitude images. Three metrics are computed together so results
can be cross-checked against each other and against published scanner specs.

## The three metrics

### NEMA single-image (Method 4)

```
SNR = mean(signal in sphere ROI) / (std(background pixels) / 0.6551)
```

The `0.6551 = √((4−π)/2)` factor corrects for Rayleigh bias — the background
of a magnitude image of zero-mean complex Gaussian noise follows a Rayleigh
distribution, not a Gaussian one (Henkelman 1985; Gudbjartsson & Patz 1995).

### NEMA MS-1 dual-acquisition (recommended)

```
noise  = std( (A − B), pooled over signal ROIs ) / √2
signal = mean( (A+B)/2, per sphere ROI )
SNR    = signal / noise
```

Two identical acquisitions with independent noise realisations. The difference
image has zero mean (Rayleigh bias cancels) and structured background cancels
too. This is the gold standard for reproducible SNR and the metric to cite.

### Internal k-space SNR

```
SNR_ksp = √mean(|ksp|²) / σ
```

Non-standard — only meaningful when you injected the noise yourself and know σ
exactly. Useful as a calibration cross-check (`snr_ksp ≈ snr_dual_peak` at
high SNR).

## Quick usage

```julia
using MRISystemPhantom, KomaMRI, Random

cfg = PhantomConfig(field = :T3, voxel_size_mm = 2.0,
    include_plates = [:T1, :water],
    slice_thickness_mm = 10.0, slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)

seq       = ir_se_2d_sequence(0.4, 0.012, 3.0; FOV=0.2, Nfe=64, Npe=64)
raw_clean = simulate(obj, seq, scanner)
ksp_clean = raw_to_kspace(raw_clean, 64, 64)

# Sphere pixel positions and background mask
descs  = sphere_descriptors(:T1, cfg)
px     = sphere_descriptor_pixels(descs, 64, 64, 0.2)
bg     = background_mask(obj, 64, 64, 0.2; erosion_px = 2)

# Full SNR report from cached noise-free k-space
rep = snr_report_from_clean(ksp_clean, 0.005;
    sphere_px = px, bg_mask = bg, rng = MersenneTwister(42))

print_snr_report(rep)
# → prints dual-acq SNR, per-sphere SNR, temporal instability
```

## Multi-block schedules

When a sequence consists of multiple acquisition blocks (e.g. a TI sweep),
use `multi_block_snr_report` to pool noise statistics across blocks for a
tighter estimate:

```julia
reports = [snr_report_from_clean(ksp_clean[b], σ; sphere_px=px, bg_mask=bg, rng=rng)
           for b in 1:n_blocks]

imgs_a = [r.image for r in reports]   # see pooled_image_snr_report
```

See [`pooled_image_snr_report`](@ref) and [`MultiBlockSNRReport`](@ref) for
the full API.

## API

```@docs
SNRReport
ImageSNRReport
MultiBlockSNRReport
background_mask
nema_stats
dual_acq_stats
image_snr_report
snr_report
snr_report_from_clean
pooled_image_snr_report
print_snr_report
snr_report_to_dict
multi_block_snr_report_to_dict
```
