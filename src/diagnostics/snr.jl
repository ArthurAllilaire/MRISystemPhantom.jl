# SNR diagnostics for reconstructed magnitude images.
#
# Three metrics are provided:
#   - NEMA single-image (Method 4, NEMA MS-1 2014): mean(ROI) / (std(bg) / 0.6551)
#     The 0.6551 = √((4-π)/2) corrects for Rayleigh bias (Henkelman 1985).
#   - NEMA MS-1 dual-acquisition: mean((A+B)/2 in ROI) / (std(A-B in ROI) / √2)
#     Gold standard — no Rayleigh correction needed, structured background cancels.
#   - Internal k-space SNR: √mean(|ksp|²) / σ  (non-standard; calibration check only)

"""
    ImageSNRReport

Image-only SNR metrics, derived purely from two reconstructed magnitude
images A and B plus the sphere ROI / background-mask geometry. No k-space
or σ bookkeeping — see [`SNRReport`](@ref) for that.

Fields:
- `background_std_a`       — std of magnitude image A over background pixels
- `background_std_b`       — std of magnitude image B over background pixels
- `diff_roi_std`           — std of (A − B) pooled over signal ROIs; dual-acq noise
- `sphere_mean_a`          — per-sphere mean signal on img_a in ROI
- `sphere_mean_b`          — per-sphere mean signal on img_b in ROI
- `sphere_means`           — (sphere_mean_a + sphere_mean_b) / 2 (used by dual-acq)
- `temporal_instability`   — `|mean_a − mean_b| / sphere_means` per sphere
                              (fractional ballpark — closer to 0 is more stable)
- `snr_nema_per_sphere_a`  — `sphere_mean_a ./ (background_std_a / 0.6551)`
                              (NEMA Method 4, single-image A)
- `snr_nema_per_sphere_b`  — `sphere_mean_b ./ (background_std_b / 0.6551)`
- `snr_nema_peak_a`        — `max(snr_nema_per_sphere_a)`
- `snr_nema_peak_b`        — `max(snr_nema_per_sphere_b)`
- `snr_dual_per_sphere`    — `sphere_means ./ (diff_roi_std / √2)`
- `snr_dual_peak`          — `max(sphere_means) / (diff_roi_std / √2)`
"""
struct ImageSNRReport
    background_std_a::Float64
    background_std_b::Float64
    diff_roi_std::Float64
    sphere_mean_a::Vector{Float64}
    sphere_mean_b::Vector{Float64}
    sphere_means::Vector{Float64}
    temporal_instability::Vector{Float64}
    snr_nema_per_sphere_a::Vector{Float64}
    snr_nema_per_sphere_b::Vector{Float64}
    snr_nema_peak_a::Float64
    snr_nema_peak_b::Float64
    snr_dual_per_sphere::Vector{Float64}
    snr_dual_peak::Float64
end

"""
    SNRReport

Aggregated SNR metrics from one (or two) acquisition(s) of the same sequence.

Wraps an [`ImageSNRReport`](@ref) with the k-space / σ bookkeeping that
only makes sense when the caller injected complex Gaussian noise on
k-space themselves.

Fields:
- `image`      — the image-derived metrics (see [`ImageSNRReport`](@ref))
- `ksp_rms`    — √mean(|ksp_A|²)
- `sigma_used` — noise σ injected on k-space
- `snr_ksp`    — `ksp_rms / sigma_used` (internal calibration metric)
"""
struct SNRReport
    image::ImageSNRReport
    ksp_rms::Float64
    sigma_used::Float64
    snr_ksp::Float64
end

const RAYLEIGH_FACTOR = 0.6551  # √((4 − π)/2)

"""
    background_mask(phantom, Npe, Nfe, FOV; erosion_px=1) → BitMatrix

Build a per-pixel mask of background regions (phantom-free pixels) by
projecting phantom spins onto the image grid via `phantom_occupancy` and
selecting zero-occupancy pixels. Optional `erosion_px` removes a ring of
background pixels adjacent to the phantom, to avoid Gibbs / partial-volume
leakage from sphere edges contaminating the noise estimate.
"""
function background_mask(phantom, Npe::Int, Nfe::Int, FOV::Real; erosion_px::Int = 1)
    occ = phantom_occupancy(phantom, Npe, Nfe, FOV)
    bg = occ .== 0.0
    if erosion_px > 0
        # Shrink bg by `erosion_px`: any bg pixel within `erosion_px` of a
        # non-bg pixel (Chebyshev distance) is removed. Cheap O(N·k²) loop.
        bg_in = copy(bg)
        for i in 1:Npe, j in 1:Nfe
            bg_in[i, j] || continue
            adjacent = false
            for di in -erosion_px:erosion_px, dj in -erosion_px:erosion_px
                (di == 0 && dj == 0) && continue
                ii, jj = i + di, j + dj
                (1 <= ii <= Npe && 1 <= jj <= Nfe) || continue
                if !bg[ii, jj]
                    adjacent = true; break
                end
            end
            adjacent && (bg_in[i, j] = false)
        end
        return bg_in
    end
    bg
end

"""
    _roi_pixels(ipe, ife, roi_radius, Npe, Nfe) → Vector{CartesianIndex{2}}

Pixel indices in a `(2·roi_radius+1)²` square ROI clamped to image bounds.
`roi_radius=0` returns the single centre pixel.
"""
@inline function _roi_pixels(ipe::Int, ife::Int, roi_radius::Int, Npe::Int, Nfe::Int)
    pe_lo = clamp(ipe - roi_radius, 1, Npe)
    pe_hi = clamp(ipe + roi_radius, 1, Npe)
    fe_lo = clamp(ife - roi_radius, 1, Nfe)
    fe_hi = clamp(ife + roi_radius, 1, Nfe)
    [CartesianIndex(i, j) for i in pe_lo:pe_hi for j in fe_lo:fe_hi]
end

"""
    nema_stats(img, sphere_px, bg_mask; roi_radius=0) → NamedTuple

Single-acquisition NEMA SNR. Returns
`(; sphere_means, background_std, snr_per_sphere, snr_peak)`. Background std is
the raw std over `bg_mask` pixels; the `/0.6551` Rayleigh correction is applied
inside `snr_per_sphere`.
"""
function nema_stats(img::AbstractMatrix{<:Real},
                    sphere_px::Vector{NTuple{2,Int}},
                    bg_mask::AbstractMatrix{Bool}; roi_radius::Int = 0)
    Npe, Nfe = size(img)
    size(bg_mask) == (Npe, Nfe) ||
        error("bg_mask size $(size(bg_mask)) ≠ img size $((Npe, Nfe))")
    bg_vals = Float64.(img[bg_mask])
    isempty(bg_vals) && error("background mask is empty (no zero-occupancy pixels)")
    bg_std = Statistics.std(bg_vals; corrected = true)
    sphere_means = Float64[
        Statistics.mean(Float64.(img[_roi_pixels(p[1], p[2], roi_radius, Npe, Nfe)]))
        for p in sphere_px
    ]
    noise = bg_std / RAYLEIGH_FACTOR
    snr_per_sphere = sphere_means ./ max(noise, eps(Float64))
    (; sphere_means, background_std = bg_std,
       snr_per_sphere, snr_peak = maximum(snr_per_sphere))
end

"""
    dual_acq_stats(img_a, img_b, sphere_px; roi_radius=0) → NamedTuple

NEMA MS-1 dual-acquisition SNR. `img_a` and `img_b` are magnitude images from
two acquisitions of the same sequence with independent noise realisations.

Method (NEMA MS-1):
  noise = std( (A − B) restricted to **signal ROIs**, pooled across all
              spheres ) / √2
  signal_per_sphere = mean( (A + B)/2 in ROI )
  SNR_per_sphere    = signal_per_sphere / noise

Evaluating the difference std **inside the signal ROI** (not the background)
is what makes this method robust: at high SNR the magnitude image is locally
Gaussian around |S|, so A − B is zero-mean Gaussian with variance 2σ_img², and
dividing by √2 recovers σ_img. This eliminates both Rayleigh bias (which
would apply if we used the background) and any structured background.

Pooling the difference values across all sphere ROIs gives enough samples for
a reliable std even when each individual ROI is small (single-pixel default).
"""
function dual_acq_stats(img_a::AbstractMatrix{<:Real},
                        img_b::AbstractMatrix{<:Real},
                        sphere_px::Vector{NTuple{2,Int}}; roi_radius::Int = 0)
    Npe, Nfe = size(img_a)
    size(img_b) == size(img_a) ||
        error("img_a and img_b must have the same size")
    A = Float64.(img_a)
    B = Float64.(img_b)
    diff = A .- B

    # Pool diff values across all sphere ROIs for the noise estimate.
    diff_pool = Float64[]
    sphere_mean_a = Float64[]
    sphere_mean_b = Float64[]
    for p in sphere_px
        pix = _roi_pixels(p[1], p[2], roi_radius, Npe, Nfe)
        append!(diff_pool, diff[pix])
        push!(sphere_mean_a, Statistics.mean(A[pix]))
        push!(sphere_mean_b, Statistics.mean(B[pix]))
    end
    length(diff_pool) >= 2 || error("dual_acq_stats: pooled diff has < 2 samples")
    diff_roi_std = Statistics.std(diff_pool; corrected = true)
    noise = diff_roi_std / sqrt(2.0)

    sphere_means = (sphere_mean_a .+ sphere_mean_b) ./ 2.0
    snr_per_sphere = sphere_means ./ max(noise, eps(Float64))
    (; sphere_mean_a, sphere_mean_b, sphere_means, diff_roi_std,
       snr_per_sphere, snr_peak = maximum(snr_per_sphere))
end

"""
    image_snr_report(img_a, img_b, sphere_px, bg_mask; roi_radius=0) → ImageSNRReport

Pure function: build an `ImageSNRReport` from two reconstructed magnitude
images of the same sequence (independent noise realisations). Does no
simulation and has no k-space dependency — see [`snr_report`](@ref) for
the full pipeline with σ / ksp bookkeeping.
"""
function image_snr_report(img_a::AbstractMatrix{<:Real},
                          img_b::AbstractMatrix{<:Real},
                          sphere_px::Vector{NTuple{2,Int}},
                          bg_mask::AbstractMatrix{Bool};
                          roi_radius::Int = 0)
    size(img_a) == size(img_b) ||
        error("image_snr_report: img_a $(size(img_a)) ≠ img_b $(size(img_b))")

    s_a = nema_stats(img_a, sphere_px, bg_mask; roi_radius = roi_radius)
    s_b = nema_stats(img_b, sphere_px, bg_mask; roi_radius = roi_radius)
    d   = dual_acq_stats(img_a, img_b, sphere_px; roi_radius = roi_radius)

    temporal_instability = abs.(d.sphere_mean_a .- d.sphere_mean_b) ./
                           max.(d.sphere_means, eps(Float64))

    ImageSNRReport(
        Float64(s_a.background_std),
        Float64(s_b.background_std),
        Float64(d.diff_roi_std),
        Float64.(d.sphere_mean_a),
        Float64.(d.sphere_mean_b),
        Float64.(d.sphere_means),
        Float64.(temporal_instability),
        Float64.(s_a.snr_per_sphere),
        Float64.(s_b.snr_per_sphere),
        Float64(s_a.snr_peak),
        Float64(s_b.snr_peak),
        Float64.(d.snr_per_sphere),
        Float64(d.snr_peak),
    )
end

"""
    snr_report(phantom, seq, scanner; σ, sphere_px, bg_mask, ksp_a, img_a,
                rng, phase_sensitive=false, roi_radius=0) → SNRReport

Build a full `SNRReport`. Caller must already have:
- `ksp_a` — first acquisition's k-space (post-noise)
- `img_a` — first acquisition's reconstructed magnitude image

Runs **one** additional `simulate(phantom, seq, scanner)` to produce
acquisition B, adds independent noise (`add_noise!` with σ), reconstructs via
`kspace_to_image`, and returns all three SNR metrics. Prefer
[`snr_report_from_clean`](@ref) when a clean k-space is already available.
"""
function snr_report(phantom, seq, scanner;
                    σ::Real,
                    sphere_px::Vector{NTuple{2,Int}},
                    bg_mask::AbstractMatrix{Bool},
                    ksp_a::AbstractMatrix{<:Complex},
                    img_a::AbstractMatrix{<:Real},
                    rng::AbstractRNG,
                    phase_sensitive::Bool = false,
                    roi_radius::Int = 0)
    Npe, Nfe = size(ksp_a)
    raw_b = Suppressor.@suppress simulate(phantom, seq, scanner)
    ksp_b = raw_to_kspace(raw_b, Npe, Nfe)
    add_noise!(ksp_b, Float64(σ); rng = rng)
    img_b = kspace_to_image(ksp_b; phase_sensitive = phase_sensitive)

    ksp_rms = sqrt(sum(abs2, ksp_a) / length(ksp_a))
    snr_ksp = σ > 0 ? ksp_rms / Float64(σ) : 0.0

    image = image_snr_report(img_a, img_b, sphere_px, bg_mask;
                             roi_radius = roi_radius)

    SNRReport(image, Float64(ksp_rms), Float64(σ), Float64(snr_ksp))
end

"""
    _make_ab_images(ksp_clean, σ; rng, phase_sensitive=false) → (; img_a, img_b)

Build two reconstructed magnitude images from a noise-free k-space by
adding two independent complex Gaussian noise realisations of magnitude
`σ` and reconstructing each. Private — shared by
[`snr_report_from_clean`](@ref) and the multi-block pooling path so the
A/B images aren't built twice.
"""
function _make_ab_images(ksp_clean::AbstractMatrix{<:Complex}, σ::Real;
                          rng::AbstractRNG, phase_sensitive::Bool = false)
    ksp_a = add_noise(ksp_clean, Float64(σ); rng = rng)
    ksp_b = add_noise(ksp_clean, Float64(σ); rng = rng)
    img_a = kspace_to_image(ksp_a; phase_sensitive = phase_sensitive)
    img_b = kspace_to_image(ksp_b; phase_sensitive = phase_sensitive)
    (; img_a, img_b)
end

"""
    snr_report_from_clean(ksp_clean, σ; sphere_px, bg_mask, rng,
                          phase_sensitive=false, roi_radius=0) → SNRReport

Build an `SNRReport` from a CACHED noise-free k-space. Adds two
independent noise realisations (A, B), reconstructs both, and computes
the standard image-domain SNR metrics. No `simulate()` call — for sweeps
that already cache a per-block noise-free `ksp_clean`.

`ksp_rms` is computed from the CLEAN k-space (the true signal RMS), not
from a noisy realisation. `RMS(S+N)² ≈ RMS(S)² + 2σ²` for independent
zero-mean complex Gaussian noise, so using a noisy ksp here would
inflate `snr_ksp` at low SNR. The companion entry point
[`snr_report`](@ref) (which doesn't have the clean ksp) intentionally
uses the noisy version — that mirrors how SNR is estimated on real
scanners.
"""
function snr_report_from_clean(ksp_clean::AbstractMatrix{<:Complex}, σ::Real;
                                sphere_px::Vector{NTuple{2,Int}},
                                bg_mask::AbstractMatrix{Bool},
                                rng::AbstractRNG,
                                phase_sensitive::Bool = false,
                                roi_radius::Int = 0)
    ab      = _make_ab_images(ksp_clean, σ; rng = rng,
                              phase_sensitive = phase_sensitive)
    ksp_rms = sqrt(sum(abs2, ksp_clean) / length(ksp_clean))
    snr_ksp = σ > 0 ? ksp_rms / Float64(σ) : 0.0
    image   = image_snr_report(ab.img_a, ab.img_b, sphere_px, bg_mask;
                               roi_radius = roi_radius)
    SNRReport(image, Float64(ksp_rms), Float64(σ), Float64(snr_ksp))
end

"""
    MultiBlockSNRReport

Per-block + pooled SNR for a multi-block schedule. Each block has its
own (A, B) noise realisation. The `pooled` field carries one
`ImageSNRReport` whose noise std is taken over (A−B) samples *pooled
across every block* — a tighter noise estimate than any single block can
give — and whose per-sphere signal is the mean across blocks (the
"schedule-average" signal each sphere receives).

Fields:
- `per_block`            — `Vector{SNRReport}` length `n_blocks`
- `pooled`               — single `ImageSNRReport` (cross-block pooling)
- `block_snr_dual_peak`  — convenience: `[r.image.snr_dual_peak for r in per_block]`
- `block_snr_nema_peak`  — convenience: `[r.image.snr_nema_peak_a for r in per_block]`
"""
struct MultiBlockSNRReport
    per_block::Vector{SNRReport}
    pooled::ImageSNRReport
    block_snr_dual_peak::Vector{Float64}
    block_snr_nema_peak::Vector{Float64}
end

"""
    pooled_image_snr_report(imgs_a, imgs_b, sphere_px, bg_mask;
                            roi_radius=0) → ImageSNRReport

Pool dual-acquisition statistics across a stack of `n_blocks` (img_a,
img_b) pairs from the same schedule. Returns one `ImageSNRReport`:

* `diff_roi_std`     — std of (A−B) pooled across every block's signal ROIs.
                       Sample count is `n_blocks · N_ROI` (tighter than any
                       single block).
* `background_std_*` — std of magnitude image over background pixels,
                       pooled across blocks (concat then std).
* `sphere_mean_*`    — per-sphere mean signal averaged across blocks
                       (the "schedule-mean" signal).
* `snr_dual_per_sphere` = `sphere_means ./ (pooled_diff_std / √2)`
* `snr_dual_peak`    = `max(snr_dual_per_sphere)`

Uses the same `RAYLEIGH_FACTOR` correction as `nema_stats` for the
single-image NEMA metrics.
"""
function pooled_image_snr_report(imgs_a::AbstractVector{<:AbstractMatrix{<:Real}},
                                  imgs_b::AbstractVector{<:AbstractMatrix{<:Real}},
                                  sphere_px::Vector{NTuple{2,Int}},
                                  bg_mask::AbstractMatrix{Bool};
                                  roi_radius::Int = 0)
    length(imgs_a) == length(imgs_b) ||
        error("pooled_image_snr_report: imgs_a/imgs_b length mismatch")
    isempty(imgs_a) && error("pooled_image_snr_report: empty image stack")
    Npe, Nfe = size(imgs_a[1])
    n_blocks = length(imgs_a)
    n_spheres = length(sphere_px)

    # Pooled diff samples across the signal ROIs of every block.
    diff_pool = Float64[]
    # Per-block per-sphere means, then averaged across blocks.
    sm_a = zeros(Float64, n_spheres)
    sm_b = zeros(Float64, n_spheres)
    # Pooled background pixel values across all blocks (one std each for A/B).
    bg_a = Float64[]
    bg_b = Float64[]

    for k in 1:n_blocks
        A = Float64.(imgs_a[k])
        B = Float64.(imgs_b[k])
        size(A) == (Npe, Nfe) == size(B) ||
            error("pooled_image_snr_report: block $k image size mismatch")
        diff = A .- B
        for (i, p) in enumerate(sphere_px)
            pix = _roi_pixels(p[1], p[2], roi_radius, Npe, Nfe)
            append!(diff_pool, diff[pix])
            sm_a[i] += Statistics.mean(A[pix])
            sm_b[i] += Statistics.mean(B[pix])
        end
        append!(bg_a, A[bg_mask])
        append!(bg_b, B[bg_mask])
    end
    sm_a ./= n_blocks
    sm_b ./= n_blocks
    sphere_means = (sm_a .+ sm_b) ./ 2.0

    length(diff_pool) >= 2 ||
        error("pooled_image_snr_report: pooled diff has < 2 samples")
    diff_roi_std = Statistics.std(diff_pool; corrected = true)
    bg_std_a = isempty(bg_a) ? 0.0 : Statistics.std(bg_a; corrected = true)
    bg_std_b = isempty(bg_b) ? 0.0 : Statistics.std(bg_b; corrected = true)

    noise_dual = diff_roi_std / sqrt(2.0)
    snr_dual_per_sphere = sphere_means ./ max(noise_dual, eps(Float64))
    noise_nema_a = bg_std_a / RAYLEIGH_FACTOR
    noise_nema_b = bg_std_b / RAYLEIGH_FACTOR
    snr_nema_per_sphere_a = sm_a ./ max(noise_nema_a, eps(Float64))
    snr_nema_per_sphere_b = sm_b ./ max(noise_nema_b, eps(Float64))
    temporal_instability = abs.(sm_a .- sm_b) ./ max.(sphere_means, eps(Float64))

    ImageSNRReport(
        bg_std_a, bg_std_b, diff_roi_std,
        sm_a, sm_b, sphere_means,
        temporal_instability,
        snr_nema_per_sphere_a, snr_nema_per_sphere_b,
        maximum(snr_nema_per_sphere_a), maximum(snr_nema_per_sphere_b),
        snr_dual_per_sphere, maximum(snr_dual_per_sphere),
    )
end

"""
    multi_block_snr_report_to_dict(rep::MultiBlockSNRReport) → Dict{String,Any}

Serialise a `MultiBlockSNRReport` to a JSON-friendly dict. `per_block` is
a list of per-block dicts (one per [`snr_report_to_dict`](@ref) call);
`pooled` flattens the pooled `ImageSNRReport`; the rest are convenience
scalars.
"""
function multi_block_snr_report_to_dict(rep::MultiBlockSNRReport)
    pooled = rep.pooled
    pooled_dict = Dict{String,Any}(
        "background_std_a"      => pooled.background_std_a,
        "background_std_b"      => pooled.background_std_b,
        "diff_roi_std"          => pooled.diff_roi_std,
        "sphere_mean_a"         => pooled.sphere_mean_a,
        "sphere_mean_b"         => pooled.sphere_mean_b,
        "sphere_means"          => pooled.sphere_means,
        "temporal_instability"  => pooled.temporal_instability,
        "snr_nema_per_sphere_a" => pooled.snr_nema_per_sphere_a,
        "snr_nema_per_sphere_b" => pooled.snr_nema_per_sphere_b,
        "snr_nema_peak_a"       => pooled.snr_nema_peak_a,
        "snr_nema_peak_b"       => pooled.snr_nema_peak_b,
        "snr_dual_per_sphere"   => pooled.snr_dual_per_sphere,
        "snr_dual_peak"         => pooled.snr_dual_peak,
    )
    Dict{String,Any}(
        "n_blocks"            => length(rep.per_block),
        "per_block"           => [snr_report_to_dict(r) for r in rep.per_block],
        "pooled"              => pooled_dict,
        "block_snr_dual_peak" => rep.block_snr_dual_peak,
        "block_snr_nema_peak" => rep.block_snr_nema_peak,
    )
end

"""
    print_snr_report(io, rep::SNRReport; label="SNR report")

One-screen pretty-print for stdout / log files.
"""
function print_snr_report(io::IO, rep::SNRReport; label::AbstractString = "SNR report")
    img = rep.image
    println(io, label, "  (σ = ", round(rep.sigma_used, sigdigits = 4), ")")
    println(io, "  ksp_rms           = ", round(rep.ksp_rms, sigdigits = 5),
                "    snr_ksp        = ", round(rep.snr_ksp, sigdigits = 4),
                "   (calibration knob, non-standard)")
    println(io, "  background_std_a  = ", round(img.background_std_a, sigdigits = 5),
                "    snr_nema_peak_a = ", round(img.snr_nema_peak_a, sigdigits = 4),
                "  (NEMA single-image, A)")
    println(io, "  background_std_b  = ", round(img.background_std_b, sigdigits = 5),
                "    snr_nema_peak_b = ", round(img.snr_nema_peak_b, sigdigits = 4),
                "  (NEMA single-image, B)")
    println(io, "  diff_roi_std      = ", round(img.diff_roi_std, sigdigits = 5),
                "    snr_dual_peak   = ", round(img.snr_dual_peak, sigdigits = 4),
                "  (NEMA MS-1 dual-acq)")
    println(io, "  per-sphere snr_dual         = ", round.(img.snr_dual_per_sphere, digits = 2))
    println(io, "  per-sphere temporal_instab. = ", round.(img.temporal_instability, digits = 4))
end
print_snr_report(rep::SNRReport; kwargs...) = print_snr_report(stdout, rep; kwargs...)

"""
    snr_report_to_dict(rep::SNRReport) → Dict{String,Any}

Serialise an `SNRReport` to a JSON-friendly `Dict{String,Any}`.
"""
function snr_report_to_dict(rep::SNRReport)
    img = rep.image
    Dict{String,Any}(
        "ksp_rms"             => rep.ksp_rms,
        "sigma_used"          => rep.sigma_used,
        "snr_ksp"               => rep.snr_ksp,
        "background_std_a"      => img.background_std_a,
        "background_std_b"      => img.background_std_b,
        "diff_roi_std"          => img.diff_roi_std,
        "sphere_mean_a"         => img.sphere_mean_a,
        "sphere_mean_b"         => img.sphere_mean_b,
        "sphere_means"          => img.sphere_means,
        "temporal_instability"  => img.temporal_instability,
        "snr_nema_per_sphere_a" => img.snr_nema_per_sphere_a,
        "snr_nema_per_sphere_b" => img.snr_nema_per_sphere_b,
        "snr_nema_peak_a"       => img.snr_nema_peak_a,
        "snr_nema_peak_b"       => img.snr_nema_peak_b,
        "snr_dual_per_sphere"   => img.snr_dual_per_sphere,
        "snr_dual_peak"         => img.snr_dual_peak,
    )
end
