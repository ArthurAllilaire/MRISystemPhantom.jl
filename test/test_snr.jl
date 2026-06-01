# Pure-Julia tests for src/diagnostics/snr.jl. Uses synthetic magnitude
# images of known signal + Gaussian noise so the expected statistics are
# closed-form. No KomaMRI `simulate(...)` calls — the only piece that
# needs a real phantom is `background_mask`, which we exercise on a
# coarse, fast `PhantomConfig(voxel_size_mm = 6.0)` build.

using Statistics

const _RAY = MRISystemPhantom.RAYLEIGH_FACTOR
const _roi_pixels_jl = MRISystemPhantom._roi_pixels

"Synthesise a magnitude image: per-sphere constant signal + magnitude of
zero-mean complex Gaussian noise (Rayleigh background). Returns
`(img, sphere_px, signals)` where `sphere_px[i]` is the (row,col) of the
centre pixel of sphere i and `signals[i]` is the planted true signal."
function _synth_image(rng::AbstractRNG; Npe = 64, Nfe = 64, σ = 0.1,
                       signals = [3.0, 2.0, 1.0])
    # Pure Rayleigh-magnitude background everywhere
    re = σ .* randn(rng, Npe, Nfe)
    im = σ .* randn(rng, Npe, Nfe)
    img = sqrt.(re.^2 .+ im.^2)
    # Plant sphere signals at well-separated pixels
    sphere_px = NTuple{2,Int}[(16, 16), (32, 32), (48, 48)]
    for (i, p) in enumerate(sphere_px)
        # Locally Gaussian around |S| at high SNR: |S + n| ≈ S + Re(n)·sign(S)
        img[p[1], p[2]] = abs(signals[i] + σ * randn(rng))
    end
    img, sphere_px, signals[firstindex(sphere_px):lastindex(sphere_px)]
end

"Background mask that excludes a small disc around each planted sphere."
function _bg_mask_avoiding(sphere_px, Npe, Nfe; pad = 3)
    bg = trues(Npe, Nfe)
    for p in sphere_px, di in -pad:pad, dj in -pad:pad
        i, j = p[1] + di, p[2] + dj
        (1 <= i <= Npe && 1 <= j <= Nfe) || continue
        bg[i, j] = false
    end
    bg
end

@testset "snr.jl" begin

    @testset "RAYLEIGH_FACTOR" begin
        @test isapprox(_RAY, sqrt((4 - π) / 2); atol = 1e-3)
    end

    @testset "_roi_pixels" begin
        # radius 0 → single centre pixel
        @test length(_roi_pixels_jl(10, 10, 0, 64, 64)) == 1
        @test _roi_pixels_jl(10, 10, 0, 64, 64)[1] == CartesianIndex(10, 10)
        # radius 1 → 3x3 = 9 in interior
        @test length(_roi_pixels_jl(10, 10, 1, 64, 64)) == 9
        # corner clamp: top-left corner with r=1 → 2x2 = 4 pixels, no OOB
        pix = _roi_pixels_jl(1, 1, 1, 64, 64)
        @test length(pix) == 4
        @test all(1 <= p[1] <= 64 && 1 <= p[2] <= 64 for p in pix)
    end

    @testset "background_mask" begin
        # Coarse phantom — same trick as test_builder.jl
        cfg = PhantomConfig(voxel_size_mm = 6.0)
        phantom = build_phantom(cfg)
        Npe, Nfe = 32, 32
        FOV = 0.2

        bg0 = background_mask(phantom, Npe, Nfe, FOV; erosion_px = 0)
        bg1 = background_mask(phantom, Npe, Nfe, FOV; erosion_px = 1)
        @test size(bg0) == (Npe, Nfe)
        @test eltype(bg0) === Bool
        # Some background pixels must exist (phantom does not fill the grid)
        @test count(bg0) > 0
        # Erosion can only remove pixels — never add
        @test count(bg1) <= count(bg0)
        # And the erosion mask must be a strict subset of the raw mask
        @test all(bg1 .<= bg0)
    end

    @testset "nema_stats" begin
        rng = MersenneTwister(0)
        σ = 0.1
        Npe, Nfe = 64, 64
        signals = [3.0, 2.0, 1.0]
        img, sphere_px, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                         σ = σ, signals = signals)
        bg = _bg_mask_avoiding(sphere_px, Npe, Nfe)

        s = nema_stats(img, sphere_px, bg)
        # Raw std of Rayleigh-distributed magnitudes ≈ σ·√((4−π)/2) = σ·RAYLEIGH_FACTOR
        @test isapprox(s.background_std, σ * _RAY; rtol = 0.05)
        # snr_per_sphere ≈ signal / (σ·RAYLEIGH_FACTOR / RAYLEIGH_FACTOR) = signal / σ
        for i in eachindex(signals)
            @test isapprox(s.snr_per_sphere[i], signals[i] / σ; rtol = 0.10)
        end
        @test s.snr_peak == maximum(s.snr_per_sphere)
        # Errors when bg_mask size mismatches img
        @test_throws ErrorException nema_stats(img, sphere_px, falses(Npe, Nfe + 1))
    end

    @testset "dual_acq_stats" begin
        rng = MersenneTwister(1)
        σ = 0.1
        Npe, Nfe = 64, 64
        signals = [3.0, 2.0, 1.0]
        img_a, sphere_px, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                           σ = σ, signals = signals)
        img_b, _, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                   σ = σ, signals = signals)

        d = dual_acq_stats(img_a, img_b, sphere_px; roi_radius = 1)
        # diff in signal ROI is zero-mean Gaussian with σ_diff = σ·√2 since
        # each image has additive σ in its ROI (high-SNR locally Gaussian).
        @test isapprox(d.diff_roi_std, σ * sqrt(2); rtol = 0.30)
        # sphere_means must be exactly (mean_a + mean_b)/2
        @test d.sphere_means == (d.sphere_mean_a .+ d.sphere_mean_b) ./ 2.0
        @test d.snr_peak == maximum(d.snr_per_sphere)
        # Errors when image sizes mismatch
        @test_throws ErrorException dual_acq_stats(img_a, zeros(Npe + 1, Nfe), sphere_px)
    end

    @testset "image_snr_report" begin
        rng = MersenneTwister(2)
        σ = 0.1
        Npe, Nfe = 64, 64
        signals = [3.0, 2.0, 1.0]
        img_a, sphere_px, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                           σ = σ, signals = signals)
        img_b, _, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                   σ = σ, signals = signals)
        bg = _bg_mask_avoiding(sphere_px, Npe, Nfe)

        rep = image_snr_report(img_a, img_b, sphere_px, bg; roi_radius = 1)
        @test rep isa ImageSNRReport
        # Peak fields match max over per-sphere
        @test rep.snr_nema_peak_a == maximum(rep.snr_nema_per_sphere_a)
        @test rep.snr_nema_peak_b == maximum(rep.snr_nema_per_sphere_b)
        @test rep.snr_dual_peak   == maximum(rep.snr_dual_per_sphere)
        # Temporal instability should be small (~σ/signal): brightest sphere
        # ≈ 0.1/3 ≈ 3%. Allow a loose bound — this is a sanity test, not a
        # statistical convergence check.
        @test maximum(rep.temporal_instability) < 0.20
        # Size mismatch surfaces from image_snr_report itself
        @test_throws ErrorException image_snr_report(img_a, zeros(Npe + 1, Nfe),
                                                     sphere_px, bg)
    end

    @testset "SNRReport nesting" begin
        rng = MersenneTwister(3)
        img_a, sphere_px, _ = _synth_image(rng; σ = 0.1)
        img_b, _, _         = _synth_image(rng; σ = 0.1)
        bg = _bg_mask_avoiding(sphere_px, size(img_a, 1), size(img_a, 2))
        image = image_snr_report(img_a, img_b, sphere_px, bg)
        rep = SNRReport(image, 1.0, 0.5, 2.0)

        @test rep.image isa ImageSNRReport
        @test rep.ksp_rms    == 1.0
        @test rep.sigma_used == 0.5
        @test rep.snr_ksp    == 2.0
        @test rep.image.snr_dual_peak === image.snr_dual_peak
    end

    @testset "snr_report_to_dict schema" begin
        rng = MersenneTwister(4)
        img_a, sphere_px, _ = _synth_image(rng; σ = 0.1)
        img_b, _, _         = _synth_image(rng; σ = 0.1)
        bg = _bg_mask_avoiding(sphere_px, size(img_a, 1), size(img_a, 2))
        rep = SNRReport(image_snr_report(img_a, img_b, sphere_px, bg),
                        1.0, 0.5, 2.0)

        d = snr_report_to_dict(rep)
        expected = Set([
            "ksp_rms", "sigma_used", "snr_ksp",
            "background_std_a", "background_std_b", "diff_roi_std",
            "sphere_mean_a", "sphere_mean_b", "sphere_means",
            "temporal_instability",
            "snr_nema_per_sphere_a", "snr_nema_per_sphere_b",
            "snr_nema_peak_a", "snr_nema_peak_b",
            "snr_dual_per_sphere", "snr_dual_peak",
        ])
        @test Set(keys(d)) == expected
        # One ksp-derived spot-check + one image-derived spot-check
        @test d["ksp_rms"]         == 1.0
        @test d["snr_dual_peak"]   == rep.image.snr_dual_peak
    end

    @testset "add_noise (non-mutating)" begin
        rng1 = MersenneTwister(123)
        rng2 = MersenneTwister(123)
        ksp = ComplexF32.(reshape(1:32, 4, 8))
        ksp_before = copy(ksp)
        out  = add_noise(ksp, 0.1; rng = rng1)
        @test ksp == ksp_before                  # input untouched
        # Same RNG seed → bitwise equal to the in-place path.
        ksp_inplace = copy(ksp)
        add_noise!(ksp_inplace, 0.1; rng = rng2)
        @test out == ksp_inplace
        # σ=0 → unchanged copy
        @test add_noise(ksp, 0.0; rng = rng1) == ksp
    end

    @testset "snr_report_from_clean" begin
        # Build a small noise-free k-space whose magnitude is known.
        rng = MersenneTwister(7)
        Npe, Nfe = 32, 32
        ksp_clean = ComplexF32.(randn(rng, ComplexF32, Npe, Nfe))
        sphere_px = NTuple{2,Int}[(8, 8), (16, 16), (24, 24)]
        bg = _bg_mask_avoiding(sphere_px, Npe, Nfe; pad = 2)

        # σ = 1.0
        σ = 1.0
        rep = snr_report_from_clean(ksp_clean, σ;
                sphere_px = sphere_px, bg_mask = bg,
                rng = MersenneTwister(99))
        # ksp_rms is computed on the CLEAN k-space, not on a noisy one.
        ksp_rms_clean = sqrt(sum(abs2, ksp_clean) / length(ksp_clean))
        @test rep.ksp_rms ≈ ksp_rms_clean
        @test rep.sigma_used == σ
        @test rep.snr_ksp ≈ ksp_rms_clean / σ
        @test rep.image isa ImageSNRReport

        # Hand-built equivalence: same RNG → same A/B → same metrics.
        rng_a = MersenneTwister(99)
        ksp_a = add_noise(ksp_clean, σ; rng = rng_a)
        ksp_b = add_noise(ksp_clean, σ; rng = rng_a)
        img_a = kspace_to_image(ksp_a; phase_sensitive = false)
        img_b = kspace_to_image(ksp_b; phase_sensitive = false)
        manual = image_snr_report(img_a, img_b, sphere_px, bg)
        @test rep.image.diff_roi_std ≈ manual.diff_roi_std
        @test rep.image.snr_dual_peak ≈ manual.snr_dual_peak

        # σ=0 → diff is floating-point noise only; ksp_rms / snr_ksp follow
        # the docstring contract (snr_ksp == 0 when σ == 0).
        rep0 = snr_report_from_clean(ksp_clean, 0.0;
                sphere_px = sphere_px, bg_mask = bg,
                rng = MersenneTwister(99))
        @test rep0.snr_ksp == 0.0
        @test rep0.image.diff_roi_std <= eps(Float32) * 10
    end

    @testset "pooled_image_snr_report" begin
        rng = MersenneTwister(11)
        Npe, Nfe = 64, 64
        # Stack of identical (img_a, img_b) pairs → pooled diff std should
        # ≈ per-block diff std (concatenating identical samples doesn't
        # change the spread). Exact equality fails because `Statistics.std`
        # uses Bessel's correction (denominator N-1): for N blocks of M
        # samples each, std_pooled² = single_std² · N(M-1)/(NM-1). Pick
        # roi_radius=3 so each ROI is 49 pixels — with 3 spheres × 3 blocks
        # the correction drops to <0.5%.
        img_a, sphere_px, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                           σ = 0.1, signals = [3.0, 2.0, 1.0])
        img_b, _, _         = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                           σ = 0.1, signals = [3.0, 2.0, 1.0])
        bg = _bg_mask_avoiding(sphere_px, Npe, Nfe; pad = 6)
        single = image_snr_report(img_a, img_b, sphere_px, bg; roi_radius = 3)
        pooled = pooled_image_snr_report([img_a, img_a, img_a],
                                          [img_b, img_b, img_b],
                                          sphere_px, bg; roi_radius = 3)
        @test pooled.diff_roi_std ≈ single.diff_roi_std rtol = 5e-3
        # Per-sphere signal is the mean of per-block means → identical input
        # blocks ⇒ pooled sphere_mean_a equals single sphere_mean_a (no
        # Bessel quirk for means).
        @test pooled.sphere_mean_a ≈ single.sphere_mean_a
        @test pooled.sphere_means  ≈ single.sphere_means

        # Two distinct blocks: pooled sphere_mean is the average of per-block
        # means. Use roi_radius=0 (single centre pixel) so the per-block mean
        # is exactly the planted-signal magnitude.
        img_a2, _, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                     σ = 0.1, signals = [6.0, 4.0, 2.0])
        img_b2, _, _ = _synth_image(rng; Npe = Npe, Nfe = Nfe,
                                     σ = 0.1, signals = [6.0, 4.0, 2.0])
        pooled2 = pooled_image_snr_report([img_a, img_a2],
                                           [img_b, img_b2],
                                           sphere_px, bg; roi_radius = 0)
        p = sphere_px[1]
        @test pooled2.sphere_mean_a[1] ≈
              (Float64(img_a[p[1], p[2]]) + Float64(img_a2[p[1], p[2]])) / 2
    end

end
