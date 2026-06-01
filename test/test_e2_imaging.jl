# Image-domain tests for the E2 reconstruction pipeline.
#
# These tests assert that the magnitude image produced by `_e2_simulate_step`
# is aligned with the phantom in physical space — i.e. an off-centre sphere
# lands at the corresponding off-centre pixel.
#
# History: these tests originally failed on main due to three independent
# bugs that they were designed to surface (see FIX_SIM_PLAN.md §1):
#  1. Missing fftshift around the IFFT (centre sphere mapped to (1,1)).
#  2. Old ROI indexing assumed unshifted recon (sphere ROIs sampled the
#     wrong pixel after the FFT fix).
#  3. Gradient amplitudes 2π× too small (effective image FOV was 2π× too
#     large, collapsing every sphere to ~centre regardless of position).
# All three are now fixed. T4 (amplitude tracking vs closed-form IR) is
# marked @test_skip pending a more accurate forward model.

using Test
using Random
using FFTW
using MRISystemPhantom
using KomaMRI

# Reach into the package for the internal recon helper. We intentionally
# bypass `e2_step!` here to test the recon in isolation, without the time-
# budget / reward bookkeeping.
const _sim_step = MRISystemPhantom._e2_simulate_step

"Build a single-sphere E2Env with no noise and no pose augmentation."
function _single_sphere_env(; centre::NTuple{3,Float64} = (0.0, 0.0, 0.0),
                              T1::Float64 = 1.0, T2::Float64 = 0.5,
                              FOV::Float64 = 0.2, Nfe::Int = 64, Npe::Int = 16,
                              voxel_size_mm::Float64 = 3.0)
    env = E2Env(;
        cfg_field             = :T3,
        voxel_size_mm         = voxel_size_mm,
        FOV                   = FOV,
        Nfe                   = Nfe,
        Npe                   = Npe,
        subset_size           = 1,
        max_blocks            = 20,
        time_budget_s         = 1.0e6,
        noise_sigma_abs       = 0.0,
        T1_sigma_rel          = 0.0,
        translation_sigma_mm  = 0.0,
        rotation_sigma_rad    = 0.0,
        rng_seed              = 0,
    )

    # Build a custom phantom with one sphere at the chosen location.
    d = SphereDescriptor(centre, CONTRAST_RADIUS_M, 1.0,
                          T1, T2, T2, 0.0, :probe)
    phantom = build_sphere(d, voxel_size_mm * 1e-3)
    @assert length(phantom.x) > 0 "probe sphere voxelisation produced zero spins"

    # Override env state to reflect the custom phantom.
    env.phantom  = phantom
    env.T1_true  = [T1]

    # Expected pixel under centred indexing (DC at array centre):
    mc_pe = Npe ÷ 2 + 1
    mc_fe = Nfe ÷ 2 + 1
    ife = mod(round(Int, centre[1] * Nfe / FOV) + Nfe ÷ 2, Nfe) + 1
    ipe = mod(round(Int, centre[2] * Npe / FOV) + Npe ÷ 2, Npe) + 1
    env.sphere_px = [(ipe, ife)]
    env, (ipe, ife), (mc_pe, mc_fe)
end

@testset "E2 imaging — recon alignment" begin

    @testset "T1: centre sphere peaks at image centre" begin
        # Single sphere at the FOV centre, long TI so the magnetisation is
        # near steady state. With a correct recon the magnitude image must
        # peak at the centre pixel (Npe÷2+1, Nfe÷2+1).
        env, _, (mc_pe, mc_fe) = _single_sphere_env(; centre = (0.0, 0.0, 0.0),
                                                     T1 = 1.0, T2 = 0.5)
        image_mag, _ = _sim_step(env, 0.693, 0.02, 3.0, 90.0)

        # ±1 pixel tolerance allows for the half-sample offset introduced by
        # centre-aligned ADC sampling vs centre-aligned image-pixel grid.
        idx = argmax(image_mag)
        @test abs(idx[1] - mc_pe) <= 1
        @test abs(idx[2] - mc_fe) <= 1
    end

    @testset "T2: off-centre sphere peaks at off-centre pixel (catches FFT-shift)" begin
        # Sphere shifted by FOV/4 along x AND y. Under a correct recon the
        # peak should land at the centred-indexing pixel for that offset.
        # Under the bare-ifft buggy recon the peak is half-FOV-wrapped, so
        # it lands at the opposite quadrant — the assertion below fails.
        FOV = 0.2
        Nfe, Npe = 64, 16
        env, (ipe, ife), _ =
            _single_sphere_env(; centre = (FOV/4, FOV/4, 0.0),
                                 T1 = 1.0, T2 = 0.5,
                                 FOV = FOV, Nfe = Nfe, Npe = Npe)
        image_mag, _ = _sim_step(env, 0.693, 0.02, 3.0, 90.0)

        idx = argmax(image_mag)
        @test abs(idx[1] - ipe) <= 1
        @test abs(idx[2] - ife) <= 1
    end

    @testset "T3: two symmetric spheres produce a symmetric image" begin
        # Two equal-amplitude spheres at (+x0, 0) and (-x0, 0). Image must be
        # mirror-symmetric across the Nfe (column) axis. Bare ifft alone
        # preserves this symmetry, but combined with the row/col origin shift
        # it can land the pair on aliased non-mirror pixels. We assert
        # equality between the two predicted pixel columns.
        FOV = 0.2
        Nfe, Npe = 64, 16
        T1, T2 = 1.0, 0.5
        x0 = FOV / 4

        d1 = SphereDescriptor((+x0, 0.0, 0.0), CONTRAST_RADIUS_M, 1.0,
                               T1, T2, T2, 0.0, :probe_pos)
        d2 = SphereDescriptor((-x0, 0.0, 0.0), CONTRAST_RADIUS_M, 1.0,
                               T1, T2, T2, 0.0, :probe_neg)
        p1 = build_sphere(d1, 3.0e-3)
        p2 = build_sphere(d2, 3.0e-3)
        phantom = p1 + p2

        env, _, (mc_pe, mc_fe) = _single_sphere_env(; FOV = FOV,
                                                     Nfe = Nfe, Npe = Npe)
        env.phantom = phantom

        image_mag, _ = _sim_step(env, 0.693, 0.02, 3.0, 90.0)

        j_left  = mc_fe - Nfe ÷ 4
        j_right = mc_fe + Nfe ÷ 4
        # Compare the two predicted-sphere columns. Tolerance of 5 % is
        # generous; the buggy recon scrambles which column carries which
        # sphere's signal, so we expect a larger discrepancy.
        col_l = image_mag[:, j_left]
        col_r = image_mag[:, j_right]
        @test isapprox(maximum(col_l), maximum(col_r); rtol = 0.05)
        # And both columns must carry significant signal (the spheres exist).
        bg = maximum(image_mag) * 0.02
        @test maximum(col_l) > bg
        @test maximum(col_r) > bg
    end

    @testset "T4: IR signal vs steady-state analytic, ±10 %" begin
        # Per-TI image-peak magnitude tracks the steady-state IR signal
        #     Mz_pre_inv  = 1 − exp(−(TR − TI)/T1)
        #     Mz_at_excite = (1 − exp(−TI/T1)) − Mz_pre_inv · exp(−TI/T1)
        #     |S| ∝ |Mz_at_excite| · exp(−TE/T2)
        # i.e. the formula `|1 − 2·exp(−TI/T1)|` is only correct in the
        # TR → ∞ limit; real IR-SE runs in steady state where Mz before
        # each inversion isn't fully recovered.
        #
        # We use TR = 4 s here so total sim time = TR · Npe = 64 s stays
        # inside KomaMRI's numerical-drift safe zone (see
        # KOMA_BUG_REPRO.md — the simulator's per-shot signal corrupts
        # past ~60–100 s of simulated time).
        # Single centre sphere, long TR/T2 so finite-Npe transient and T2
        # decay are negligible. Five TIs scan the IR curve. The ratio of
        # centre-pixel magnitudes across TIs should match the analytic IR
        # signal ratio. Under the buggy recon, image[mc, mc] samples the
        # FOV-corner (≈ zero for a centred sphere) → ratios are noise.
        FOV = 0.2
        Nfe, Npe = 64, 16
        T1, T2 = 1.0, 1.0
        env, _, (mc_pe, mc_fe) = _single_sphere_env(; centre = (0.0, 0.0, 0.0),
                                                     T1 = T1, T2 = T2,
                                                     FOV = FOV,
                                                     Nfe = Nfe, Npe = Npe)

        TIs = [0.1, 0.3, 0.693, 1.5, 3.0]
        TR  = 4.0
        TE  = 0.02

        mags = Float64[]
        for ti in TIs
            image_mag, _ = _sim_step(env, ti, TE, TR, 90.0)
            win = image_mag[mc_pe-1:mc_pe+1, mc_fe-1:mc_fe+1]
            push!(mags, Float64(maximum(win)))
        end

        # Steady-state IR magnitude (no T2 factor needed — same TE across
        # all TIs cancels in ratios).
        function ss_ir(ti, T1, TR)
            Mz_pre_inv = 1.0 - exp(-(TR - ti) / T1)
            Mz_at_exc  = (1.0 - exp(-ti / T1)) - Mz_pre_inv * exp(-ti / T1)
            return abs(Mz_at_exc)
        end
        analytic = [ss_ir(ti, T1, TR) for ti in TIs]

        # Normalise both observed and analytic by their final entry to
        # remove the unknown global amplitude factor (sphere volume × PD ×
        # KomaMRI signal scale × recon kernel normalisation).
        obs_ratio = mags ./ mags[end]
        ana_ratio = analytic ./ analytic[end]

        for k in eachindex(TIs)
            @test isapprox(obs_ratio[k], ana_ratio[k]; rtol = 0.10, atol = 0.05)
        end
    end

    @testset "T5: pose translation moves the image peak" begin
        # A centred phantom that we then translate by (+tx, +ty) should
        # produce a peak at a predictably shifted pixel. The ROI mapping
        # in `_e2_build_episode_phantom` uses the same transform, so this
        # also tests that ROI vs image stay in lockstep.
        FOV = 0.2
        Nfe, Npe = 64, 16
        env, _, (mc_pe, mc_fe) = _single_sphere_env(; centre = (0.0, 0.0, 0.0),
                                                     FOV = FOV,
                                                     Nfe = Nfe, Npe = Npe)
        # Apply a pure translation in-plane: FOV/8 in x, -FOV/8 in y.
        tx, ty = FOV / 8, -FOV / 8
        env.phantom = apply_transform!(env.phantom, (0.0, 0.0, 0.0),
                                        (tx, ty, 0.0))

        image_mag, _ = _sim_step(env, 0.693, 0.02, 3.0, 90.0)
        idx = argmax(image_mag)

        expected_fe = mod(round(Int, tx * Nfe / FOV) + Nfe ÷ 2, Nfe) + 1
        expected_pe = mod(round(Int, ty * Npe / FOV) + Npe ÷ 2, Npe) + 1
        @test abs(idx[1] - expected_pe) <= 1
        @test abs(idx[2] - expected_fe) <= 1
    end

    @testset "T8: T1-plate slice — recon matches phantom geometry" begin
        # Build the real 14-sphere T1 plate (no augmentation), simulate one
        # block at long TI/TR so all spheres are at positive saturation,
        # then compare the magnitude image to a phantom-derived ground-truth
        # occupancy image. Under a correct recon these should be strongly
        # positively correlated (bright pixels coincide with sphere centres).
        #
        # Diagnostic 2026-05-11: this test fails decisively on main even
        # after applying every plausible FFT-shift combination — the
        # reconstructed image is a featureless blob at the FOV centre with
        # no resolved sphere structure. Inspection of |ksp| per row shows an
        # asymmetric distribution (smooth rise to row 16, plateau on rows
        # 17–32) inconsistent with a normal Fourier-encoded k-space. There
        # is a second imaging-pipeline bug beyond fftshift — likely in the
        # PE gradient bookkeeping or cross-shot magnetisation state — that
        # this test will continue to catch until fixed. See FIX_SIM_PLAN.md
        # §1.6 (to be added) for the follow-up investigation.
        FOV = 0.2
        Nfe, Npe = 64, 16
        cfg = PhantomConfig(field = :T3, voxel_size_mm = 2.0,
                             include_plates = [:T1])
        phantom = build_phantom(cfg)
        @assert length(phantom.x) > 100 "T1 plate phantom suspiciously sparse"

        env = E2Env(;
            cfg_field             = :T3,
            voxel_size_mm         = 2.0,
            FOV                   = FOV,
            Nfe                   = Nfe,
            Npe                   = Npe,
            subset_size           = 1,
            max_blocks            = 5,
            time_budget_s         = 1.0e6,
            noise_sigma_abs       = 0.0,
            T1_sigma_rel          = 0.0,
            translation_sigma_mm  = 0.0,
            rotation_sigma_rad    = 0.0,
            rng_seed              = 0,
        )
        env.phantom = phantom

        # Long TI + long TR + 90° excitation → every sphere produces near-PD
        # transverse magnetisation, regardless of its T1.
        image_mag, _ = _sim_step(env, 3.0, 0.02, 8.0, 90.0)

        # ------- Ground-truth occupancy from the phantom spins -----------
        occ = phantom_occupancy(phantom, Npe, Nfe, FOV)

        # Sanity: occupancy is non-trivial.
        @test sum(occ .> 0) >= 14

        # ------- Pearson correlation, flattened ---------------------------
        a = vec(Float64.(image_mag))
        b = vec(occ)
        a .-= sum(a) / length(a)
        b .-= sum(b) / length(b)
        r = sum(a .* b) / (sqrt(sum(a .^ 2) * sum(b .^ 2)) + eps())

        # Under correct recon we typically see r > 0.6 on this geometry.
        # Under the buggy recon r collapses to ~0 (or weakly negative).
        @test r > 0.5

        # ------- Top-K overlap -------------------------------------------
        # Of the 14 brightest image pixels (one per sphere, roughly), most
        # should sit on an occupied pixel.
        K = 14
        flat_img = vec(image_mag)
        topk = partialsortperm(flat_img, 1:K; rev = true)
        on_phantom = count(i -> occ[i] > 0, topk)
        @test on_phantom >= 8   # ≥ 57 % of the top-K hit a real sphere pixel
    end

    @testset "T9: end-to-end multi-block fit converges (noiseless)" begin
        # Centre sphere, run a small adaptive schedule via e2_step!, and
        # assert the fitter's running estimate ends within 10 % of truth.
        # This is the integration test — exercises seq → simulate →
        # k-space stacking → recon → ROI sample → α-correct → fitter.
        # Under the FFT-shift bug, the ROI samples the wrong pixel (DC
        # corner ≈ 0 for a centred sphere), so the fitter is fed
        # near-zero magnitudes at every TI and the recovered T1 is
        # garbage.
        env, _, _ = _single_sphere_env(; centre = (0.0, 0.0, 0.0),
                                          T1 = 1.0, T2 = 0.5)
        e2_reset!(env; rng_seed = 0)
        # Override the reset's auto-built phantom & ROI with our single-sphere
        # configuration (reset rebuilds from the T1-plate pool).
        d = SphereDescriptor((0.0, 0.0, 0.0), CONTRAST_RADIUS_M, 1.0,
                              1.0, 0.5, 0.5, 0.0, :probe)
        env.phantom = build_sphere(d, env.voxel_size_mm * 1e-3)
        env.T1_true = [1.0]
        mc_pe = env.Npe ÷ 2 + 1
        mc_fe = env.Nfe ÷ 2 + 1
        env.sphere_px = [(mc_pe, mc_fe)]

        TIs = [0.1, 0.3, 0.7, 1.5, 3.0]
        for ti in TIs
            _ = e2_step!(env, Float64[ti, 0.02, 5.0, 90.0, 0.0])
        end

        # Allow 15 % rtol — the ROI samples the centred pixel which may
        # carry a sub-peak signal due to half-sample offset and finite voxel
        # size. Noiseless single-sphere fit should still recover T1 well.
        @test isapprox(env.T1_est[1], 1.0; rtol = 0.15)
    end

    @testset "T9b: end-to-end fit at non-90° α (noiseless)" begin
        # Same integration path as T9 but at α=40°. Exercises the sin(α)
        # magnitude correction + cos(α) recurrence through the real e2_step!
        # pipeline (ALPHA_DOF.md tests). Catches any recon/normalisation
        # regression that only bites away from 90°.
        env, _, _ = _single_sphere_env(; centre = (0.0, 0.0, 0.0),
                                          T1 = 1.0, T2 = 0.5)
        e2_reset!(env; rng_seed = 0)
        d = SphereDescriptor((0.0, 0.0, 0.0), CONTRAST_RADIUS_M, 1.0,
                              1.0, 0.5, 0.5, 0.0, :probe)
        env.phantom = build_sphere(d, env.voxel_size_mm * 1e-3)
        env.T1_true = [1.0]
        env.sphere_px = [(env.Npe ÷ 2 + 1, env.Nfe ÷ 2 + 1)]

        for ti in (0.1, 0.3, 0.7, 1.5, 3.0)
            _ = e2_step!(env, Float64[ti, 0.02, 5.0, 40.0, 0.0])   # α = 40°
        end
        @test isapprox(env.T1_est[1], 1.0; rtol = 0.15)
    end

end
