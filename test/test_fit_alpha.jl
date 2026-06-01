# Thorough T1-fitter tests at non-90° excitation flip angle (see ALPHA_DOF.md).
#
# Run A frees α as a learned action. The fitter is already α-aware:
#  - cos(α_exc) enters the Mz recurrence in transient_mz_at_excite_npe,
#  - the env divides the measured magnitude by sin(α_exc) before fitting.
# These tests pin both halves: across a sweep of α the corrected magnitude is
# fit back to the true T1, and a single fit fed a *mix* of α values still
# recovers T1 (per-sample α_excs honoured, not a single global α).

using Test
using Random
using MRISystemPhantom

@testset "fitter at non-90° α" begin

    # Mimic the real E2 pipeline for one shot at flip angle α_exc:
    #   measured magnitude  = sin(α)·|Mz_at_excite|   (what the scanner produces)
    #   corrected magnitude = measured / sin(α)       (what the env feeds the fit)
    # so the fitter sees |Mz_at_excite| and its forward model |A·Mz| fits at A=1.
    corrected_mag(T1, TI, TR, α; Npe) =
        abs(transient_mz_at_excite_npe(T1, TI, TR, π, α; Npe = Npe))

    # Sweep α ∈ {15..90°} × {short,mid,long} T1: the sin(α)-corrected magnitude
    # fits back to the true T1 within 2% (noiseless). Pins cos(α) recurrence +
    # sin(α) correction are mutually consistent.
    @testset "noiseless recovery across α × T1" begin
        Npe = 8
        rng = MersenneTwister(7)
        n   = 12
        TIs = sort!(exp.(log(0.01) .+ (log(3.0) - log(0.01)) .* rand(rng, n)))
        TRs = TIs .+ 0.5 .+ 2.0 .* rand(rng, n)
        for T1 in (0.046, 0.367, 1.398)            # short / mid / long fleet T1s
            for αdeg in (15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
                α   = deg2rad(αdeg)
                αes = fill(α, n)
                αs  = fill(π, n)                   # 180° inversion prep
                mags = [corrected_mag(T1, TIs[i], TRs[i], α; Npe = Npe)
                        for i in 1:n]
                fit = fit_t1_generalized_ir(TIs, αs, mags;
                            TRs = TRs, α_excs = αes, Npe = Npe,
                            T1_range = (0.01, 3.0), n_grid = 600)
                @test isapprox(fit.T1, T1; rtol = 0.02)
            end
        end
    end

    @testset "mixed-α schedule recovers T1" begin
        # Per-sample α_excs must be honoured: alternate 40° and 90° within one fit.
        Npe = 8
        rng = MersenneTwister(11)
        n   = 12
        TIs = sort!(exp.(log(0.01) .+ (log(3.0) - log(0.01)) .* rand(rng, n)))
        TRs = TIs .+ 0.5 .+ 2.0 .* rand(rng, n)
        αes = [iseven(i) ? deg2rad(40.0) : deg2rad(90.0) for i in 1:n]
        αs  = fill(π, n)
        for T1 in (0.131, 0.509, 0.998)
            mags = [corrected_mag(T1, TIs[i], TRs[i], αes[i]; Npe = Npe)
                    for i in 1:n]
            fit = fit_t1_generalized_ir(TIs, αs, mags;
                        TRs = TRs, α_excs = αes, Npe = Npe,
                        T1_range = (0.01, 3.0), n_grid = 600)
            @test isapprox(fit.T1, T1; rtol = 0.05)
        end
    end

    @testset "small-α fit stays finite under noise" begin
        # α = 5° → sin(α) ≈ 0.087: the magnitude correction amplifies noise.
        # The fit must still converge (no NaN/Inf) and stay in range.
        Npe = 8
        rng = MersenneTwister(3)
        n   = 12
        T1  = 0.509
        α   = deg2rad(5.0)
        TIs = sort!(exp.(log(0.01) .+ (log(3.0) - log(0.01)) .* rand(rng, n)))
        TRs = TIs .+ 0.5 .+ 2.0 .* rand(rng, n)
        αes = fill(α, n); αs = fill(π, n)
        σ   = 0.01
        mags = [corrected_mag(T1, TIs[i], TRs[i], α; Npe = Npe) +
                σ / max(abs(sin(α)), 1e-3) * randn(rng) for i in 1:n]
        fit = fit_t1_generalized_ir(TIs, αs, mags;
                    TRs = TRs, α_excs = αes, Npe = Npe,
                    T1_range = (0.01, 3.0), n_grid = 600,
                    abs_noise_sigma = σ)
        @test isfinite(fit.T1)
        @test 0.01 <= fit.T1 <= 3.0
    end

    @testset "α=90° reproduces textbook-IR recovery (regression)" begin
        # The α-aware path must not redefine the α=90° baseline.
        Npe = 8
        T1  = 1.0
        TIs = [0.05, 0.3, 1.0, 2.0]
        TRs = [0.6,  1.0, 1.5, 2.5]
        αes = fill(π/2, 4); αs = fill(π, 4)
        mags = [corrected_mag(T1, TIs[i], TRs[i], π/2; Npe = Npe) for i in 1:4]
        fit = fit_t1_generalized_ir(TIs, αs, mags;
                    TRs = TRs, α_excs = αes, Npe = Npe,
                    T1_range = (0.05, 5.0), n_grid = 600)
        @test isapprox(fit.T1, T1; rtol = 0.03)
    end

end
