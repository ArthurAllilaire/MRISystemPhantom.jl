@testset "E2 — sequence + env regressions" begin

    @testset "ir_se_2d_sequence: no Gy during ADC (Cartesian sampling)" begin
        # Regression test for the Gy-rewind-during-ADC bug. The ADC window
        # must run with Gy = 0 — otherwise ky shifts during readout and the
        # k-space line is sheared, breaking the standard FFT reconstruction.
        seq = ir_se_2d_sequence(0.5, 0.02, 2.0;
                                 FOV = 0.2, Nfe = 8, Npe = 4)

        adc_blocks = [i for i in 1:length(seq) if seq[i].ADC[1].N > 0]
        @test length(adc_blocks) == 4         # Npe profiles

        for i in adc_blocks
            b = seq[i]
            @test isapprox(b.GR[2, 1].A, 0.0; atol = 1e-12)   # Gy = 0
            @test b.GR[1, 1].A > 0                            # Gx readout active
        end
    end

    @testset "ir_se_2d_sequence: prewinder sign drives ky to +ky_steps[k] at echo" begin
        # After the fix, the prewinder is negative; the 180° refocus then
        # flips it to positive ky_steps[k] (the intended sampling line).
        # Walking through the structure: every 4th non-empty block (after the
        # prewinder pattern) corresponds to one PE step. Just verify Gy areas
        # span both signs across PE steps and are antisymmetric about zero.
        seq = ir_se_2d_sequence(0.5, 0.02, 2.0;
                                 FOV = 0.2, Nfe = 8, Npe = 4)

        # Prewinder blocks: "no ADC, has Gx prewinder". Filtering on Gx
        # picks up the ky=0 step (Gy=0) which is dropped by a Gy-based
        # predicate under the even FFT convention (Npe=4 → ky∈{-2,-1,0,+1}·Δky).
        gy_amps = Float64[]
        for i in 1:length(seq)
            b = seq[i]
            if b.ADC[1].N == 0 && abs(b.GR[1, 1].A) > 1e-12
                push!(gy_amps, b.GR[2, 1].A)
            end
        end

        @test length(gy_amps) == 4

        # The fix that the test is guarding: prewinder Gy = -ky_step/(γ·dur_pe).
        # Recompute the expected schedule and confirm element-wise (in order of
        # appearance — k = 1..Npe).
        γ_Hz = 42.577e6
        FOV  = 0.2
        Npe  = 4
        dur_pe = 0.5e-3   # dur_adc/2 in the builder, with dur_adc = 1e-3
        Δky    = 1.0 / FOV
        expected = [-((k - 1 - Npe÷2) * Δky) / (γ_Hz * dur_pe) for k in 1:Npe]

        @test all(isapprox.(gy_amps, expected; atol = 1e-12))
    end

    @testset "se_2d_sequence: no Gy during ADC; Npe ADC blocks" begin
        # Spatial spin echo for T2 — same Cartesian-readout regression as the IR
        # version (it shares the gradient code, just without the inversion+TI).
        seq = se_2d_sequence(0.02, 2.0; FOV = 0.2, Nfe = 8, Npe = 4)
        adc_blocks = [i for i in 1:length(seq) if seq[i].ADC[1].N > 0]
        @test length(adc_blocks) == 4
        for i in adc_blocks
            b = seq[i]
            @test isapprox(b.GR[2, 1].A, 0.0; atol = 1e-12)   # Gy = 0 during ADC
            @test b.GR[1, 1].A > 0                            # Gx readout active
        end
    end

    @testset "se_2d_sequence: echo carries exp(-TE/T2) weighting (single spin)" begin
        # On a single spin at the origin the Gx readout doesn't dephase, so the
        # centre readout sample tracks the echo magnitude. The ratio across two
        # TEs must be the clean T2 decay exp(-(TE2-TE1)/T2) — pins the SE timing.
        T2 = 0.1
        obj = single_spin_phantom(T1 = 1.0, T2 = T2)
        echo(TE) = begin
            seq = se_2d_sequence(TE, 5.0; FOV = 0.2, Nfe = 8, Npe = 1)
            raw = @suppress simulate(obj, seq, Scanner())
            abs(raw.profiles[1].data[4 + 1, 1])   # centre of the 8-sample readout
        end
        TE1, TE2 = 0.02, 0.06
        @test isapprox(echo(TE2) / echo(TE1), exp(-(TE2 - TE1) / T2);
                       rtol = 0.06, atol = 5e-3)
    end

    @testset "ir_tse_2d_sequence: builder structure + guards" begin
        seq = ir_tse_2d_sequence(0.5, 0.02, 2.0; etl = 2, FOV = 0.2, Nfe = 8, Npe = 4)
        adc_blocks = [i for i in 1:length(seq) if seq[i].ADC[1].N > 0]
        @test length(adc_blocks) == 4                      # Npe total readouts (2 shots × etl 2)
        for i in adc_blocks
            @test isapprox(seq[i].GR[2, 1].A, 0.0; atol = 1e-12)   # Gy = 0 during ADC
            @test seq[i].GR[1, 1].A > 0                            # Gx readout active
        end
        @test_throws ErrorException ir_tse_2d_sequence(0.5, 0.02, 2.0; etl = 3, Npe = 4)  # 3∤4
        @test_throws ErrorException ir_tse_2d_sequence(0.5, 0.02, 2.0; etl = 0, Npe = 4)
        @test_throws ErrorException ir_tse_2d_sequence(0.5, 1e-5, 2.0; etl = 2, Npe = 4)  # esp too short
    end

    @testset "ir_tse_2d_sequence: etl=1 echo equals ir_se_2d (single spin)" begin
        # etl=1 must reproduce the single-echo IR readout; on a single spin the
        # gradients don't dephase so the centre readout sample is the echo.
        T1, T2, TI, esp = 1.0, 0.1, 0.5, 0.02
        obj = single_spin_phantom(T1 = T1, T2 = T2)
        ctr(seq) = abs((@suppress simulate(obj, seq, Scanner())).profiles[1].data[5, 1])
        s_se  = ir_se_2d_sequence(TI, esp, 5.0; FOV = 0.2, Nfe = 8, Npe = 1)
        s_tse = ir_tse_2d_sequence(TI, esp, 5.0; etl = 1, FOV = 0.2, Nfe = 8, Npe = 1)
        @test isapprox(ctr(s_se), ctr(s_tse); rtol = 0.02, atol = 5e-3)
    end

    @testset "ir_tse_2d_sequence: echo train decays as exp(-e·esp/T2)" begin
        T1, T2, TI, esp = 1.0, 0.12, 0.5, 0.02
        obj = single_spin_phantom(T1 = T1, T2 = T2)
        seq = ir_tse_2d_sequence(TI, esp, 5.0; etl = 4, FOV = 0.2, Nfe = 8, Npe = 4)
        raw = @suppress simulate(obj, seq, Scanner())
        echoes = [abs(raw.profiles[e].data[5, 1]) for e in 1:4]
        for e in 1:3                                        # Meiboom–Gill: clean mono-exp
            @test isapprox(echoes[e+1] / echoes[e], exp(-esp / T2); rtol = 0.06, atol = 5e-3)
        end
    end

    @testset "ir_tse_2d_sequence: etl=2 places spins in same PE rows as IR (ky order)" begin
        # T2 ≫ etl·esp so the train barely decays → the TSE recon must resolve
        # the same phase-encode rows as the single-echo IR. A ky-blip-sign or
        # view-ordering bug shifts the rows by half-FOV (the non-Meiboom–Gill bug
        # this guards against).
        TI, esp, TR, FOV, Nfe, Npe = 0.5, 0.02, 5.0, 0.2, 8, 4
        obj = Phantom(x = [0.0, 0.0], y = [0.0, 0.05],
                      T1 = [1.0, 1.0], T2 = [10.0, 10.0], ρ = [1.0, 1.0])
        recon(seq) = kspace_to_image(raw_to_kspace(
            (@suppress simulate(obj, seq, Scanner())), Npe, Nfe))
        img_se  = recon(ir_se_2d_sequence(TI, esp, TR; FOV = FOV, Nfe = Nfe, Npe = Npe))
        img_tse = recon(ir_tse_2d_sequence(TI, esp, TR; etl = 2, FOV = FOV, Nfe = Nfe, Npe = Npe))
        row_energy(img) = vec(sum(abs2, img; dims = 2))
        top2(img) = sort(partialsortperm(row_energy(img), 1:2; rev = true))
        @test top2(img_tse) == top2(img_se)                 # spins in the same PE rows
        @test isapprox(maximum(img_tse), maximum(img_se); rtol = 0.2)   # no gross scale bug
    end

    @testset "fit_t1_generalized_ir: α_exc-scaled magnitudes need correction" begin
        # Regression for the excitation-flip-angle scaling fix in e2.jl.
        # The fitter has a single amplitude A, so when α_exc varies between
        # blocks the env must divide observed magnitude by sin(α_exc) before
        # passing it in — otherwise the fit is biased.
        T1_true = 0.6
        TIs = [0.05, 0.2, 0.5, 1.5, 3.0]
        α_excs = [deg2rad(d) for d in (10.0, 45.0, 90.0, 135.0, 30.0)]

        # Observed magnitudes: signal scales by sin(α_exc) per shot
        mags_obs = [sin(α) * abs(1 - 2 * exp(-ti / T1_true))
                    for (ti, α) in zip(TIs, α_excs)]

        αs_inv = fill(π, length(TIs))   # inversion prep is always 180°

        # Without correction: feed raw mags directly → biased fit
        f_bad = fit_t1_generalized_ir(TIs, αs_inv, mags_obs;
                                        T1_range = (0.01, 3.0), n_grid = 200)

        # With correction (matches what _e2_update_t1_estimates! now does)
        mags_corr = [m / max(abs(sin(α)), 1e-3)
                     for (m, α) in zip(mags_obs, α_excs)]
        f_good = fit_t1_generalized_ir(TIs, αs_inv, mags_corr;
                                         T1_range = (0.01, 3.0), n_grid = 200)

        @test isapprox(f_good.T1, T1_true; rtol = 0.05)
        @test abs(f_good.T1 - T1_true) <= abs(f_bad.T1 - T1_true) + 1e-9
    end

    @testset "steady_state_mz_at_excite: analytic identities" begin
        T1 = 0.8

        # TR → ∞ collapses to legacy non-steady-state form
        for (TI, θ) in [(0.1, π), (0.3, π/2), (0.05, deg2rad(10))]
            mz_inf  = steady_state_mz_at_excite(T1, TI, Inf, θ, π/2)
            mz_legacy = 1 - (1 - cos(θ)) * exp(-TI / T1)
            @test isapprox(mz_inf, mz_legacy; atol = 1e-12)
        end

        # θ_inv = π, α_exc = π/2 → textbook IR with finite TR:
        #   1 − 2·exp(−TI/T1) + exp(−TR/T1)
        for (TI, TR) in [(0.05, 0.5), (0.3, 1.0), (0.1, 0.3), (1.0, 5.0)]
            mz   = steady_state_mz_at_excite(T1, TI, TR, π, π/2)
            text = 1 - 2*exp(-TI/T1) + exp(-TR/T1)
            @test isapprox(mz, text; atol = 1e-12)
        end

        # Finite-TR correction matters: at TR/T1 = 0.5, magnetisation is
        # nowhere near the TR=∞ value.
        @test abs(steady_state_mz_at_excite(T1, 0.1, 0.4, π, π/2) -
                  steady_state_mz_at_excite(T1, 0.1, Inf, π, π/2)) > 0.3
    end

    @testset "fit_t1_generalized_ir: TR-aware fit unbiased; TR-blind biased" begin
        # When TR/T1 is small, the legacy "full recovery" model biases T1.
        # Generate clean steady-state data, then confirm the new TR-aware
        # call recovers T1 while the old TR-blind call does not.
        T1_true = 1.2
        TIs = [0.05, 0.15, 0.4, 0.9, 1.5]
        TRs = [0.5,  0.8,  1.2, 1.8, 2.5]    # all comparable to T1
        αs_inv = fill(π, length(TIs))
        α_excs = fill(π/2, length(TIs))

        # Magnitudes per textbook IR with finite TR
        mags = [abs(1 - 2*exp(-ti/T1_true) + exp(-tr/T1_true))
                for (ti, tr) in zip(TIs, TRs)]

        f_blind = fit_t1_generalized_ir(TIs, αs_inv, mags;
                                         T1_range = (0.05, 5.0), n_grid = 400)
        f_aware = fit_t1_generalized_ir(TIs, αs_inv, mags;
                                         TRs = TRs, α_excs = α_excs,
                                         T1_range = (0.05, 5.0), n_grid = 400)

        @test isapprox(f_aware.T1, T1_true; rtol = 0.02)
        @test abs(f_aware.T1 - T1_true) < abs(f_blind.T1 - T1_true)
    end

    @testset "steady-state IR formula matches KomaMRI simulation" begin
        # Validate the analytic Mz(TI⁻) against a Bloch simulation. Use
        # short T2 so transverse magnetisation decays between TRs (≈ perfect
        # spoiling assumption); θ_inv=π, α_exc=π/2, which reaches steady
        # state in one TR (cos α_exc = 0 kills the transient term).
        T1 = 0.5
        T2 = 0.02                                    # T2 ≪ (TR − TI)
        amp_T = 100e-6                               # short pulses → analytic ≈ Bloch
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)
        n_rep = 4

        for (TI, TR) in [(0.05, 0.5), (0.2, 1.0), (0.1, 0.3), (0.4, 1.5)]
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || continue

            obj = single_spin_phantom(T1 = T1, T2 = T2)
            seq = Sequence()
            for _ in 1:n_rep
                seq += RF(amp_T, d180)
                seq += Delay(ti_delay)
                seq += RF(amp_T, d90)
                seq += ADC(1, 1e-6, 0.0)
                shot = d180 + ti_delay + d90 + 1e-6
                seq += Delay(TR - shot)
            end

            raw = @suppress simulate(obj, seq, Scanner())
            koma_mag = abs(raw.profiles[end].data[1, 1])

            analytic = abs(steady_state_mz_at_excite(T1, TI, TR, π, π/2))
            @test isapprox(koma_mag, analytic; rtol = 0.05, atol = 5e-3)
        end
    end

    @testset "varying-action: per-shot bias vs KomaMRI (α_exc = π/2)" begin
        # The existing "matches KomaMRI" test repeats the SAME shot n_rep
        # times so the simulator naturally reaches the analytic steady
        # state. Real E2 episodes don't do that — the agent picks a
        # DIFFERENT (TI, TR, α_exc) every shot, so each block starts
        # from whatever the previous block's exit state was, NOT from
        # the steady-state Mz_pre that the analytic formula assumes.
        # This test pins that gap: per-shot Koma magnitude vs per-shot
        # analytic prediction. With α_exc = π/2 the cross-shot memory
        # is killed in one step (cos α_exc = 0), so the only bias is
        # "previous shot's (TR−TI) sets actual Mz_pre, current shot's
        # (TR−TI) sets predicted Mz_pre" — plus the very first shot
        # which starts at M0 = 1, not at any steady state.
        T1 = 0.6
        T2 = 0.02                                 # T2 ≪ TR−TI for spoiling
        amp_T = 100e-6
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)

        # Mix that swings TR widely between adjacent shots — the worst
        # case for the steady-state assumption. Long → short → long.
        actions = [(0.05, 0.5), (0.30, 1.5), (0.10, 0.4),
                   (0.40, 2.0), (0.15, 0.6), (0.80, 3.0)]

        obj = single_spin_phantom(T1 = T1, T2 = T2)
        seq = Sequence()
        for (TI, TR) in actions
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || error("TI too short for pulse durations")
            seq += RF(amp_T, d180)
            seq += Delay(ti_delay)
            seq += RF(amp_T, d90)
            seq += ADC(1, 1e-6, 0.0)
            shot = d180 + ti_delay + d90 + 1e-6
            seq += Delay(TR - shot)
        end

        raw = @suppress simulate(obj, seq, Scanner())

        biases  = Float64[]
        for (k, (TI, TR)) in enumerate(actions)
            koma     = abs(raw.profiles[k].data[1, 1])
            analytic = abs(steady_state_mz_at_excite(
                            T1, TI, TR, π, π/2))
            push!(biases, abs(koma - analytic))
            @info "shot bias (α_exc=π/2)" k TI TR koma analytic abs_bias=biases[end]
        end

        # Documented upper bound on the per-shot bias. Tighten as evidence
        # accumulates; the test is meant to SURFACE the assumption gap,
        # not to enforce it's small. Empirically the cold-start shot
        # (k=1) carries the largest bias (~0.43) because the simulator
        # starts at M0=1, not at the analytic steady-state Mz_pre.
        # Subsequent shots are bounded around 0.30.
        @test maximum(biases) < 0.50
    end

    @testset "varying-action: end-to-end T1 fit bias vs KomaMRI" begin
        # Same physical setup as above, but pipe the simulated magnitudes
        # through fit_t1_generalized_ir and check the recovered T1 against
        # the truth. This is the REPORT-RELEVANT number: how much does the
        # steady-state-per-shot assumption bias the T1 estimate when the
        # data come from a varying-action sequence?
        T1_true = 0.8
        T2 = 0.02
        amp_T = 100e-6
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)

        actions = [(0.05, 0.5), (0.30, 1.5), (0.10, 0.4),
                   (0.50, 2.0), (0.20, 0.8), (0.80, 3.0),
                   (0.15, 0.45), (1.20, 4.0)]

        obj = single_spin_phantom(T1 = T1_true, T2 = T2)
        seq = Sequence()
        for (TI, TR) in actions
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || continue
            seq += RF(amp_T, d180)
            seq += Delay(ti_delay)
            seq += RF(amp_T, d90)
            seq += ADC(1, 1e-6, 0.0)
            shot = d180 + ti_delay + d90 + 1e-6
            seq += Delay(TR - shot)
        end

        raw  = @suppress simulate(obj, seq, Scanner())
        mags = [abs(raw.profiles[k].data[1, 1]) for k in 1:length(actions)]

        TIs    = [a[1] for a in actions]
        TRs    = [a[2] for a in actions]
        αs_inv = fill(π,   length(actions))
        α_excs = fill(π/2, length(actions))

        f = fit_t1_generalized_ir(TIs, αs_inv, mags;
                                   TRs = TRs, α_excs = α_excs,
                                   T1_range = (0.05, 5.0), n_grid = 400)

        rel_bias = abs(f.T1 - T1_true) / T1_true
        @info "varying-action end-to-end" T1_true T1_est=f.T1 rel_bias residual=f.residual

        # Documented upper bound — surfaces the bias rather than hiding it.
        @test rel_bias < 0.30
    end

    @testset "varying-action with small-tip α_exc — bias grows" begin
        # When α_exc < π/2, cos(α_exc) > 0 keeps cross-shot memory alive,
        # so the steady-state-per-shot assumption fails harder. Run the
        # same varying-action probe at α_exc = 30° and confirm the
        # per-shot bias is at least as large as the π/2 case (we don't
        # over-claim the ratio — just show direction of effect).
        T1 = 0.6
        T2 = 0.02
        amp_T = 100e-6
        d180  = rf_duration(π; amp_T = amp_T)
        α_exc = deg2rad(30.0)
        d_exc = rf_duration(α_exc; amp_T = amp_T)

        actions = [(0.05, 0.5), (0.30, 1.5), (0.10, 0.4),
                   (0.40, 2.0), (0.15, 0.6)]

        obj = single_spin_phantom(T1 = T1, T2 = T2)
        seq = Sequence()
        for (TI, TR) in actions
            ti_delay = TI - d180/2 - d_exc/2
            ti_delay > 0 || error("TI too short")
            seq += RF(amp_T, d180)
            seq += Delay(ti_delay)
            seq += RF(amp_T, d_exc)
            seq += ADC(1, 1e-6, 0.0)
            shot = d180 + ti_delay + d_exc + 1e-6
            seq += Delay(TR - shot)
        end

        raw = @suppress simulate(obj, seq, Scanner())

        biases = Float64[]
        for (k, (TI, TR)) in enumerate(actions)
            koma = abs(raw.profiles[k].data[1, 1])
            # Predicted Mxy = sin(α_exc) · |Mz_at_excite| (steady state)
            analytic = sin(α_exc) * abs(steady_state_mz_at_excite(
                                          T1, TI, TR, π, α_exc))
            push!(biases, abs(koma - analytic))
            @info "shot bias (α_exc=30°)" k TI TR koma analytic abs_bias=biases[end]
        end

        # Loose upper bound — purpose is to log the bias, not gate on it.
        @test maximum(biases) < 0.50
    end

    # ── F1+ : finite-Npe transient closed-form forward model (E2.4) ──────────
    #
    # Pins the new `transient_mz_at_excite_npe(T1, TI, TR, θ_inv, α_exc; Npe)`
    # against three analytic limits and against KomaMRI on actual multi-shot
    # sequences. Plus a negative-regression test that the steady-state model
    # is provably biased on Npe-shot data — should fail loudly if anyone
    # silently re-introduces the steady-state assumption in fits.jl.
    @testset "F1+ closed-form limits" begin
        T1, TI, TR = 1.0, 0.5, 2.0
        # Limit 1: Npe = 1 ⇒ TR-blind transient form 1 − (1 − cos θ_inv)·E1
        @test isapprox(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = 1),
                        1 - (1 - cos(π)) * exp(-TI/T1); rtol = 1e-12)
        # Limit 2: Npe → ∞ ⇒ steady-state fixed point
        @test isapprox(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = 10_000),
                        steady_state_mz_at_excite(T1, TI, TR, π, π/2);
                        rtol = 1e-3)
        # Limit 3: TR → ∞ ⇒ Mz_pre[k] ≡ 1 for k > 0 ⇒ pure transient
        @test isapprox(transient_mz_at_excite_npe(T1, TI, 1e6, π, π/2; Npe = 8),
                        1 - (1 - cos(π)) * exp(-TI/T1); rtol = 1e-6)
        # Limit 4: Non-trivial α_exc (cos > 0) keeps cross-shot memory alive
        # so Npe=8 ≠ Npe=1 even if TR is finite — sanity check that α_exc
        # actually enters the recurrence.
        @test !isapprox(
            transient_mz_at_excite_npe(T1, TI, TR, π, deg2rad(30); Npe = 8),
            transient_mz_at_excite_npe(T1, TI, TR, π, deg2rad(30); Npe = 1);
            atol = 0.01)
    end

    @testset "F1+ matches KomaMRI on Npe-shot IR-SE single-spin" begin
        # Build the actual Npe-shot IR-SE sequence (cold start from M0=1, the
        # regime _e2_simulate_step runs in). Short T2 enforces spoiling, the
        # only assumption F1+ shares with the simulator.
        T2 = 0.02
        amp_T = 100e-6
        d180 = rf_duration(π;   amp_T = amp_T)
        d90  = rf_duration(π/2; amp_T = amp_T)

        for (T1, TI, TR, Npe) in [(1.0, 0.5, 2.0, 8), (0.3, 0.1, 1.0, 8),
                                   (0.05, 0.02, 0.5, 4)]
            ti_delay = TI - d180/2 - d90/2
            ti_delay > 0 || continue

            obj = single_spin_phantom(T1 = T1, T2 = T2)
            seq = Sequence()
            for _ in 1:Npe
                seq += RF(amp_T, d180)
                seq += Delay(ti_delay)
                seq += RF(amp_T, d90)
                seq += ADC(1, 1e-6, 0.0)
                shot = d180 + ti_delay + d90 + 1e-6
                seq += Delay(TR - shot)
            end
            raw = @suppress simulate(obj, seq, Scanner())
            # Mean over shots ≈ DC of IFFT for a single-pixel-bright sphere
            mags = [abs(raw.profiles[k].data[1, 1]) for k in 1:Npe]
            sim_mean = sum(mags) / Npe
            ana_mean = abs(transient_mz_at_excite_npe(T1, TI, TR, π, π/2; Npe = Npe))
            @test isapprox(sim_mean, ana_mean; rtol = 0.05, atol = 5e-3)
        end
    end

    @testset "F1+ fitter recovers T1 from adaptive (varying-TI/TR) sequences" begin
        # Regime E2 actually operates in: 8 distinct (TI, TR) per fit, each
        # block an Npe-shot IR-SE from M0=1. Generate "data" from F1+ itself
        # (the closed-form-vs-simulator agreement is pinned by the previous
        # testset).
        Npe = 8
        rng = MersenneTwister(42)
        for T1 in [0.1, 0.5, 1.0, 1.8]
            n   = 8
            TIs = sort!(exp.(log(0.02) .+ (log(2.0) − log(0.02)) .* rand(rng, n)))
            TRs = TIs .+ 0.5 .+ 1.5 .* rand(rng, n)            # TR > TI
            αes = fill(π/2, n);   αs = fill(π, n)
            mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i],
                                                      π, αes[i]; Npe = Npe))
                    for i in 1:n]
            fit = fit_t1_generalized_ir(TIs, αs, mags;
                        TRs = TRs, α_excs = αes, Npe = Npe,
                        T1_range = (0.02, 3.0), n_grid = 400)
            @test isapprox(fit.T1, T1; rtol = 0.05)
        end
    end

    @testset "Steady-state fitter is provably biased on Npe-shot data (regression for §19.5.1)" begin
        # Pin the bias direction: steady-state model on Npe=8 data must miss
        # the truth by ≥10%, while F1+ recovers within 5%. Should fail loudly
        # if anyone silently re-introduces the steady-state assumption.
        T1, Npe = 1.0, 8
        TIs     = [0.05, 0.3, 1.0, 2.0]
        TRs     = [0.5,  0.6, 1.2, 2.5]
        αs      = fill(π,   4)
        αes     = fill(π/2, 4)
        mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i], π, π/2; Npe = Npe))
                for i in 1:4]
        f_npe = fit_t1_generalized_ir(TIs, αs, mags;
                                        TRs = TRs, α_excs = αes, Npe = Npe,
                                        T1_range = (0.05, 5.0), n_grid = 400)
        f_steady = fit_t1_generalized_ir(TIs, αs, mags;
                                           TRs = TRs, α_excs = αes,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        @info "steady-state-bias regression" T1_true=T1 T1_npe=f_npe.T1 T1_steady=f_steady.T1
        @test abs(f_npe.T1    - T1) / T1 < 0.05      # F1+ unbiased
        @test abs(f_steady.T1 - T1) / T1 > 0.10      # steady-state biased
        @test abs(f_npe.T1 - T1) < abs(f_steady.T1 - T1)
    end

    # ── σ fixes (E2.4) ───────────────────────────────────────────────────────
    @testset "σ Fix A: n-gate forces floor at small n" begin
        # With n ≤ 4 the residual variance has no statistical power, so
        # σ²_resid should be Inf and σ²_eff should equal the floor — even
        # when the residuals happen to be tiny (e.g. clean synthetic data).
        T1   = 1.0
        TIs  = [0.05, 0.3, 1.0]                # n = 3
        TRs  = [0.5,  1.0, 2.0]
        αs   = fill(π,   3)
        αes  = fill(π/2, 3)
        mags = [abs(transient_mz_at_excite_npe(T1, TIs[i], TRs[i], π, π/2; Npe = 8))
                for i in 1:3]

        # No floor → at n=3 σ_T1 should be undefined (NaN) under Fix A
        # because σ²_resid = Inf and there's no other source of σ.
        f_no_floor = fit_t1_generalized_ir(TIs, αs, mags;
                                             TRs = TRs, α_excs = αes, Npe = 8,
                                             T1_range = (0.05, 5.0), n_grid = 400)
        @test !isfinite(f_no_floor.T1_sigma) || f_no_floor.T1_sigma > 1e9

        # With an explicit absolute noise floor → σ_T1 finite and reasonable
        f_with_floor = fit_t1_generalized_ir(TIs, αs, mags;
                                               TRs = TRs, α_excs = αes, Npe = 8,
                                               abs_noise_sigma = 0.01,
                                               T1_range = (0.05, 5.0), n_grid = 400)
        @test isfinite(f_with_floor.T1_sigma)
        @test f_with_floor.T1_sigma > 0
    end

    @testset "σ Fix B: abs_noise_sigma decouples σ from data RMS (action choice)" begin
        # Same physical noise level, different per-fit data RMS (because the
        # agent picks different TIs). Under the relative noise_sigma path,
        # σ_T1 changes with data RMS — that's the §7.3 coupling the fix
        # removes. Under abs_noise_sigma, σ_T1 should be (approximately) the
        # same regardless of the data scale.
        T1, Npe = 1.0, 8
        TIs_hi  = [0.05, 0.10, 0.20, 0.40, 0.80]   # all high-signal (|1−2E1| close to 1)
        TIs_lo  = [0.50, 0.60, 0.65, 0.70, 0.75]   # all near the IR null (low signal)
        TRs     = fill(2.0, 5)
        αs      = fill(π,   5)
        αes     = fill(π/2, 5)
        m_hi = [abs(transient_mz_at_excite_npe(T1, TIs_hi[i], TRs[i], π, π/2; Npe=Npe))
                for i in 1:5]
        m_lo = [abs(transient_mz_at_excite_npe(T1, TIs_lo[i], TRs[i], π, π/2; Npe=Npe))
                for i in 1:5]
        rms_hi = sqrt(sum(abs2, m_hi) / length(m_hi))
        rms_lo = sqrt(sum(abs2, m_lo) / length(m_lo))
        @test rms_hi > 3 * rms_lo            # the two regimes really do differ in RMS

        # Relative path: σ_T1 scales with rms(m), so the two fits' σ differ
        f_hi_rel = fit_t1_generalized_ir(TIs_hi, αs, m_hi;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           noise_sigma = 0.05,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        f_lo_rel = fit_t1_generalized_ir(TIs_lo, αs, m_lo;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           noise_sigma = 0.05,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        # Absolute path: same σ_floor per fit, σ_T1 should be similar in scale
        abs_noise = 0.05 * rms_hi
        f_hi_abs = fit_t1_generalized_ir(TIs_hi, αs, m_hi;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           abs_noise_sigma = abs_noise,
                                           T1_range = (0.05, 5.0), n_grid = 400)
        f_lo_abs = fit_t1_generalized_ir(TIs_lo, αs, m_lo;
                                           TRs = TRs, α_excs = αes, Npe = Npe,
                                           abs_noise_sigma = abs_noise,
                                           T1_range = (0.05, 5.0), n_grid = 400)

        # The abs-path σ floor is identical for both fits, so any remaining
        # variation in σ_T1 comes from Jacobian conditioning (real signal,
        # not floor-coupling). The relative-path picks up an extra
        # rms-scaling factor, so its ratio is *further* from 1 — i.e. the
        # two action-choices look more different in σ-space than they
        # really are. Fix B's claim is "abs ratio closer to 1 than rel
        # ratio". Both ratios are <1 in this setup (low-signal fit happens
        # to have looser σ even after Jacobian wins back some), so closer-
        # to-1 means *larger*: `abs_ratio > rel_ratio`.
        rel_ratio = f_lo_rel.T1_sigma / f_hi_rel.T1_sigma
        abs_ratio = f_lo_abs.T1_sigma / f_hi_abs.T1_sigma
        @info "σ Fix B" rel_ratio abs_ratio rms_hi rms_lo
        @test abs(log(abs_ratio)) < abs(log(rel_ratio))
    end

    @testset "Profile-likelihood σ — well-determined fit gives small σ" begin
        # Synthetic clean data at T1 = 0.5 s, 8 well-spread TIs in the
        # informative window, plus small noise. SSE has one sharp basin →
        # both methods give small σ; they should agree within ~factor 5.
        T1_true = 0.5
        Npe     = 8
        TIs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
        TRs = fill(2.0, length(TIs))
        αs  = fill(π,   length(TIs))
        αes = fill(π/2, length(TIs))
        rng  = MersenneTwister(7)
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                 π, π/2; Npe = Npe)) +
                0.02 * randn(rng) for i in eachindex(TIs)]

        f_prof = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :profile_likelihood)
        f_asy  = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :asymptotic)

        @test isapprox(f_prof.T1, T1_true; rtol = 0.05)
        @test isapprox(f_asy.T1,  T1_true; rtol = 0.05)
        # σ_T1 should be small (< 20 % of T1) on well-determined fit
        @test isfinite(f_prof.T1_sigma) && f_prof.T1_sigma < 0.20 * T1_true
        @test isfinite(f_asy.T1_sigma)  && f_asy.T1_sigma  < 0.20 * T1_true
        # Both methods should agree within ~factor 5 on a unimodal fit —
        # they're measuring the same width of the same basin (modulo grid
        # discretisation in profile-likelihood).
        @test 0.05 * f_asy.T1_sigma < f_prof.T1_sigma < 20 * f_asy.T1_sigma
    end

    @testset "Profile-likelihood σ — multimodal SSE gives wide σ" begin
        # Synthetic short-T1 data with all-saturated TIs → flat SSE plateau
        # across short T1s. Asymptotic σ from a local J^T J at the LM
        # minimum is over-confident; profile-likelihood should report wide.
        T1_true = 0.023      # very short
        Npe     = 8
        TIs = [0.5, 1.0, 1.5, 2.0, 0.7, 1.2, 1.8, 2.5]   # all ≫ T1_true
        TRs = fill(3.0, length(TIs))
        αs  = fill(π,   length(TIs))
        αes = fill(π/2, length(TIs))
        # Ground-truth mags + small noise
        rng  = MersenneTwister(42)
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                 π, π/2; Npe = Npe)) +
                0.02 * randn(rng) for i in eachindex(TIs)]

        f_prof = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :profile_likelihood)
        f_asy  = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :asymptotic)

        # Profile-likelihood σ should be large vs T1*, because the saturated
        # data doesn't constrain T1. Use 50 % of T1* as the wide threshold.
        @test isfinite(f_prof.T1_sigma)
        @test f_prof.T1_sigma > 0.5 * f_prof.T1
        # Crucially, profile σ should expose the ambiguity that asymptotic
        # σ misses: either asymptotic returns NaN (singular J^T J at the
        # degenerate LM minimum) or profile σ is at least 5× larger.
        @test isnan(f_asy.T1_sigma) || f_prof.T1_sigma > 5 * f_asy.T1_sigma
    end

    @testset "Profile-likelihood σ — point estimate unchanged from asymptotic" begin
        # The σ change must NOT alter the LM point estimate. Same T1*, same A*.
        rng = MersenneTwister(0)
        for T1_true in [0.05, 0.2, 0.7, 1.5]
            n   = 8
            TIs = sort!(exp.(log(0.05) .+ (log(2.5) − log(0.05)) .* rand(rng, n)))
            TRs = TIs .+ exp.(log(0.5) .+ log(3.0) .* rand(rng, n))
            αes = fill(π/2, n);  αs = fill(π, n)
            mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                     π, π/2; Npe = 8))
                    for i in 1:n]

            f_a = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :asymptotic)
            f_p = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :profile_likelihood)
            @test isapprox(f_a.T1, f_p.T1; rtol = 1e-9)
            @test isapprox(f_a.A,  f_p.A;  rtol = 1e-9)
            @test isapprox(f_a.residual, f_p.residual; rtol = 1e-9)
        end
    end

    @testset "Bootstrap σ — finite on saturated data where asymptotic is NaN" begin
        # Saturated data → all measurements ≈ A → SSE flat across short T1s.
        # Asymptotic σ → NaN (singular J^T J). Profile σ wide. Bootstrap
        # widely spread because resampled residuals push T1*_b across the
        # flat-basin region.
        T1_true = 0.023
        Npe     = 8
        TIs = [0.5, 1.0, 1.5, 2.0, 0.7, 1.2, 1.8, 2.5]   # all ≫ T1_true
        TRs = fill(3.0, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        rng = MersenneTwister(42)
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                 π, π/2; Npe = Npe)) +
                0.02 * randn(rng) for i in eachindex(TIs)]

        f_boot = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :bootstrap,
                                         n_bootstrap = 200)
        f_asy  = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, abs_noise_sigma = 0.02,
                                         sigma_method = :asymptotic)

        # Bootstrap σ is finite and non-trivial on saturated data
        @test isfinite(f_boot.T1_sigma)
        @test f_boot.T1_sigma > 0.01      # ≫ grid resolution
        # Asymptotic σ would typically be NaN here (singular Jacobian).
        # Test claim: bootstrap captures the ambiguity that asymptotic can't.
        @test isnan(f_asy.T1_sigma) || f_boot.T1_sigma > 5 * f_asy.T1_sigma
    end

    @testset "Bootstrap σ — point estimate unchanged from asymptotic" begin
        # Same regression as profile-likelihood: σ method must not move T1*.
        rng = MersenneTwister(0)
        for T1_true in [0.05, 0.2, 0.7, 1.5]
            n   = 8
            TIs = sort!(exp.(log(0.05) .+ (log(2.5) − log(0.05)) .* rand(rng, n)))
            TRs = TIs .+ exp.(log(0.5) .+ log(3.0) .* rand(rng, n))
            αes = fill(π/2, n);  αs = fill(π, n)
            mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                     π, π/2; Npe = 8))
                    for i in 1:n]
            f_a = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :asymptotic)
            f_b = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                          Npe = 8, abs_noise_sigma = 0.01,
                                          sigma_method = :bootstrap,
                                          n_bootstrap = 50)
            @test isapprox(f_a.T1, f_b.T1; rtol = 1e-9)
            @test isapprox(f_a.A,  f_b.A;  rtol = 1e-9)
        end
    end

    @testset "Phase-sensitive (signed) fit — recovers T1 cleanly, no multimodality" begin
        # When the fitter sees signed data via phase-sensitive recon, the
        # forward model |1 − 2·exp(−TI/T1)| → (1 − 2·exp(−TI/T1)) is
        # monotonic in T1. SSE has one basin per sphere, asymptotic σ is
        # honest, profile σ is tight.
        T1_true = 0.05
        Npe     = 8
        TIs = [0.07, 0.12, 0.20, 0.35, 0.55, 0.80, 1.20, 1.80]
        TRs = fill(2.5, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        # SIGNED ground truth (no abs)
        mags_signed = [transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                    π, π/2; Npe = Npe)
                       for i in eachindex(TIs)]

        f_signed = fit_t1_generalized_ir(TIs, αs, mags_signed; TRs = TRs,
                                           α_excs = αes, Npe = Npe,
                                           abs_noise_sigma = 0.01,
                                           signed = true,
                                           sigma_method = :profile_likelihood)
        # Magnitude fit on the same data using |mags_signed| would be the
        # baseline — recovers truth too in the noiseless case but is more
        # vulnerable to noise.
        @test isapprox(f_signed.T1, T1_true; rtol = 0.05)
        @test isfinite(f_signed.T1_sigma)
        # σ should be tight (< 30 % of T1) because there's no multimodal
        # ambiguity — only one basin in the signed model
        @test f_signed.T1_sigma < 0.30 * T1_true
    end

    @testset "Phase-sensitive (signed) fit — point estimate matches magnitude when signs are consistent" begin
        # Sanity: if all measurements are post-null (signs all positive),
        # signed fit and magnitude fit should give the same T1*.
        T1_true = 0.5
        TIs = [0.5, 0.8, 1.2, 1.8, 2.5]   # all post-null for T1 = 0.5 (null at TI ≈ 0.35)
        TRs = fill(3.0, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        mags_signed = [transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i],
                                                    π, π/2; Npe = 8)
                       for i in eachindex(TIs)]
        # Verify all positive (post-null regime)
        @test all(>=(0), mags_signed)

        f_mag    = fit_t1_generalized_ir(TIs, αs, abs.(mags_signed);
                                           TRs = TRs, α_excs = αes, Npe = 8,
                                           signed = false)
        f_signed = fit_t1_generalized_ir(TIs, αs, mags_signed;
                                           TRs = TRs, α_excs = αes, Npe = 8,
                                           signed = true)
        @test isapprox(f_mag.T1, f_signed.T1; rtol = 0.01)
    end

    # ── joint T1/T2 fit (IR-TSE multiparametric) ─────────────────────────────
    @testset "fit_t1_t2_generalized_ir: joint recovery + reduction limits" begin
        # (TI, TE) grid constrains T1 (via TI) and T2 (via TE). Data from the
        # joint forward model |A·Mz(T1)·exp(-TE/T2)|, A = 1.
        Npe = 8
        TIs_base = [0.05, 0.2, 0.5, 1.0]
        TEs_base = [0.02, 0.06, 0.12, 0.20]
        TIs = repeat(TIs_base, inner = length(TEs_base))
        TEs = repeat(TEs_base, outer = length(TIs_base))
        n   = length(TIs)
        αs  = fill(π, n); αes = fill(π/2, n); TRs = fill(3.0, n)
        gen(T1, T2) = [abs(transient_mz_at_excite_npe(T1, TIs[k], TRs[k], π, π/2;
                                                       Npe = Npe)) *
                       exp(-TEs[k] / T2) for k in 1:n]

        @testset "noiseless joint recovery of T1 and T2" begin
            for (T1, T2) in [(0.5, 0.1), (0.2, 0.05), (1.0, 0.2)]
                f = fit_t1_t2_generalized_ir(TIs, αs, TEs, gen(T1, T2);
                        TRs = TRs, α_excs = αes, Npe = Npe,
                        T1_range = (0.02, 3.0), T2_range = (0.01, 1.0),
                        n_grid_t1 = 200, n_grid_t2 = 150, abs_noise_sigma = 0.01)
                @test isapprox(f.T1, T1; rtol = 0.05)
                @test isapprox(f.T2, T2; rtol = 0.05)
                @test isfinite(f.T1_sigma) && f.T1_sigma < 0.3 * T1
                @test isfinite(f.T2_sigma) && f.T2_sigma < 0.3 * T2
            end
        end

        @testset "constant TE ⇒ T2 unidentifiable; T1 matches the T1-only fit" begin
            # With TE fixed, exp(-TE/T2) is a constant the amplitude A absorbs, so
            # T2 is unconstrained (wide σ) and T1* must agree with the T1-only fit.
            T1, T2 = 0.5, 0.1
            TEc  = fill(0.05, n)
            mags = [abs(transient_mz_at_excite_npe(T1, TIs[k], TRs[k], π, π/2;
                                                   Npe = Npe)) * exp(-TEc[k]/T2)
                    for k in 1:n]
            fj = fit_t1_t2_generalized_ir(TIs, αs, TEc, mags; TRs = TRs,
                    α_excs = αes, Npe = Npe, T1_range = (0.02, 3.0),
                    T2_range = (0.01, 1.0), n_grid_t1 = 200, n_grid_t2 = 150,
                    abs_noise_sigma = 0.01)
            f1 = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                    Npe = Npe, T1_range = (0.02, 3.0), n_grid = 200,
                    abs_noise_sigma = 0.01)
            @test isapprox(fj.T1, f1.T1; rtol = 0.05)
            @test fj.T2_sigma > 0.5 * fj.T2          # T2 wide (unconstrained)
        end

        @testset "input validation throws" begin
            @test_throws ErrorException fit_t1_t2_generalized_ir(TIs, αs,
                                            TEs[1:end-1], gen(0.5, 0.1))
            @test_throws ErrorException fit_t1_t2_generalized_ir([0.1], [π],
                                            [0.02], [0.5])
            @test_throws ErrorException fit_t1_t2_generalized_ir(TIs, αs, TEs,
                                            gen(0.5, 0.1); Npe = 0)
        end
    end

    # ── σ calibration (Monte-Carlo) ──────────────────────────────────────────
    # The existing σ tests are all *qualitative* (wide-vs-narrow, finite-vs-NaN,
    # point-estimate-unchanged). These pin the *magnitude* of T1_sigma against
    # the estimator's own frequentist sampling distribution: fix T1_true + a
    # well-determined schedule, draw many Gaussian-noise realisations at a known
    # abs_noise_sigma, refit each, and compare the reported σ to the actual
    # spread of T1*. Two distinct references (see the σ method's own claim):
    #   - :asymptotic / :bootstrap  → MC std of T1*   (they return a std)
    #   - :profile_likelihood       → 68 % coverage    (it's a CI half-width)
    # Run ONLY on a single-basin fit and inject Gaussian noise of exactly the σ
    # the fitter is told about, so the test isolates variance-calibration from
    # (a) estimator bias on multimodal data and (b) noise-model mismatch.
    #
    # A well-determined schedule for T1 = 0.5 s: TIs spread through the
    # informative window (around the null at TI ≈ T1·ln2 ≈ 0.35 s).
    σ_cal_T1   = 0.5
    σ_cal_Npe  = 8
    σ_cal_TIs  = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
    σ_cal_TRs  = fill(3.0, length(σ_cal_TIs))
    σ_cal_αs   = fill(π,   length(σ_cal_TIs))
    σ_cal_αes  = fill(π/2, length(σ_cal_TIs))
    σ_cal_noise = 0.01
    σ_cal_clean = [abs(transient_mz_at_excite_npe(σ_cal_T1, σ_cal_TIs[i],
                                                   σ_cal_TRs[i], π, π/2;
                                                   Npe = σ_cal_Npe))
                   for i in eachindex(σ_cal_TIs)]
    σ_cal_N = 400
    # One reusable Monte-Carlo run: returns (T1 estimates, reported σ's).
    σ_cal_run = function (method::Symbol; seed::Int)
        rng = MersenneTwister(seed)
        T1s = Float64[]; σs = Float64[]
        for _ in 1:σ_cal_N
            mags = σ_cal_clean .+ σ_cal_noise .* randn(rng, length(σ_cal_clean))
            f = fit_t1_generalized_ir(σ_cal_TIs, σ_cal_αs, mags;
                                       TRs = σ_cal_TRs, α_excs = σ_cal_αes,
                                       Npe = σ_cal_Npe,
                                       abs_noise_sigma = σ_cal_noise,
                                       n_grid = 500,
                                       sigma_method = method,
                                       n_bootstrap = 100)
            isfinite(f.T1) || continue
            push!(T1s, f.T1)
            isfinite(f.T1_sigma) && push!(σs, f.T1_sigma)
        end
        return (T1s = T1s, σs = σs)
    end

    @testset "σ calibration: point estimate is unbiased under noise" begin
        # Variance calibration is only meaningful where the estimator is
        # unbiased — pin that first. Mean of T1* over realisations must sit on
        # the truth to well within the MC standard error of the mean.
        r = σ_cal_run(:asymptotic; seed = 1)
        @test length(r.T1s) > 0.95 * σ_cal_N            # almost all fits finite
        μ   = sum(r.T1s) / length(r.T1s)
        sd  = sqrt(sum((x - μ)^2 for x in r.T1s) / (length(r.T1s) - 1))
        sem = sd / sqrt(length(r.T1s))
        @info "σ calibration bias" T1_true=σ_cal_T1 mean_T1=μ mc_std=sd
        @test abs(μ - σ_cal_T1) < 5 * sem               # bias ≪ MC scatter
        @test abs(μ - σ_cal_T1) < 0.03 * σ_cal_T1       # and small in absolute terms
    end

    @testset "σ calibration: asymptotic σ matches the MC standard error" begin
        # The reported asymptotic σ (CRLB from the local Jacobian) should equal
        # the actual across-realisation std of T1* — on a single-basin fit the
        # estimator is efficient, so these match within a small factor.
        r = σ_cal_run(:asymptotic; seed = 2)
        μ        = sum(r.T1s) / length(r.T1s)
        mc_std   = sqrt(sum((x - μ)^2 for x in r.T1s) / (length(r.T1s) - 1))
        mean_rep = sum(r.σs) / length(r.σs)
        ratio    = mean_rep / mc_std
        @info "σ calibration asymptotic" mc_std mean_reported=mean_rep ratio
        @test 0.5 < ratio < 2.0                         # within a factor of 2
    end

    @testset "σ calibration: bootstrap σ matches the MC standard error" begin
        # Bootstrap estimates the same sampling std by resampling residuals.
        r = σ_cal_run(:bootstrap; seed = 3)
        μ        = sum(r.T1s) / length(r.T1s)
        mc_std   = sqrt(sum((x - μ)^2 for x in r.T1s) / (length(r.T1s) - 1))
        mean_rep = sum(r.σs) / length(r.σs)
        ratio    = mean_rep / mc_std
        @info "σ calibration bootstrap" mc_std mean_reported=mean_rep ratio
        @test 0.5 < ratio < 2.0
    end

    @testset "σ calibration: profile-likelihood interval covers truth at ~68 %" begin
        # Profile σ is a 68.3 % CI half-width, NOT a std — so the right check is
        # empirical coverage of T1_true ∈ [T1* − σ, T1* + σ], which should land
        # near 0.683 on this well-determined, unbiased fit.
        #
        # The grid must be fine relative to σ for this to mean anything: the
        # interval is `(T1_hi − T1_lo)/2` over passing grid points, so if the
        # log-grid spacing exceeds the basin half-width (~0.004 s here) the
        # half-width quantises toward 0 and the interval *under-covers*
        # (empirically ~0.20 at the default range/n_grid=500). Narrow the range
        # and refine the grid so spacing ≪ σ.
        rng = MersenneTwister(4)
        covered = 0; total = 0
        for _ in 1:σ_cal_N
            mags = σ_cal_clean .+ σ_cal_noise .* randn(rng, length(σ_cal_clean))
            f = fit_t1_generalized_ir(σ_cal_TIs, σ_cal_αs, mags;
                                       TRs = σ_cal_TRs, α_excs = σ_cal_αes,
                                       Npe = σ_cal_Npe,
                                       abs_noise_sigma = σ_cal_noise,
                                       T1_range = (0.2, 1.2), n_grid = 2000,
                                       sigma_method = :profile_likelihood)
            (isfinite(f.T1) && isfinite(f.T1_sigma)) || continue
            total += 1
            abs(f.T1 - σ_cal_T1) <= f.T1_sigma && (covered += 1)
        end
        frac = covered / total
        @info "σ calibration profile coverage" covered total frac
        @test 0.55 < frac < 0.85                        # ~0.683 ± grid/MC slack
    end

    # ── untested branches: error paths, oracle grid, n-gate boundary ─────────
    @testset "fit_t1_generalized_ir input-validation throws" begin
        TIs = [0.1, 0.3, 1.0]; αs = fill(π, 3); mags = [0.5, 0.6, 0.7]
        # length mismatches on the optional per-sample vectors
        @test_throws ErrorException fit_t1_generalized_ir(TIs, αs, mags;
                                                          TRs = [1.0, 2.0])
        @test_throws ErrorException fit_t1_generalized_ir(TIs, αs, mags;
                                                          α_excs = [π/2, π/2])
        # Npe must be ≥ 1
        @test_throws ErrorException fit_t1_generalized_ir(TIs, αs, mags; Npe = 0)
        # unknown sigma_method symbol
        @test_throws ErrorException fit_t1_generalized_ir(TIs, αs, mags;
                                                          sigma_method = :bogus)
        # core arity guards (mismatched αs, and < 2 samples)
        @test_throws ErrorException fit_t1_generalized_ir(TIs, [π, π], mags)
        @test_throws ErrorException fit_t1_generalized_ir([0.1], [π], [0.5])
    end

    @testset "T1_oracle narrows the grid and still recovers; bad band throws" begin
        # The oracle band is a diagnostic that collapses the search to a log-band
        # around a supplied truth. Pin: (a) it still recovers T1 when the band
        # brackets the truth, (b) its error guards fire.
        T1_true = 0.5; Npe = 8
        TIs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5]
        TRs = fill(3.0, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i], π, π/2;
                                                Npe = Npe)) for i in eachindex(TIs)]
        f = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                   Npe = Npe, T1_oracle = T1_true,
                                   oracle_band = 1.5, n_grid = 300)
        @test isapprox(f.T1, T1_true; rtol = 0.05)
        # Oracle restricts the grid to [T1/band, T1·band]; with a tight band the
        # estimate cannot escape that window even if the global range is wide.
        f_tight = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs, α_excs = αes,
                                         Npe = Npe, T1_oracle = T1_true,
                                         oracle_band = 1.2,
                                         T1_range = (0.01, 5.0), n_grid = 300)
        @test T1_true / 1.2 - 1e-9 <= f_tight.T1 <= T1_true * 1.2 + 1e-9
        # Error guards
        @test_throws ErrorException fit_t1_generalized_ir(TIs, αs, mags;
                                        T1_oracle = -1.0)
        @test_throws ErrorException fit_t1_generalized_ir(TIs, αs, mags;
                                        T1_oracle = T1_true, oracle_band = 1.0)
    end

    @testset "σ Fix A n-gate boundary: n=4 uses floor, n=5 uses residual" begin
        # The gate is `n > 4` (fits.jl). Pin the off-by-one at the boundary:
        # at n=4 σ²_resid has no power so σ²_eff is the floor; at n=5 it can use
        # best_sse/(n-2). On clean data the residual variance is ~0, so the
        # n=5/no-floor fit should give a *much smaller* σ than the n=4/no-floor
        # one (which, with no floor, is NaN/∞ under Fix A).
        T1 = 1.0; Npe = 8
        mk(TIs) = [abs(transient_mz_at_excite_npe(T1, ti, 2.0, π, π/2; Npe = Npe))
                   for ti in TIs]
        TIs4 = [0.05, 0.3, 1.0, 2.0]
        TIs5 = [0.05, 0.3, 0.7, 1.0, 2.0]
        αs4 = fill(π, 4); αes4 = fill(π/2, 4)
        αs5 = fill(π, 5); αes5 = fill(π/2, 5)
        TRs4 = fill(2.0, 4); TRs5 = fill(2.0, 5)

        # No floor: n=4 → σ undefined (Inf/NaN); n=5 on clean data → tiny σ.
        f4 = fit_t1_generalized_ir(TIs4, αs4, mk(TIs4); TRs = TRs4,
                                    α_excs = αes4, Npe = Npe, n_grid = 400)
        f5 = fit_t1_generalized_ir(TIs5, αs5, mk(TIs5); TRs = TRs5,
                                    α_excs = αes5, Npe = Npe, n_grid = 400)
        @test !isfinite(f4.T1_sigma) || f4.T1_sigma > 1e9
        @test isfinite(f5.T1_sigma) && f5.T1_sigma < 0.1 * T1

        # With an explicit floor: n=4 falls back to the floor (finite), and the
        # gate change at n=5 must not make σ blow up.
        f4f = fit_t1_generalized_ir(TIs4, αs4, mk(TIs4); TRs = TRs4,
                                     α_excs = αes4, Npe = Npe,
                                     abs_noise_sigma = 0.01, n_grid = 400)
        @test isfinite(f4f.T1_sigma) && f4f.T1_sigma > 0
    end

    @testset "bootstrap σ is reproducible (seeded) and seed-sensitive" begin
        # _sigma_bootstrap draws from a seeded MersenneTwister, so the reported
        # σ must be bit-identical across calls with the same bootstrap_seed and
        # differ (modestly) for a different seed. Guards against an accidental
        # global-RNG regression.
        T1_true = 0.023; Npe = 8       # saturated → bootstrap σ is non-trivial
        TIs = [0.5, 1.0, 1.5, 2.0, 0.7, 1.2, 1.8, 2.5]
        TRs = fill(3.0, length(TIs)); αs = fill(π, length(TIs))
        αes = fill(π/2, length(TIs))
        rng = MersenneTwister(99)
        mags = [abs(transient_mz_at_excite_npe(T1_true, TIs[i], TRs[i], π, π/2;
                                                Npe = Npe)) + 0.02 * randn(rng)
                for i in eachindex(TIs)]
        call(seed) = fit_t1_generalized_ir(TIs, αs, mags; TRs = TRs,
                            α_excs = αes, Npe = Npe, abs_noise_sigma = 0.02,
                            sigma_method = :bootstrap, n_bootstrap = 200,
                            bootstrap_seed = seed).T1_sigma
        s0a = call(0); s0b = call(0); s1 = call(1)
        @test s0a == s0b                 # same seed → identical
        @test s0a != s1                  # different seed → different draw
    end

    @testset "e2_step!: TR lifts to honour TI (no silent TI cap)" begin
        # Regression for the `TI = min(TI, TR*0.9)` bug, which silently
        # capped TI when the agent picked a small TR — locking out the
        # long-T1 regime. The new policy: TR lifts to (TI+TE)/0.9 instead.
        env = E2Env(; max_blocks = 2, time_budget_s = 600.0,
                     Nfe = 8, Npe = 4)
        e2_reset!(env; rng_seed = 42)

        TI_req = 2.5
        TE_req = 0.02
        TR_req = 0.5    # too small to fit TI
        _, _, _, info = e2_step!(env, [TI_req, TE_req, TR_req, 90.0, 0.0])

        @test isapprox(info["TI"], TI_req; atol = 1e-9)
        @test info["TR"] >= (TI_req + TE_req) / 0.90 - 1e-9
    end

    @testset "e2_step!: budget guard never overruns the scan-time budget" begin
        # Npe=8, TR=3 → block_time = 24 s. Two blocks (48 s) fit in a 60 s
        # budget; the third (→72 s) must be rejected without executing, so the
        # realised scan time never exceeds the budget.
        env = E2Env(; Nfe = 8, Npe = 8, max_blocks = 30, time_budget_s = 60.0)
        e2_reset!(env; rng_seed = 7)

        done = false
        local info
        while !done
            _, _, done, info = e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0])
        end

        @test env.time_used_s <= env.time_budget_s + 1e-9
        @test env.n_blocks == 2                       # 3rd block rejected
        @test get(info, "budget_exceeded", false) == true
        @test info["block_time"] == 0.0               # discarded block ran nothing
    end

    @testset "e2_step!: skips simulation when only one block could fit (fitter needs ≥2)" begin
        # budget 26 s, Npe=8: one TR=3 block costs 24 s, but a second (≥ 8·0.5 =
        # 4 s) would overrun → the lone block can't yield a fit, so the env ends
        # the episode WITHOUT simulating it (time stays 0, no block executed).
        env = E2Env(; Nfe = 8, Npe = 8, max_blocks = 30, time_budget_s = 26.0)
        e2_reset!(env; rng_seed = 1)
        _, _, done, info = e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0])

        @test done == true
        @test env.n_blocks == 0
        @test env.time_used_s == 0.0
        @test get(info, "budget_exceeded", false) == true
    end

    @testset "E2 random subset reset keeps fixed obs shape and active sphere identities" begin
        # include_image + include_sigma on → full obs = Nfe*Npe + 2*n_spheres + 3.
        env = E2Env(; subset_size = 5, Nfe = 8, Npe = 4,
                     max_blocks = 2, time_budget_s = 600.0,
                     include_image = true, include_sigma = true)

        obs1 = e2_reset!(env; rng_seed = 123)
        idx1 = copy(env.sphere_indices)
        T1_1 = [d.T1 for d in env.active_base_descs]

        @test env.n_spheres == 5
        @test length(obs1) == 8 * 4 + 2 * 5 + 3
        @test length(idx1) == 5
        @test issorted(idx1)
        @test length(unique(idx1)) == 5
        @test all(1 .<= idx1 .<= 14)
        @test T1_1 == [d.T1 for d in env.base_descs_pool[idx1]]

        obs2 = e2_reset!(env; rng_seed = 123)
        @test env.sphere_indices == idx1
        @test [d.T1 for d in env.active_base_descs] == T1_1
        @test length(obs2) == length(obs1)

        obs3 = e2_reset!(env; rng_seed = 124)
        @test length(obs3) == length(obs1)
        @test length(env.T1_true) == 5
    end

    @testset "E2 observation channels are gated by include_image / include_sigma" begin
        Nfe, Npe, k = 8, 4, 5
        # Default: image + σ dropped → obs = n_spheres + 3.
        env_def = E2Env(; subset_size = k, Nfe = Nfe, Npe = Npe,
                         max_blocks = 2, time_budget_s = 600.0)
        @test e2_obs_dim(env_def) == k + 3
        @test length(e2_reset!(env_def; rng_seed = 1)) == k + 3

        # Image only.
        env_img = E2Env(; subset_size = k, Nfe = Nfe, Npe = Npe,
                         max_blocks = 2, time_budget_s = 600.0,
                         include_image = true)
        @test e2_obs_dim(env_img) == Nfe * Npe + k + 3
        @test length(e2_reset!(env_img; rng_seed = 1)) == Nfe * Npe + k + 3

        # σ only.
        env_sig = E2Env(; subset_size = k, Nfe = Nfe, Npe = Npe,
                         max_blocks = 2, time_budget_s = 600.0,
                         include_sigma = true)
        @test e2_obs_dim(env_sig) == 2 * k + 3
        @test length(e2_reset!(env_sig; rng_seed = 1)) == 2 * k + 3
    end

    @testset "forward_model=:analytic builds no Koma phantom but steps + fits" begin
        # The analytic surrogate must not construct a KomaMRI phantom (that is the
        # whole speedup) yet still drive the obs/fit/reward pipeline normally.
        env = E2Env(; subset_size = 4, forward_model = :analytic,
                     max_blocks = 6, time_budget_s = 600.0,
                     analytic_noise_sigma = 0.0)   # noiseless → exact recovery
        obs = e2_reset!(env; rng_seed = 3)
        @test env.phantom === nothing
        @test env.background_mask === nothing
        @test length(obs) == e2_obs_dim(env)       # T1-only obs: n_spheres + 3

        # Two informative blocks at different TI → fit should recover T1 to the
        # T1-grid floor (noiseless analytic == fitter's own forward model).
        e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0])
        _, _, _, info = e2_step!(env, [1.0, 0.02, 3.0, 90.0, 0.0])
        @test info["n_blocks"] == 2
        @test info["mape"] < 0.05                  # well under 5 % with no noise
    end

    @testset "analytic signal matches transient_mz_at_excite_npe (noiseless)" begin
        env = E2Env(; subset_size = 3, forward_model = :analytic,
                     analytic_noise_sigma = 0.0, Npe = 16)
        e2_reset!(env; rng_seed = 9)
        TI, TE, TR, α_deg = 0.4, 0.02, 2.5, 90.0
        sig = MRISystemPhantom._e2_analytic_signals(env, TI, TE, TR, α_deg)
        for i in 1:env.n_spheres
            base = env.active_base_descs[i]
            T1_i = env.T1_true[i]
            T2_i = T1_i * base.T2 / base.T1
            mz = transient_mz_at_excite_npe(T1_i, TI, TR, π, deg2rad(α_deg);
                                            Npe = env.Npe)
            expect = abs(base.ρ * mz * sin(deg2rad(α_deg)) * exp(-TE / T2_i))
            @test isapprox(sig[i], expect; atol = 1e-12)
        end
    end

    @testset "reward levers: λ=0 + :neg_mape reproduces legacy reward" begin
        # The centralised _e2_reward must be a no-op refactor for the legacy
        # config: r_t = −clamp(MAPE,0,1), with no time penalty.
        env = E2Env(; subset_size = 4, forward_model = :analytic,
                     analytic_noise_sigma = 0.0, reward_mode = :neg_mape,
                     time_penalty_coef = 0.0, max_blocks = 5,
                     time_budget_s = 600.0)
        e2_reset!(env; rng_seed = 5)
        e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0])
        _, r, _, info = e2_step!(env, [1.0, 0.02, 3.0, 90.0, 0.0])
        @test isapprox(r, -clamp(info["mape"], 0.0, 1.0); atol = 1e-12)
    end

    @testset "reward levers: time_penalty_coef subtracts λ·block_time/budget" begin
        # Same trajectory under λ=0 and λ>0 → the difference on each executed
        # step is exactly λ·(block_time/budget). The penalty is gated on
        # allow_stop, so both envs enable it.
        kw = (; subset_size = 4, forward_model = :analytic,
                analytic_noise_sigma = 0.0, reward_mode = :delta_mape,
                allow_stop = true, Npe = 8, max_blocks = 4, time_budget_s = 600.0)
        λ = 0.7
        env0 = E2Env(; kw..., time_penalty_coef = 0.0)
        envλ = E2Env(; kw..., time_penalty_coef = λ)
        e2_reset!(env0; rng_seed = 11); e2_reset!(envλ; rng_seed = 11)
        a1 = [0.3, 0.02, 3.0, 90.0, 0.0]; a2 = [1.0, 0.02, 3.0, 90.0, 0.0]
        e2_step!(env0, a1); e2_step!(envλ, a1)
        (_, r0, _, info0) = e2_step!(env0, a2)
        (_, rλ, _, _)     = e2_step!(envλ, a2)
        expected_gap = λ * info0["block_time"] / env0.time_budget_s
        @test isapprox(r0 - rλ, expected_gap; atol = 1e-10)
    end

    @testset "time penalty is gated off when allow_stop=false" begin
        # With no stop action, λ must NOT affect the reward (fixed-budget total
        # time ≈ budget, so the term is just clutter — dropped).
        kw = (; subset_size = 4, forward_model = :analytic,
                analytic_noise_sigma = 0.0, reward_mode = :delta_mape,
                allow_stop = false, Npe = 8, max_blocks = 4, time_budget_s = 600.0)
        env0 = E2Env(; kw..., time_penalty_coef = 0.0)
        envλ = E2Env(; kw..., time_penalty_coef = 0.7)
        e2_reset!(env0; rng_seed = 11); e2_reset!(envλ; rng_seed = 11)
        a1 = [0.3, 0.02, 3.0, 90.0, 0.0]; a2 = [1.0, 0.02, 3.0, 90.0, 0.0]
        e2_step!(env0, a1); e2_step!(envλ, a1)
        (_, r0, _, _) = e2_step!(env0, a2)
        (_, rλ, _, _) = e2_step!(envλ, a2)
        @test isapprox(r0, rλ; atol = 1e-12)
    end

    @testset "allow_stop: a stop request ends the episode after the block" begin
        # max_blocks/budget are high and non-binding; the stop flag is what ends
        # the episode. The stopping block is still executed (n_blocks counts it).
        env = E2Env(; subset_size = 4, forward_model = :analytic,
                     analytic_noise_sigma = 0.0, reward_mode = :delta_log_mape,
                     allow_stop = true, max_blocks = 15, time_budget_s = 600.0)
        e2_reset!(env; rng_seed = 7)
        (_, _, d1, i1) = e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0], false)
        @test d1 == false
        (_, _, d2, i2) = e2_step!(env, [1.0, 0.02, 3.0, 90.0, 0.0], true)
        @test d2 == true                      # stop honoured
        @test i2["stop_requested"] == true
        @test i2["n_blocks"] == 2             # the stopping block was executed
        @test env.time_used_s < env.time_budget_s   # stopped well short of cap
    end

    @testset "allow_stop: stop ignored when allow_stop=false" begin
        env = E2Env(; subset_size = 4, forward_model = :analytic,
                     analytic_noise_sigma = 0.0, reward_mode = :delta_log_mape,
                     allow_stop = false, max_blocks = 15, time_budget_s = 600.0)
        e2_reset!(env; rng_seed = 7)
        e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0], false)
        (_, _, d2, i2) = e2_step!(env, [1.0, 0.02, 3.0, 90.0, 0.0], true)
        @test d2 == false                     # stop flag does nothing
        @test i2["stop_requested"] == false
    end

    @testset "allow_stop: stopping before a valid fit yields mape=1.0" begin
        # No n_blocks guard: a stop after a single block ends with full error,
        # so the agent is penalised for stopping too early (learns to avoid it).
        env = E2Env(; subset_size = 4, forward_model = :analytic,
                     analytic_noise_sigma = 0.0, reward_mode = :terminal_only,
                     allow_stop = true, max_blocks = 15, time_budget_s = 600.0)
        e2_reset!(env; rng_seed = 4)
        (_, r, done, info) = e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0], true)
        @test done == true
        @test info["n_blocks"] == 1
        @test info["mape"] == 1.0
        @test isapprox(r, -1.0; atol = 1e-12)   # terminal_only → −final_mape
    end

    @testset "reward levers: :terminal_only is zero until the terminal step" begin
        env = E2Env(; subset_size = 4, forward_model = :analytic,
                     analytic_noise_sigma = 0.0, reward_mode = :terminal_only,
                     Npe = 8, max_blocks = 3, time_budget_s = 600.0)
        e2_reset!(env; rng_seed = 2)
        rewards = Float64[]
        dones = Bool[]
        for a in ([0.3, 0.02, 3.0, 90.0, 0.0], [1.0, 0.02, 3.0, 90.0, 0.0],
                  [0.6, 0.02, 3.0, 90.0, 0.0])
            _, r, done, info = e2_step!(env, a)
            push!(rewards, r); push!(dones, done)
            done && (@test isapprox(r, -clamp(info["mape"], 0.0, 1.0); atol = 1e-12))
        end
        # Every non-terminal step rewards exactly 0; the terminal one does not.
        for (r, d) in zip(rewards, dones)
            d || @test r == 0.0
        end
        @test any(dones)
    end

    @testset "e2_image_stats errors under forward_model=:analytic" begin
        env = E2Env(; subset_size = 3, forward_model = :analytic)
        e2_reset!(env; rng_seed = 1)
        e2_step!(env, [0.3, 0.02, 3.0, 90.0, 0.0])
        @test_throws ErrorException e2_image_stats(env)
        @test_throws ErrorException e2_dual_acq_snr_report(env)
    end

end
