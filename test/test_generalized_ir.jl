@testset "generalized IR signal and fitting" begin

    @testset "generalized_ir_signal analytical identities" begin
        T1 = 0.5; T2 = 0.1
        sig_ir = generalized_ir_signal(T1, T2; TI = 0.5, α = π,   n_adc = 4, dur_adc = 0.0)
        sig_sr = generalized_ir_signal(T1, T2; TI = 0.5, α = π/2, n_adc = 4, dur_adc = 0.0)
        sig_eq = generalized_ir_signal(T1, T2; TI = 10.0, α = π,  n_adc = 4, dur_adc = 0.0)

        # IR at TI=T1·ln 2 is the null:
        null_ti = T1 * log(2)
        sig_null = generalized_ir_signal(T1, T2; TI = null_ti, α = π, n_adc = 1, dur_adc = 0.0)
        @test sig_null[1] < 1e-10

        # IR formula: |1 − 2·exp(−TI/T1)|
        @test isapprox(sig_ir[1], abs(1 - 2*exp(-0.5/T1)); atol = 1e-10)
        # SR formula: |1 − exp(−TI/T1)|
        @test isapprox(sig_sr[1], abs(1 - exp(-0.5/T1)); atol = 1e-10)
        # TI ≫ T1 → fully recovered
        @test isapprox(sig_eq[1], 1.0; atol = 1e-3)

        # T2 decay during readout
        s = generalized_ir_signal(1.0, 0.05; TI = 10.0, α = π, n_adc = 3, dur_adc = 0.1)
        @test s[1] > s[end]
        @test isapprox(s[end] / s[1], exp(-0.1/0.05); rtol = 1e-6)
    end

    @testset "fit_t1_generalized_ir recovers T1 from clean data" begin
        TIs = [10e-3, 30e-3, 100e-3, 300e-3, 1000e-3, 3000e-3]
        for T1_true in (0.05, 0.2, 0.5, 1.5)
            αs = fill(π, length(TIs))         # pure IR
            mags = [generalized_ir_signal(T1_true, 10.0;
                                          TI = ti, α = α, n_adc = 1, dur_adc = 0.0)[1]
                    for (ti, α) in zip(TIs, αs)]
            f = fit_t1_generalized_ir(TIs, αs, mags;
                                      T1_range = (T1_true/10, T1_true*10),
                                      n_grid = 300)
            @test isapprox(f.T1, T1_true; rtol = 0.02)
        end

        # Mixed α (IR + SR) should still recover T1
        T1_true = 0.4
        TIs = repeat([10e-3, 100e-3, 300e-3, 1000e-3], 2)
        αs  = [fill(π, 4)..., fill(π/2, 4)...]
        mags = [generalized_ir_signal(T1_true, 10.0; TI = ti, α = α,
                                      n_adc = 1, dur_adc = 0.0)[1]
                for (ti, α) in zip(TIs, αs)]
        f = fit_t1_generalized_ir(TIs, αs, mags; n_grid = 300)
        @test isapprox(f.T1, T1_true; rtol = 0.03)
    end

    @testset "fit rejects pathological input" begin
        @test_throws ErrorException fit_t1_generalized_ir([0.1], [π], [0.1])
        @test_throws ErrorException fit_t1_generalized_ir([0.1, 0.2], [π], [0.1, 0.2])
    end

end
