@testset "E1 — generalized IR + env" begin

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

    @testset "E1Env construction & dims" begin
        env = E1Env(rng_seed = 0)
        @test e1_n_actions(env) == 18
        @test e1_obs_dim(env) == 64 + 3 + 18
        @test length(e1_action_table(env)) == 18

        env2 = E1Env(TI_set = [100e-3, 300e-3],
                     α_set_deg = [90.0, 180.0], rng_seed = 0)
        @test e1_n_actions(env2) == 4

        @test_throws ErrorException E1Env(backend = :bogus)
    end

    @testset "reset determinism" begin
        a = E1Env(rng_seed = 17); e1_reset!(a)
        b = E1Env(rng_seed = 17); e1_reset!(b)
        @test a.T1_true == b.T1_true
        @test a.T2_true == b.T2_true

        c = E1Env(rng_seed = 18); e1_reset!(c)
        @test c.T1_true != a.T1_true
    end

    @testset "step! advances and terminates" begin
        env = E1Env(rng_seed = 42, max_blocks = 5)
        e1_reset!(env)
        @test !env.done
        @test env.n_blocks == 0
        for i in 1:5
            obs, r, done, info = e1_step!(env, 1)
            @test length(obs) == e1_obs_dim(env)
            @test env.n_blocks == i
            @test haskey(info, "T1_true")
            @test haskey(info, "T1_est")
            @test haskey(info, "err")
            if i < 5
                @test !done
            else
                @test done
            end
        end
        @test_throws ErrorException e1_step!(env, 1)     # after done
    end

    @testset "step! rejects invalid action" begin
        env = E1Env(rng_seed = 0); e1_reset!(env)
        @test_throws ErrorException e1_step!(env, 0)
        @test_throws ErrorException e1_step!(env, 999)
    end

    @testset "T1_true sampled inside the configured range" begin
        env = E1Env(rng_seed = 123)
        lo, hi = env.T1_sample_range
        for _ in 1:50
            e1_reset!(env)
            @test lo ≤ env.T1_true ≤ hi
            @test env.T2_true ≤ env.T1_true
            @test env.T2_true ≥ env.T2_factor_range[1] * env.T1_true - 1e-12
        end
    end

    @testset "analytical backend reaches < 5% after 12 IR blocks (smoke)" begin
        env = E1Env(rng_seed = 99, max_blocks = 12, λ_time = 0.0)
        e1_reset!(env)
        # Cycle through IR actions (α = 180° = the 3rd α_set entry →
        # action indices 3, 6, 9, 12, 15, 18).
        ir_actions = 3:3:18
        terminal_err = 1.0
        for k in 1:12
            _, _, done, info = e1_step!(env, ir_actions[1 + (k-1) % 6])
            if done
                terminal_err = info["err"]
                break
            end
        end
        @test terminal_err < 0.05
    end

    @testset "simulate backend matches analytical within a few percent" begin
        env_a = E1Env(rng_seed = 7, max_blocks = 6, backend = :analytical)
        env_s = E1Env(rng_seed = 7, max_blocks = 6, backend = :simulate)
        e1_reset!(env_a); e1_reset!(env_s)
        @test env_a.T1_true == env_s.T1_true

        ir_actions = 3:3:18
        err_a, err_s = 1.0, 1.0
        for k in 1:6
            _, _, _, info_a = e1_step!(env_a, ir_actions[k])
            _, _, _, info_s = e1_step!(env_s, ir_actions[k])
            err_a = info_a["err"]; err_s = info_s["err"]
        end
        # Simulated backend is biased by RF duration (~0.5% offset in E0) —
        # accept generous tolerance on the gap.
        @test abs(err_a - err_s) < 0.05
    end
end
